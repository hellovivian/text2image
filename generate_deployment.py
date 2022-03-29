# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

import math
import random
# from email.policy import default
from urllib.request import urlopen
from tqdm import tqdm
import sys
import os
import flask
# pip install taming-transformers doesn't work with Gumbel, but does not yet work with coco etc
# appending the path does work with Gumbel, but gives ModuleNotFoundError: No module named 'transformers' for coco etc
sys.path.append('taming-transformers')
from itertools import product


from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)	# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP, RAdam

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re
# Supress warnings
import warnings
warnings.filterwarnings('ignore')


# Various functions and classes
def sinc(x):
	return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
	cond = torch.logical_and(-a < x, x < a)
	out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
	return out / out.sum()


def ramp(ratio, width):
	n = math.ceil(width / ratio + 1)
	out = torch.empty([n])
	cur = 0
	for i in range(out.shape[0]):
		out[i] = cur
		cur += ratio
	return torch.cat([-out[1:].flip([0]), out])[1:-1]


class ReplaceGrad(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x_forward, x_backward):
		ctx.shape = x_backward.shape
		return x_forward

	@staticmethod
	def backward(ctx, grad_in):
		return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, min, max):
		ctx.min = min
		ctx.max = max
		ctx.save_for_backward(input)
		return input.clamp(min, max)

	@staticmethod
	def backward(ctx, grad_in):
		input, = ctx.saved_tensors
		return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


def vector_quantize(x, codebook):
	d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
	indices = d.argmin(-1)
	x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
	return replace_grad(x_q, x)


class Prompt(nn.Module):
	def __init__(self, embed, weight=1., stop=float('-inf')):
		super().__init__()
		self.register_buffer('embed', embed)
		self.register_buffer('weight', torch.as_tensor(weight))
		self.register_buffer('stop', torch.as_tensor(stop))

	def forward(self, input):
		input_normed = F.normalize(input.unsqueeze(1), dim=2)
		embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
		dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
		dists = dists * self.weight.sign()
		return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

#NR: Split prompts and weights
def split_prompt(prompt):
	vals = prompt.rsplit(':', 2)
	vals = vals + ['', '1', '-inf'][len(vals):]
	return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
	def __init__(self, cut_size, cutn, cut_pow=1.):
		super().__init__()
		self.cut_size = cut_size
		self.cutn = cutn
		self.cut_pow = cut_pow # not used with pooling
		
		# Pick your own augments & their order
		augment_list = []
		for item in augments[0]:
			if item == 'Ji':
				augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
			elif item == 'Sh':
				augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
			elif item == 'Gn':
				augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
			elif item == 'Pe':
				augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
			elif item == 'Ro':
				augment_list.append(K.RandomRotation(degrees=15, p=0.7))
			elif item == 'Af':
				augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
			elif item == 'Et':
				augment_list.append(K.RandomElasticTransform(p=0.7))
			elif item == 'Ts':
				augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
			elif item == 'Cr':
				augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
			elif item == 'Er':
				augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
			elif item == 'Re':
				augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
				
		self.augs = nn.Sequential(*augment_list)
		self.noise_fac = 0.1

		# Pooling
		self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
		self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

	def forward(self, input):
		cutouts = []
		
		for _ in range(self.cutn):            
			# Use Pooling
			cutout = (self.av_pool(input) + self.max_pool(input))/2
			cutouts.append(cutout)
			
		batch = self.augs(torch.cat(cutouts, dim=0))
		
		if self.noise_fac:
			facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
			batch = batch + facs * torch.randn_like(batch)
		return batch


def load_vqgan_model(config_path, checkpoint_path):
	global gumbel
	gumbel = False
	config = OmegaConf.load(config_path)
	if config.model.target == 'taming.models.vqgan.VQModel':
		model = vqgan.VQModel(**config.model.params)
		model.eval().requires_grad_(False)
		model.init_from_ckpt(checkpoint_path)
	elif config.model.target == 'taming.models.vqgan.GumbelVQ':
		model = vqgan.GumbelVQ(**config.model.params)
		model.eval().requires_grad_(False)
		model.init_from_ckpt(checkpoint_path)
		gumbel = True
	elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
		parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
		parent_model.eval().requires_grad_(False)
		parent_model.init_from_ckpt(checkpoint_path)
		model = parent_model.first_stage_model
	else:
		raise ValueError(f'unknown model type: {config.model.target}')
	del model.loss
	return model


def resize_image(image, out_size):
	ratio = image.size[0] / image.size[1]
	area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
	size = round((area * ratio)**0.5), round((area / ratio)**0.5)
	return image.resize(size, Image.LANCZOS)

# Set the optimiser
def get_opt(opt_name, opt_lr ,z):
	if opt_name == "Adam":
		opt = optim.Adam([z], lr=opt_lr)	# LR=0.1 (Default)
	elif opt_name == "AdamW":
		opt = optim.AdamW([z], lr=opt_lr)	
	elif opt_name == "Adagrad":
		opt = optim.Adagrad([z], lr=opt_lr)	
	elif opt_name == "Adamax":
		opt = optim.Adamax([z], lr=opt_lr)	
	elif opt_name == "DiffGrad":
		opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
	elif opt_name == "AdamP":
		opt = AdamP([z], lr=opt_lr)		    
	elif opt_name == "RAdam":
		opt = RAdam([z], lr=opt_lr)		    
	elif opt_name == "RMSprop":
		opt = optim.RMSprop([z], lr=opt_lr)
	else:
		print("Unknown optimiser. Are choices broken?")
		opt = optim.Adam([z], lr=opt_lr)
	return opt

"""
Takes in a latent 
"""
# Vector quantize
def synth(z):
	z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
	return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

"""
Writes the loss
synthesizes 
Saves the output
"""
@torch.no_grad()
def checkin(i, losses, z, output, output_dir = "", iterations = 1000):
	losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
	if i == iterations and sum(losses).item() >= 0.95:
		output = "garbage" +output
	tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
	out = synth(z)
	info = PngImagePlugin.PngInfo()
# 	info.add_text('comment', f'{prompts}')
	print('saving here: ' + output_dir + output)
	TF.to_pil_image(out[0].cpu()).save(output_dir + output, pnginfo=info) 	

"""
iii is the image
"""
def ascend_txt(z, pMs):
	out = synth(z)
	iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
	result = []
	for prompt in pMs:
		result.append(prompt(iii))
	return result # return loss


def train(i,z, opt, pMs, output, z_min, z_max, output_dir="", iterations = 1000):
	opt.zero_grad(set_to_none=True)
	lossAll = ascend_txt(z, pMs)
	
	if i % display_freq == 0:
		checkin(i, lossAll,z, output[:-4]+"time=0" +output[-4:], output_dir=output_dir, iterations = iterations)
	   
	loss = sum(lossAll)
	loss.backward()
	opt.step()
	
	with torch.no_grad():
		z.copy_(z.maximum(z_min).minimum(z_max))

cutn = 32
cut_pow = 1
optimizer = 'Adam'
torch.backends.cudnn.deterministic = True
augments = [['Af', 'Pe', 'Ji', 'Er']]
replace_grad = ReplaceGrad.apply
clamp_with_grad = ClampWithGrad.apply

cuda_device = 0
device = torch.device(cuda_device)
clip_model='ViT-B/32'
vqgan_config=f'checkpoints/vqgan_imagenet_f16_16384.yaml'
vqgan_checkpoint=f'checkpoints/vqgan_imagenet_f16_16384.ckpt'

# Do it
device = torch.device(cuda_device)
model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
jit = True if float(torch.__version__[:3]) < 1.8 else False
perceptor = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)

cut_size = perceptor.visual.input_resolution

replace_grad = ReplaceGrad.apply
clamp_with_grad = ClampWithGrad.apply
make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)
torch.backends.cudnn.deterministic = True
augments = [['Af', 'Pe', 'Ji', 'Er']]
optimizer = 'Adam'

step_size=0.1
cutn = 32
cut_pow = 1
seed = 64
display_freq=50






def generate(prompt_string, output_name,iterations = 100,  size=(128, 128), seed=16, width=128, height=128, output_dir=""):
	torch.cuda.empty_cache()
	pMs=[]
	prompts = [prompt_string]
	output = output_name
   
	for prompt in prompts:
		txt, weight, stop = split_prompt(prompt)
		embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
		pMs.append(Prompt(embed, weight, stop).to(device))

	learning_rate = 0.1

	# Output for the user
	print('Using device:', device)
	print('Optimising using:', optimizer)
	print('Using text prompts:', prompts)  
	print('Using seed:', seed)
   
	i = 0

	f = 2**(model.decoder.num_resolutions - 1)
	toksX, toksY = width // f, height // f
	sideX, sideY = toksX * f, toksY * f

	e_dim = model.quantize.e_dim
	n_toks = model.quantize.n_e
	z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
	z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]


	one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
	z = one_hot @ model.quantize.embedding.weight

	z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
	z_orig = z.clone()
	z.requires_grad_(True)

	opt = get_opt(optimizer, learning_rate,z)

	normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
									  std=[0.26862954, 0.26130258, 0.27577711])

	with tqdm() as pbar:
		while True:            

			# Training time
			train(i,z, opt, pMs, output_name, z_min, z_max, output_dir, iterations)

			# Ready to stop yet?
			if i == iterations:

				break

			i += 1
			pbar.update()


normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])



import flask
from flask import Flask, send_file, request, jsonify, g, render_template, url_for, stream_with_context
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
def create_app():


	app = Flask(__name__)

	app.config.from_mapping(
	SEND_FILE_MAX_AGE_DEFAULT = 0
	)
	return app


exporting_threads = {}


app = create_app()


exporting_threads = {}
app = create_app()
@app.route('/', methods=["POST"])
def evaluate():
    prompts = request.get_json(force=True)
    for prompt in prompts["prompts"]:
        print(prompt)
        prompt_pattern = r'([A-Za-z\s]+)( in the style of )([A-Za-z\s]+)'
        prompt_pattern_match = re.match(prompt_pattern, prompt)
        if prompt_pattern_match:  
            subject = prompt_pattern_match.group(1)
            style = prompt_pattern_match.group(3)
            
            output_name = f"{subject}_{style}_100.jpg"
            if output_name in os.listdir("../shared_images/"):
                print("prev generated not generating")
            else:
                generate(prompt, f"{subject}_{style}_100.jpg", output_dir = "../shared_images/")
        else:
            output_name = f"{prompt}_custom_100.jpg"
            if output_name in os.listdir("../shared_images/"):
                print("prev generated not generating")
            else:
                generate(prompt, f"{prompt}_custom_100.jpg", output_dir = "../shared_images/")
#             print("no match no generation")
#             print(prompt)
    return jsonify(prompts)

def run():
    app.run(host='0.0.0.0',port=8891, threaded=False, debug=True)
run()





