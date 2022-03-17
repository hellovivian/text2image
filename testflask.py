
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
@app.route('/', methods=["POST"])
def evaluate():
    prompts = request.get_json(force=True)
    print(prompts)
    return jsonify(prompts)

def run():
    app.run(host='0.0.0.0',port=8880, threaded=False, debug=True)
run()


