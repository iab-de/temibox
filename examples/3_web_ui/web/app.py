import os
import json
import logging
from flask import Flask, send_from_directory, abort, request, make_response
from flask_cors import CORS

from temibox.pipeline import StandardPipeline
from domain import WebDocument

ASSET_DIR = f"{os.getcwd()}/examples/3_web_ui/assets"
MODEL_DIR = f"{os.getcwd()}/examples/3_web_ui/export"

# Configure logger
logger = logging.getLogger("temibox example")
logging.basicConfig()
logger.setLevel(logging.INFO)

# Initialize web app
app = Flask("temibox example")
CORS(app)

# Load pipeline
pipeline = StandardPipeline.load(folder = MODEL_DIR,
                                 suffix = "webmodel")

# Define endpoints
@app.route("/assets/<path:filename>")
def static_assets(filename: str):
    fs_path = f"{ASSET_DIR}/{filename}"
    if not os.path.isfile(fs_path):
        logger.warning(f"Could not find requested file '{filename}'")
        return abort(404)

    return send_from_directory(ASSET_DIR, filename)

@app.route("/")
def index():
    return static_assets("index.html")

@app.route("/api/predict", methods=["POST"])
def prediction():
    try:
        data = request.get_json()
        title = data["title"]
        text = data["text"]
    except Exception as e:
        return make_response("fehlende Daten", 400)

    with pipeline.modes(inference = True, cuda=True):
        preds = pipeline.predict(document = WebDocument(title=title, text = text))

    return make_response(json.dumps(preds[0].payload.to_dict("records")), 200)
