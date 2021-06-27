from typing import final
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import product
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template, jsonify, make_response
from pathlib import Path
import requests
import json
from image_util import download_images_parallel_starting_point as images_downloader
from offline import extract_features_in_path
import os
import constants
app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path(constants.FEATURED).glob("*.npy"):
    features.append(np.load(feature_path))
    # img_paths.append(Path("./static/img") / (feature_path.stem))
    img_paths.append(feature_path.stem) #TODO change to product ID 
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":",
                                                                                    ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        # L2 distances to features
        dists = np.linalg.norm(features-query, axis=1)
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

@app.route('/api/v1/detect', methods=['POST'])
def detect():
    request_data = request.get_json()
    image_path = None
    if request_data:
        if 'image' in request_data:
            img_path = request_data['image'] #json path
            response = requests.get(img_path)
            os.chdir(constants.CACHE_DIR)
            #TODO: Idea, rename files to cache imgs
            file = open("img.png", "rb+")
            file.write(response.content)

            # Save query image
            img = Image.open(file)  # PIL image


            # Run search
            query = fe.extract(img)
            # L2 distances to features
            dists = np.linalg.norm(features-query, axis=1)
            ids = np.argsort(dists)[:30]  # Top 30 results
            scores = [(dists[id], img_paths[id]) for id in ids]


            results = []

            for item in scores:
                try:
                    arr = str(item[1]).split('_')
                    results.append({
                    "product_id": str(arr[0]),
                    "color_id": str(arr[1]),
                    "size_id": str(arr[2])
                    })
                except:
                    print("Error: Image\t" + item[1] +  "is not following the convention")

            response = make_response(
                jsonify(results),
                200,
            )
            response.headers["Content-Type"] = "application/json"
            return response

        else:
            response = make_response(
                jsonify({"error": "Please use the image param in the body"}),
                400,
            )
            response.headers["Content-Type"] = "application/json"
            return response

    else:
        response = make_response(
            jsonify({"error": "This endpoint accepts param named: image only,"}),
            400,
        )
        response.headers["Content-Type"] = "application/json"
        return response


#NOT DONE
@app.route('/api/v1/train', methods=['POST'])
def train():
    request_data = request.get_json()
    if request_data:
        if 'images' in request_data:
            #1 - download_images
            images_downloader(request_data['images'])
            # os.chdir(os.getcwd()+"/static/img")
            #TODO: revisit this function
           
            print(constants.INFERENCE_QUEUE_DIR)
            extract_features_in_path(constants.INFERENCE_QUEUE_DIR)

            response = make_response(
                jsonify({"training_status":"done"}),
                200,
            )
            response.headers["Content-Type"] = "application/json"
            return response

        else:
                response = make_response(
                    jsonify({"error": "Please use the images paramter in the body"}),
                    400,
                )
                response.headers["Content-Type"] = "application/json"
                return response

    else:
            response = make_response(
                    jsonify({"error" : "This endpoint requires request paramter named: images only, and dosen't accept empty body requests"}),
                    400,
                )
            response.headers["Content-Type"] = "application/json"
            return response


@app.errorhandler(404)
def not_found(e):
    """Page not found."""
    # return make_response(render_template("error.html"), 404)
    return render_template("error.html")


if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1')
