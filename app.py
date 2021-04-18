import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template,jsonify,make_response
from pathlib import Path
import requests
import json


app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]
       
            
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


@app.route('/detect',methods=['POST'])
def detect():
    request_data = request.get_json()
    image_path = None
    if request_data:
        if 'image' in request_data:
            img_path = request_data['image']
            response = requests.get(img_path)

            file = open("img.png", "rb+")
            file.write(response.content)


            
            
            # Save query image
            img = Image.open(file)  # PIL image
            # uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + img
            # img.save(uploaded_img_path)

            # Run search
            query = fe.extract(img)
            dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
            ids = np.argsort(dists)[:30]  # Top 30 results
            scores = [(dists[id], img_paths[id]) for id in ids]
            
            # Store results in a dictionary
            results = []

            for item in scores:
                results.append({
                    "filename" : str(item[1]),
                    "uncertainty": str(item[0])
                })
            
            print(jsonify(results))
            response = make_response(
                    jsonify(results),
                    200,
                )
            response.headers["Content-Type"] = "application/json"
            return response
    
        else:
             response = make_response(
                    jsonify({"error" : "Please use the image param in the body"}),
                    400,
                )
             response.headers["Content-Type"] = "application/json"
             return response 

    else: 
        response = make_response(
                    jsonify({"error" : "This endpoint accepts param named: image only,"}),
                    400,
                )
        response.headers["Content-Type"] = "application/json"
        return response 
        
        

if __name__=="__main__":
    app.run(debug=False,host='127.0.0.1')
