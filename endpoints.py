import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template,jsonify,make_response
from pathlib import Path
import requests
import json


fe = FeatureExtractor()

class Endpoints:
     def __init__(self):
         
    
    def index():