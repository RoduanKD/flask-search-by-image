from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os

def extract_features_in_path(saved_images_path):
    feature_extractor = FeatureExtractor()
    print( " extract_features_in_path: "  + saved_images_path)
    orgin_path = os.getcwd()
    os.chdir(saved_images_path)
    for img_path in os.scandir(saved_images_path):
        if (img_path.path.endswith("jpg")
                or img_path.path.endswith("png") and img_path.is_file()):
            print('IMAGE PATH FOR BISHER: ------->>>>\t'+img_path.path)
            feature = feature_extractor.extract(img=Image.open(img_path.path))

            new_file_path_after_training = img_path.name[ :-5] + "1" + img_path.name[-4:] #flip 0 to 1 to indicate image has been trained
            os.rename( img_path.name,  new_file_path_after_training)
            feature_path = Path("../feature") / (new_file_path_after_training.split(".") [0]+ ".npy")
            np.save(feature_path, feature)
    os.chdir(orgin_path)

if __name__ == '__main__':
    extract_features_in_path("C:\\dev\\workspace\\python\\sis\\static\\img\\")
