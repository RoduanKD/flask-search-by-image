from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
import os
import filetype

def extract_features_in_path(saved_images_path):
    feature_extractor = FeatureExtractor()
    print( " extract_features_in_path: "  + saved_images_path)
    orgin_path = os.getcwd()
    os.chdir(saved_images_path)
    for img_path in os.scandir(saved_images_path):
        # TODO: change to if file type is image
        if (filetype.is_image(img_path.path)):
            print('IMAGE PATH FOR BISHER: ------->>>>\t'+img_path.path)
            feature = feature_extractor.extract(img=Image.open(img_path.path))

            # new_file_path_after_training = img_path.name[ :-5] + "1" + img_path.name[-4:] #flip 0 to 1 to indicate image has been trained
            new_file_path_after_training = '../trained/' + img_path.name
            # print(img_path.name, new_file_path_after_training)
            os.rename( img_path.name,  new_file_path_after_training)
            feature_path = Path("../feature") / (new_file_path_after_training.split(".") [0]+ ".npy")
            np.save(feature_path, feature)
    os.chdir(orgin_path)

if __name__ == '__main__':
    extract_features_in_path("C:\\dev\\workspace\\python\\sis\\static\\img\\")
