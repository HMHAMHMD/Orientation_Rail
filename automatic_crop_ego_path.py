import os
import json
import numpy as np
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from PIL import Image

from src.utils.interface import Detector
from src.utils.visualization import draw_ego_path

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

base_path = os.path.dirname(__file__)


classification_model_path = os.path.join(base_path, "weights", "fortuitous-goat-12")
regression_model_path = os.path.join(base_path, "weights", "chromatic-laughter-5")
segmentation_model_path = os.path.join(base_path, "weights", "twinkling-rocket-21")

imgs_pths = list(os.listdir(r"C:\Projet_Datascientest_v2\rs19_val\jpgs\rs19_val"))

for idx,img_pth in enumerate(imgs_pths):
    img_path = os.path.join(r"C:\Projet_Datascientest_v2\rs19_val\jpgs\rs19_val", img_pth)
    img = Image.open(img_path)
    detector = Detector(classification_model_path, "auto", "pytorch", device) # the mthode can be changed 
    for i in range(50):  # multiple iterations to get a stable crop
        crop_coords = detector.get_crop_coords()
        ego_path = detector.detect(img) 
    vis = draw_ego_path(img, ego_path, crop_coords=crop_coords)

     # Extract the original file name without extension
    img_name = os.path.splitext(os.path.basename(img_pth))[0]
    
    # create json
    vis.save(os.path.join("C:\Projet_Datascientest_v2\ego_path_jpg_classification", f"{img_name}.jpg"))

    # Assuming ego_path is an Image object
    ego_path_np = np.array(ego_path)  # Convert Image to NumPy array
    
    with open(f"C:\Projet_Datascientest_v2\ego_path_json_classification\{img_name}.json", "w") as json_file:
        json.dump(ego_path_np.tolist(), json_file)


