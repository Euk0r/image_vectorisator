
import sys
import numpy as np
#import cv2
import torch
from PIL import Image
import constants
import glob
import os

from building_footprint_segmentation.seg.binary.models import ReFineNet
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from building_footprint_segmentation.utils.py_network import (
    to_input_image_tensor,
    add_extra_dimension,
    convert_tensor_to_numpy,
)
from building_footprint_segmentation.utils.operations import handle_image_size

MAX_SIZE = 384

def cached_model():
    refine_net = ReFineNet()
    path = constants.WEIGHTS_PATH
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    refine_net.load_state_dict(checkpoint, strict=False)
    return refine_net

model = cached_model()

def process():
    tiles = []
    for file_name in sorted(glob.glob('tiles/'+'*.*')):
        tiles.append(file_name)
    if tiles:
        for tile in tiles:
            print(tile.split('\\')[-1])
            original_image = np.array(Image.open(tile))
            original_height, original_width = original_image.shape[:2]
            if (original_height, original_width) != (MAX_SIZE, MAX_SIZE):
                original_image = handle_image_size(original_image, (MAX_SIZE, MAX_SIZE))
            # Apply Normalization
            normalized_image = min_max_image_net(img=original_image)
            tensor_image = add_extra_dimension(to_input_image_tensor(normalized_image))
            with torch.no_grad():
                # Perform prediction
                prediction = model(tensor_image)
                prediction = prediction.sigmoid()
            prediction_binary = convert_tensor_to_numpy(prediction[0]).reshape(
                (MAX_SIZE, MAX_SIZE)
            )
            im1 = Image.fromarray(prediction_binary*255).convert("L")
            im1.save("tiles_segmented/" + tile.split('\\')[-1])

if __name__ == "__main__":
    process()  
