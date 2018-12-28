import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import utils

#Command Line Arguments

ap = argparse.ArgumentParser(description='predict-file')
# Input image file path
ap.add_argument('input_img', nargs='*', action="store", default="./flowers/test/1/image_06752.jpg")
# Checkpoint directory
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store")
# Probability most likely classes
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
#Category of Image 
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
# Device processor type
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")


# Define the command line arguements

pa = ap.parse_args()

path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
path = pa.checkpoint

# load the trained model
model=utils.load_checkpoint(path)

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

# check the probability of the image
probabilities = utils.predict(path_image, model, number_of_outputs, power)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])
i=0
while i < number_of_outputs:
    print("{} 's  probability is {}".format(labels[i], probability[i]))
    i += 1
    