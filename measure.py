from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
import scipy.misc


def readImage():
    def openImage(path):
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print("nice try, come again!")
            return False
    img = False
    while (img is False ):
        path = input("Enter image path: ")
        img = openImage(path)
    return img

resnet = InceptionResnetV1(pretrained='vggface2').eval()
img1 = readImage()
img2 = readImage()
mtcnn = MTCNN(keep_all=True,prewhiten=True)
img1 = mtcnn(img1)
img2 = mtcnn(img2)
e1 = resnet(img1)
e2 = resnet(img2)

print("Distance: ")
print((e1 - e2).norm().item())
