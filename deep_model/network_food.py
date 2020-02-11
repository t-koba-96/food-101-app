import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from deep_model.utils import util,datas
from deep_model.models import model

import json
import numpy as np
from PIL import Image


def predict(img,at_vgg,at_net):
    device=torch.device('cuda')

    net=at_vgg.to(device)
    at_net=at_net.to(device)
    at_vgg.eval()
    at_net.eval()

    #classes
    classes=datas.class_list()

    normalize = transforms.Normalize(
                             mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    preprocess = transforms.Compose([
                             transforms.Resize(224),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             normalize
                             ]) 

    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)

    #label
    food_label=util.show_predict(img_tensor,classes,at_vgg,device)

    #attention
    food_img=util.show_attention(img_tensor,at_net,0,device)

    return food_label,food_img  





