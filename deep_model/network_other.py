# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import json
import numpy as np
from PIL import Image

def predict(img):
    resnet = models.resnet50(pretrained=True)
    resnet.eval()
    normalize = transforms.Normalize(
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             normalize
                             ]) 
    class_index = json.load(open('imagenet_class_index.json', 'r'))  
    labels = {int(key):value for (key, value) in class_index.items()}

    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    
    out = resnet(img_tensor)

    out = nn.functional.softmax(out, dim=1)
    out = out.data.numpy()

    num = np.argsort(-out)
    label_1 = labels[num[0,0]]
    label_2 = labels[num[0,1]]
    label_3 = labels[num[0,2]]

    return  label_1[1],label_2[1],label_3[1]