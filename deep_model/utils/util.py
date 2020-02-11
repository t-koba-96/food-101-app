import torch
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
import cv2
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def imshape(image):
    image=image/2+0.5
    npimg=image.numpy()
    return np.transpose(npimg,(1,2,0))



def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show



def normalize_heatmap(x):
    min = x.min()
    max = x.max()
    result = (x-min)/(max-min)
    return result


def show_predict(images,classes,net,device):

   images_gpu = images.to(device)
   outputs=net(images_gpu)
   _,predicted=torch.max(outputs,1)
   predicted=predicted.cpu()
   return classes[predicted[0]]


def show_attention(images,net,image_num,device): 

   images_gpu = images.to(device)
   at_outputs=net(images_gpu)
   at_predicted=at_outputs.cpu()
   attention=at_predicted.detach()
   img=imshape(images[image_num,:])

   #attention map
   heatmap = attention[image_num,:,:,:]
   heatmap = heatmap.numpy()
   heatmap = np.average(heatmap,axis=0)
   heatmap = normalize_heatmap(heatmap)
   # 元の画像と同じサイズになるようにヒートマップのサイズを変更
   heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
   #特徴ベクトルを256スケール化
   heatmap = np.uint8(255 * heatmap)
   # RGBに変更
   heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
   #戻す
   heatmap=heatmap/255
   # 0.5はヒートマップの強度係数
   s_img = heatmap * 0.5 + img

   #plt
   image_list=[img,heatmap,s_img]
   name_list=["Input","Attention","Result"]
   fig=plt.figure(figsize=(6, 2))
   for i,data in enumerate(image_list):
       fig.add_subplot(1, 3, i+1, title = name_list[i])
       plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                    bottom=False,
                    left=False,
                    right=False,
                    top=False)
       plt.imshow(data)
   graph=io.BytesIO()
   plt.savefig(graph,format="png",facecolor='moccasin')

   return graph