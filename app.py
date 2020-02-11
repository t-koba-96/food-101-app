# -*- coding: utf-8 -*-
import os.path
import io , base64  
import cv2
import numpy as np
import torch
from PIL import Image
from deep_model import network_other,network_food
from deep_model.models import model
from flask import Flask, jsonify, abort, make_response,render_template,url_for,request,redirect,send_file

app = Flask(__name__)
at_vgg = None
at_net = None

def load_model():
    global at_vgg,at_net
    print("loading model weight . . .")
    #classification model
    at_vgg = model.at_vgg()
    at_vgg.load_state_dict(torch.load("vgg_at.pth"))
    #attention model
    at_net=model.attention_net(at_vgg)
    print("weight loaded!")

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/result',methods = ['post'])
def posttest():
    img_file = request.files['img_file']

    # パスの取得
    fileName = img_file.filename

    # 名前と拡張子に分割
    root, extension = os.path.splitext(fileName)

    # 拡張子の制限
    extension = extension.lower()
    types = set([".jpg", ".jpeg", ".png"])
    if extension not in types:
        return render_template('index.html',message = "Chose an Image (JPG or PNG) ",color = "red")

    # 空のインスタンス作成
    start = io.BytesIO()
    end = io.BytesIO()

    if request.form['task'] == 'foodnet':
        
        # classification
        rgb_image = Image.open(img_file).convert("RGB")
        food_label,food_img = network_food.predict(rgb_image,at_vgg,at_net)

        #b64型に変換
        s_b64str = base64.b64encode(food_img.getvalue()).decode("utf-8")
        s_b64data = "data:image/png;base64,{}".format(s_b64str)

        return render_template('food.html' ,f_img = s_b64data ,f_label = food_label)
        
    elif request.form['task'] == 'imagenet':

        # PILで読み込む
        rgb_image = Image.open(img_file).convert("RGB")
        label_1,label_2,label_3 = network_other.predict(rgb_image)
        #空のインスタンスに保存
        rgb_image.save(start, 'png')
        #b64型に変換
        s_b64str = base64.b64encode(start.getvalue()).decode("utf-8")
        s_b64data = "data:image/png;base64,{}".format(s_b64str)

        return render_template('others.html' ,s_img = s_b64data ,cl_label_1 = label_1 ,cl_label_2 = label_2,cl_label_3 = label_3)



# errors
@app.errorhandler(400)
def noimage_error(error):
    return render_template('index.html',message = "Choose both image and processing type!",color = "red")
@app.errorhandler(413)
def size_error(error):
    return render_template('index.html',message = "Image size too big!",color = "red")
@app.errorhandler(503)
def other_error(error):
     return 'InternalServerError\n', 503

# 実行
if __name__ == '__main__':
    load_model()
    app.debug = False
    app.run(port = "8050")