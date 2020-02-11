# Test app for food-101 using Flask.

<img width="1389" alt="スクリーンショット 2020-02-12 1 03 23" src="https://user-images.githubusercontent.com/38309191/74254281-9368a100-4d33-11ea-863f-cca55121ace9.png">



## Details  

・Demo app for [food-101 repository](https://github.com/t-koba-96/food-101). 

・You can see the demo example video at [demo page](https://t-koba-96.github.io/app/food-101-app/).

## Use it on your local (Need GPU)

・Clone this repo:

`$ git clone https://github.com/t-koba-96/food-101-app.git`  
`$ cd food-101-app`

・Setup virtual environment

`$ python3 -m venv <environment-name>`  
`$ source <environment-name>/bin/activate`

・Install required package  

`$ pip install -r requirements.txt`

・Install model weight

`$ sh get_weigth.sh `

・Launch the app  

`$ python app.py`

You can now see your website at local host  http://127.0.0.1:8050/ .  
It takes a while for first launch since it loads the model weight first.
