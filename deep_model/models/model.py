import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

# attention layer with ReLu
class attention_ReLu(nn.Module):
    def __init__(self):
        super(attention_ReLu, self).__init__()
        
        self.conv1 = nn.Conv2d(512,512,3,1,1)
        self.conv2 = nn.Conv2d(512,1,3,1,1)


    def forward(self, x):
               
        y = F.relu(self.conv2(x))
        x = F.relu((self.conv1(x))*y)

        return x
    

# attention layer with sigmoid
class attention_sigmoid(nn.Module):
    def __init__(self):
        super(attention_sigmoid, self).__init__()
        
        self.conv1 = nn.Conv2d(512,512,3,1,1)
        self.conv2 = nn.Conv2d(512,512,3,1,1)


    def forward(self, x):
               
        y = torch.sigmoid(self.conv2(x))
        x = F.relu((self.conv1(x))*y)

        return x


#vgg with attention layer
class at_vgg(nn.Module):
    def __init__(self):
        super(at_vgg, self).__init__()
        features=[]
        for i in range(26):
           features.append((vgg16(pretrained = True).features)[i])
        
        #attention_ReLu() or attention_sigmoid()
        features.append(attention_sigmoid())
        
        for i in range(3):
           features.append((vgg16(pretrained = True).features)[i+28])
        
        self.features = nn.ModuleList(features)
        
        
        #GAP 今回使わない
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        
        classifier=[]
        
        for i in range(6):
           classifier.append((vgg16(pretrained = True).classifier)[i])
        
        classifier.append(nn.Linear(4096,101))
        
        self.classifier = nn.ModuleList(classifier)
    

    def forward(self, x):
        
        for ii,model in enumerate(self.features):
            x = model(x)
            
        x = x.view(x.size(0), -1)
        
        for ii,model in enumerate(self.classifier):
            x = model(x)

        return x

# attention network
class attention_net(nn.Module):
    def __init__(self,net):
        super(attention_net, self).__init__()
        features=[]
        for i in range(27):
           features.append((net.features)[i])

        self.features = nn.ModuleList(features)

    def forward(self, x):
        
        for ii,model in enumerate(self.features):
           
          if ii in {26}:
               #sigmoid or ReLu
               y = torch.sigmoid(model.conv2(x))
          else:
               x = model(x)

        return y