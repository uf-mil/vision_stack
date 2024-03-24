import os
import torch
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import datetime

from Layer import PreprocessLayer

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '../ml/weights/model_best_2842.pth.tar')

class UnderWaterImageEnhancementLayer(PreprocessLayer):
    def __init__(self, size) -> None:
        """
        Passes the image through an underwater image enhancement generation AI model.
        """
        super().__init__(size, "underwaterImageEnhancement")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PhysicalNN()
        self.model = torch.nn.DataParallel(model).to(self.device)
        self.checkpoint = torch.load(WEIGHTS_PATH, map_location=self.device)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        print("=> loaded model at epoch {}".format(self.checkpoint['epoch']))
        self.model = self.model.module
        self.model.eval()

        self.testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.unloader = transforms.ToPILImage()
    
    def process(self, image):
        starttime = datetime.datetime.now()
        img = Image.fromarray(image)
        inp = self.testtransform(img).unsqueeze(0)
        inp = inp.to(self.device)
        out = self.model(inp)
       
        # Place result images in directory
        corrected = self.unloader(out.cpu().squeeze(0))
        print(type(corrected))
        endtime = datetime.datetime.now()
        print(endtime-starttime)
        img_array = np.array(corrected)
        return (img_array, None)

class AConvBlock(nn.Module):
    def __init__(self):
        super(AConvBlock,self).__init__()

        block = [nn.Conv2d(3,3,3,padding = 1)]
        block += [nn.PReLU()]

        block += [nn.Conv2d(3,3,3,padding = 1)]
        block += [nn.PReLU()]

        block += [nn.AdaptiveAvgPool2d((1,1))]
        block += [nn.Conv2d(3,3,1)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(3,3,1)]
        block += [nn.PReLU()]
        self.block = nn.Sequential(*block)

    def forward(self,x):
        return self.block(x)

class tConvBlock(nn.Module):
    def __init__(self):
        super(tConvBlock,self).__init__()

        block = [nn.Conv2d(6,8,3,padding=1,dilation=1)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(8,8,3,padding=2,dilation=2)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(8,8,3,padding=5,dilation=5)]
        block += [nn.PReLU()]

        block += [nn.Conv2d(8,3,3,padding=1)]
        block += [nn.PReLU()]
        self.block = nn.Sequential(*block)
    def forward(self,x):
        return self.block(x)

class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN,self).__init__()

        self.ANet = AConvBlock()
        self.tNet = tConvBlock()

    def forward(self,x):
        A = self.ANet(x)
        t = self.tNet(torch.cat((x*0+A,x),1))
        out = ((x-A)*t + A)
        return torch.clamp(out,0.,1.)
