import argparse
import os
import numpy as np
from albumentations import CLAHE,Compose
from modeling.deeplab import *
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import cv2 


class Inferer():
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.model = DeepLab(num_classes=2,
                    backbone='resnet',
                    output_stride=8)
        ckpt = torch.load('run/updated/model_best.pth')['state_dict']
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((480,640)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])

    def predict(self,frame):
        h,w,c = frame.shape
        data = {"image":(frame)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        png = self.img_transform(png_img)
        png = png.unsqueeze(0)
        png = png.to(self.device)
        outputs = self.model(png)
        _,prediction = torch.max(outputs,1)
        prediction = prediction[0].cpu().clone().numpy()
        pred_hand = cv2.resize(prediction*255,(w,h),interpolation=cv2.INTER_NEAREST).astype('uint8')
        pred_hand_ = np.stack((pred_hand,pred_hand,pred_hand),axis=2)
        dst = cv2.addWeighted(frame, 0.3, pred_hand_, 0.7, 0)
        return dst




if __name__ == '__main__':
    INFERER = Inferer()
    cap = cv2.VideoCapture(0)
    while(True):
        ret,frame = cap.read()
        if ret:
            result = INFERER.predict(frame)
            cv2.imshow('Overlay',result)
        if cv2.waitKey(33)>0:
            break
    cv2.destroyAllWindows()

