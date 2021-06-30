import os
import numpy as np
from albumentations import CLAHE,Compose
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset
import json 

# UTG_Dataset, YHG_Dataset

class UTG_Dataset():
    def __init__(self,phase):
        self.phase = phase
        self.png_root = '/SSD3/jumi/HandSemSeg/UTG/'+phase.lower()
        self.annot_root ='/SSD3/jumi/HandSemSeg/UTG/'+phase.lower()+'annot'
        self.fn = os.listdir(self.png_root)
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((480,640)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.tgt_transform = transforms.Compose([
                                                  transforms.Resize((480,640), interpolation=Image.NEAREST),
                                                  np.asarray,
                                                  torch.LongTensor
                                                 ])
        
    def __len__(self):
        return len(self.fn)

    def __getitem__(self, idx):
    
        img_name = self.fn[idx]
        img_name_ = 'UTG_'+ img_name
      
        png_img = cv2.imread(os.path.join(self.png_root, img_name))
        data = {"image":np.array(png_img)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        annot_img = Image.open(os.path.join(self.annot_root, img_name))
        
        png = self.img_transform(png_img)
        annot = self.tgt_transform(annot_img)
        annot = annot/255
                                  
        return png,annot,img_name_

class YHG_Dataset():
    def __init__(self,phase):
        self.phase = phase
        self.png_root = '/SSD3/jumi/HandSemSeg/Yale_Human_Grasp/'+phase.lower()
        self.annot_root ='/SSD3/jumi/HandSemSeg/Yale_Human_Grasp/'+phase.lower()+'annot'
        self.fn = os.listdir(self.png_root)
        self.clahe = CLAHE(p=0.8)
        self.img_transform = transforms.Compose([
                                                  transforms.Resize((480,640)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
        self.tgt_transform = transforms.Compose([
                                                  transforms.Resize((480,640), interpolation=Image.NEAREST),
                                                  np.asarray,
                                                  torch.LongTensor
                                                 ])
        
    def __len__(self):
        return len(self.fn)

    def __getitem__(self, idx):
    
        img_name = self.fn[idx]
        img_name_ = 'YHG_'+ img_name
      
        png_img = cv2.imread(os.path.join(self.png_root, img_name))
        data = {"image":np.array(png_img)}
        aug = self.clahe(**data)
        png_img = aug["image"]
        png_img = Image.fromarray(png_img)
        annot_img = Image.open(os.path.join(self.annot_root, img_name))
        
        png = self.img_transform(png_img)
        annot = self.tgt_transform(annot_img)
        annot = annot/255
                                  
        return png,annot,img_name_
