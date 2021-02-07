import torch.utils.data as data
import json
import os
import cv2
import torch
import numpy as np


class MaskDataset(data.Dataset):
    
    def __init__(self, args, transform, json_path):
             
        if isinstance(json_path, str):
            json_path = [json_path]
        
        self.args = args
        self.data = []
        self.transform = transform
        self.label_reflection = None
        for file_path in json_path:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.data += data
              
    
    def __len__(self):
        return len(self.data)
    
        
    def __getitem__(self, index):
        
        image = cv2.imread(self.data[index][0])
        label = cv2.imread(self.data[index][1])
        image_ori = image.copy()
        
        if not self.label_reflection is None:
            label = self.label_reflection[label]
                    
        if self.transform is not None:
            image, label = self.transform(image, label)
                        
        return image, label
        
