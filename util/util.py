import numpy as np
import torch
import cv2

def tensor2img(image_tensor):
    
    mean= [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for tensor in image_tensor:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
  
    image_numpy = image_tensor.detach().to(torch.device('cpu')).numpy()    
    
    image_numpy = np.transpose(image_numpy, (0, 2, 3, 1)) * 255
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = image_numpy[...,::-1] # RGB2BGR
    
    return image_numpy.astype(np.uint8)