from basic_part import *
from basic_args import obtain_basic_args
import torch
import torch.distributed as dist
import datetime
from dataset.dataloader import get_dataloader
from loss.loss import *
from util.meter import AverageMeter
from util.show_seg import show_seg
import torch.distributed as dist
from logger import Logger
from tensorboardX import SummaryWriter
from copy import deepcopy
import numpy as np
from util.util import tensor2img
import cv2
import os
import dataset.transforms as transforms
from util.util import tensor2img

img_dir = '/search/speech/xz/datasets/segment/CelebAMask-HQ/test_img'
save_dir = 'temp_img'
cp_path = 'snapshots/checkpoint/epoch-2-50.pth'

checkpoint  = torch.load(cp_path)
args = checkpoint['args']

net = get_model(args)
net.load_state_dict(checkpoint['state_dict'])
net.cuda()
net.eval()

eval_transform = transforms.Compose([
    transforms.TrainScale2WH((args.input_width, args.input_height)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

for img_path in os.listdir(img_dir):
    
    print(img_path)
    
    img = cv2.imread(os.path.join(img_dir, img_path))
    img_name = img_path.split('/')[-1].split('.')[0]
 
    
    img_tensor, _ = eval_transform(img, None)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    
    img_tensor = img_tensor.cuda()
    with torch.no_grad():
        out, _, _ = net(img_tensor)
    
    img_np = tensor2img(img_tensor)[0]
    out_np = out.cpu().numpy().argmax(1)[0]
    # print(out.cpu().numpy()[0, 0])
    h, w, _ = img_np.shape
    canvas = np.zeros((h, w*2, 3))   
    prediction = show_seg(img_np, out_np, args.nclass)
    
    canvas[:,:w,:] = img_np    
    canvas[:,w:2*w,:] = prediction
    
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format(img_name)), canvas)




  