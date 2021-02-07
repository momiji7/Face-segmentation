import torch
import sys, math
import numpy as np
import numbers
import cv2
import random


class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img, label):
    for t in self.transforms:
      img, label = t(img, label)
    return img, label


class Normalize(object):
  """Normalize an tensor image with mean and standard deviation.
  Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  Args:
    mean (sequence): Sequence of means for (R, G, B) channels respecitvely.
    std (sequence): Sequence of standard deviations for (R, G, B) channels
      respecitvely.
  """

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensors, label):
    """
    Args:
      tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
      Tensor: Normalized image.
    """
    # TODO: make efficient
    if isinstance(tensors, list): is_list = True
    else:                         is_list, tensors = False, [tensors]

    for tensor in tensors:
      for t, m, s in zip(tensor, self.mean, self.std):
        t.sub_(m).div_(s)
    
    if is_list == False: tensors = tensors[0]

    return tensors, label




class TrainScale2WH(object):

  # Rescale the input image to the given size.
 

  def __init__(self, target_size):
    assert isinstance(target_size, tuple) or isinstance(target_size, list), 'The type of target_size is not right : {}'.format(target_size)
    assert len(target_size) == 2, 'The length of target_size is not right : {}'.format(target_size)
    assert isinstance(target_size[0], int) and isinstance(target_size[1], int), 'The type of target_size is not right : {}'.format(target_size)
    self.target_size   = target_size

  def __call__(self, img, label):


    ow, oh = self.target_size[0], self.target_size[1]
    img = cv2.resize(img, (ow, oh), interpolation = cv2.INTER_LINEAR)
    
    if label is not None:
        label = cv2.resize(label, (ow, oh), interpolation = cv2.INTER_NEAREST)

    return img, label


class AugScale(object):
    
  # Rescale the input image to the given size. Data Augmentation 
  # 'interpolation =' 不能少

  def __init__(self, scales = (1, )):
    
    self.scales = scales

  def __call__(self, img, label):
    
    scale = random.choice(self.scales)
    
    img_h, img_w, _ = img.shape    
    img_nh, img_nw = int(scale*img_h), int(scale*img_w)  
    img = cv2.resize(img, (img_nw, img_nh), interpolation = cv2.INTER_LINEAR)
    
    if label is not None:
        l_h, l_w, _ = label.shape
        l_nh, l_nw = int(scale*l_h), int(scale*l_w) 
        label = cv2.resize(label, (l_nw, l_nh), interpolation = cv2.INTER_NEAREST)
    
    return img, label



class AugCrop(object):

  def __init__(self, crop_x, crop_y, center_perterb_max, fill=0):
    assert isinstance(crop_x, int) and isinstance(crop_y, int) and isinstance(center_perterb_max, numbers.Number)
    self.crop_x = crop_x
    self.crop_y = crop_y
    self.center_perterb_max = center_perterb_max
    assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
    self.fill   = fill

  def __call__(self, img, label):
    ## AugCrop has something wrong... For unsupervised data
    ## 在图像加干扰并且固定到一个定大小
    ## label和img size必须一样
    
    (h, w) = img.shape[:2]
    ch, cw = h/2, w/2
    dice_x, dice_y = random.random(), random.random()
    x_offset = int( (dice_x-0.5) * 2 * self.center_perterb_max)
    y_offset = int( (dice_y-0.5) * 2 * self.center_perterb_max)

    x1 = int(round( cw + x_offset - self.crop_x / 2. ))
    y1 = int(round( ch + y_offset - self.crop_y / 2. ))
    x2 = x1 + self.crop_x
    y2 = y1 + self.crop_y

    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        pad = max(0-x1, 0-y1, x2-w+1, y2-h+1)
        assert pad > 0, 'padding operation in crop must be greater than 0'
        img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=self.fill)
        if label is not None:
            label = cv2.copyMakeBorder(label,pad,pad,pad,pad,cv2.BORDER_CONSTANT)
        
        # 调整到新图像的原点坐标， 旧图像的原点在新图像中是(pad, pad)    
        x1, x2, y1, y2 = x1 + pad, x2 + pad, y1 + pad, y2 + pad
      
    
    # 正常的crop
    img = img[y1:y2,x1:x2,:]
    
    if label is not None:
        label = label[y1:y2,x1:x2,:]

    
    return img, label



class AugHorizontalFlip(object):


  def __init__(self, flip_prob):
    assert isinstance(flip_prob, numbers.Number)   
    self.flip_prob = flip_prob

  def __call__(self, img, label):
    
    # atts_list = ['bg', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    #    'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']      

    dice = random.random()
    if dice > self.flip_prob:
      return img, label
    
    img = cv2.flip(img, 1)
    
    if label is not None: 
        label_ori = cv2.flip(label, 1)
        label = label_ori.copy()
        label[label_ori==2] = 3
        label[label_ori==3] = 2
        label[label_ori==4] = 5
        label[label_ori==5] = 4
        label[label_ori==7] = 8
        label[label_ori==8] = 7

    return img, label


# copy from https://github.com/CoinCheung/BiSeNet/blob/master/lib/transform_cv2.py
class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, img, label):
        
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            img = self.adj_brightness(img, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            img = self.adj_contrast(img, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            img = self.adj_saturation(img, rate)
        return img, label

    def adj_saturation(self, img, rate):
        M = np.float32([ # BGR
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = img.shape
        img = np.matmul(img.reshape(-1, 3), M).reshape(shape)/3
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def adj_brightness(self, img, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[img]

    def adj_contrast(self, img, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[img]


class ToTensor(object):
  """
  numpy.ndarray (H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """
  def __call__(self, img, label):
 
    pic = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #opencv内置 numpy 先调整色彩通道，再调整维度
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    img = img.float().div(255)
    
    if label is not None:
        # label = torch.from_numpy(label.transpose((2, 0, 1))).long()
        label = torch.from_numpy(label[:,:,0]).long()
      
    return img, label





