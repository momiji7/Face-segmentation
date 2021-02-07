import numpy as np
import cv2


# atts_list = ['bg', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
#        'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']  

# (0, 0, 0) 代表非预测的区域

colour_panel = [(255,255,255),      (0,0,128),      (0,0,255),
                (0,128,0),    (0,128,128),    (0,128,255),
                (0,255,0),    (0,255,128),    (0,255,255),
                (128,0,0),    (128,0,128),    (128,0,255),
                (128,128,0),  (128,128,128),  (128,128,255),
                (128,255,0),  (128,255,128),  (128,255,255),
                (255,0,0),    (255,0,128),    (255,0,255),    
               ]



def show_seg(img, seg, nclass, mode = None):
    
    h, w = seg.shape[0], seg.shape[1]
    
    if seg.ndim < 3 or seg.shape[2] == 1:
        seg_t = np.zeros((h, w, 3))
        seg_t[:,:,0], seg_t[:,:,1], seg_t[:,:,2] = seg, seg, seg
        seg = seg_t
    
    if mode == None:
        img_mask_col = np.zeros((h, w, 3))
    elif mode == 'add':
        img_mask_col = img.copy()
        img_mask_col = cv2.resize(img_mask_col, (w, h))

    for i in range(nclass):
        
        mask = np.where(seg == i, 1, 0) 
        colours_map = np.zeros((h, w, 3))
        colours_map[:,:,0], colours_map[:,:,1], colours_map[:,:,2] =  colour_panel[i][0], colour_panel[i][1], colour_panel[i][2]
        
        if mode == None:
            img_mask_col = colours_map * mask + (1 - mask) * img_mask_col
        elif mode == 'add':
            img_mask_col = colours_map * mask + img_mask_col

    return img_mask_col
    
    

