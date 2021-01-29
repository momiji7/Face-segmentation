import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.resnet import resnext50_32x4d

class SpatialPath(nn.Module):
    
    
    def __init__(self, args):
        super(SpatialPath, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        
        return x
    
class ARM(nn.Module):
    
    def __init__(self, in_c):
        super(ARM, self).__init__()
        
        out_c = in_c
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False)
        self.bn      = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        
        y = self.avgpool(x)
        y = self.conv1x1(y)
        y = self.bn(y)
        
        return F.sigmoid(y) * x
    
class FFM(nn.Module):
    
    def __init__(self, in_c, out_c):
        super(FFM, self).__init__()
           
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_1 = nn.Conv2d(out_c, out_c // 4, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(out_c // 4, out_c, kernel_size=1, stride=1, bias=False)
        
        
    def forward(self, sp_feat, cp_feat):
        
        x = torch.cat([sp_feat, cp_feat], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        
        y = self.avgpool(x)
        y = F.sigmoid(self.conv1x1_2(F.relu(self.conv1x1_1(y))))
        
        out = x + x * y
        
        return out

class ContextPath(nn.Module):
    
    def __init__(self, args, arm_channel):
        super(ContextPath, self).__init__()
        
        self.resnext50_32x4d = resnext50_32x4d(pretrained=True)
        self.arm16 = ARM(arm_channel[0][0])
        self.arm32 = ARM(arm_channel[1][0])
        
        self.conv16 = nn.Conv2d(arm_channel[0][0], arm_channel[0][1], kernel_size=1, stride=1)
        self.bn16   = nn.BatchNorm2d(arm_channel[0][1])
        
        self.conv32 = nn.Conv2d(arm_channel[1][0], arm_channel[1][1], kernel_size=1, stride=1)
        self.bn32   = nn.BatchNorm2d(arm_channel[1][1])
        
        
        self.deconv_avgpool    = nn.ConvTranspose2d(arm_channel[1][0], arm_channel[1][1], kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv_avgpool_bn = nn.BatchNorm2d(arm_channel[1][1])
        self.deconv_32down     = nn.ConvTranspose2d(arm_channel[1][1], arm_channel[0][1],  kernel_size=4, stride=2, padding=1, bias=False)
        
        self.avg_h, self.avg_w = args.input_height // 32, args.input_width // 32
        
    def forward(self, x):
        
        x_16down, x_32down, x_avgpool = self.resnext50_32x4d(x)
                
        x_16down_arm = self.arm16(x_16down)
        x_32down_arm = self.arm32(x_32down)
        
        x_16down_arm = F.relu(self.bn16(self.conv16(x_16down_arm)))
        x_32down_arm = F.relu(self.bn32(self.conv32(x_32down_arm)))
        
        x_avgpool = F.relu(self.deconv_avgpool_bn(self.deconv_avgpool(x_avgpool)))
        
        # mode='bilinear', align_corners=True
        x_avgpool_up = F.interpolate(x_avgpool, (self.avg_h, self.avg_w))
        x_32down_up  = F.interpolate(x_avgpool_up + x_32down_arm, scale_factor = 2, mode='bilinear', align_corners=True)
        
        x_8down = F.interpolate(x_16down_arm + x_32down_up, scale_factor = 2, mode='bilinear', align_corners=True)
        
        return x_8down, x_16down_arm, x_32down_arm
        


class BiseNet(nn.Module):
    
    def __init__(self, args):
        super(BiseNet, self).__init__()
        
        self.arm_channel = [(1024, 128), (2048, 128)]
               
        self.cp = ContextPath(args, self.arm_channel)
        self.sp = SpatialPath(args)
        
        self.ffm = FFM(self.arm_channel[0][1]*2, self.arm_channel[0][1])
        
        self.conv_feat8  = nn.Conv2d(self.arm_channel[0][1], args.nclass, kernel_size=1, stride=1, bias=False)
        self.conv_feat16 = nn.Conv2d(self.arm_channel[0][1], args.nclass, kernel_size=1, stride=1, bias=False)
        self.conv_feat32 = nn.Conv2d(self.arm_channel[1][1], args.nclass, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        
        sp_feat = self.sp(x)        
        cp_feat8, cp_feat16, cp_feat32 = self.cp(x)
        
        fus_feat = self.ffm(sp_feat, cp_feat8)
               
        out8 = self.conv_feat8(fus_feat)
        out8 = F.interpolate(out8, scale_factor = 8, mode='bilinear', align_corners=True)
        
        out16 = self.conv_feat16(cp_feat16)
        out16 = F.interpolate(out16, scale_factor = 16, mode='bilinear', align_corners=True)
        
        out32 = self.conv_feat32(cp_feat32)
        out32 = F.interpolate(out32, scale_factor = 32, mode='bilinear', align_corners=True)
        
        return out8, out16, out32
        
        
        
        
        
       