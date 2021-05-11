from basic_part import *
from basic_args import obtain_basic_args
import torch
import torch.nn as nn
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
from util.util import tensor2img, transform_tensor2img
from util.evaluation import ConfusionMatrix
import os

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)



def train(args):
    
    args.has_apex = True and not args.no_apex
    try:
        from apex import amp, parallel
        from apex.parallel import DistributedDataParallel, SyncBatchNorm
    except ImportError:
        args.has_apex = False
    
    num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = num_gpus > 1
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = torch.cuda.device_count(),
                                rank=args.local_rank)
        
    logname = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'))
    logger = Logger(args.save_path, logname)

        
    train_dataloader = get_dataloader(args, 'train')
    eval_dataloaders = []
    if not args.eval_lists is None:
        eval_lists = deepcopy(args.eval_lists)
        for el in eval_lists:
            args.eval_lists = el
            eval_dataloader = get_dataloader(args, 'eval')
            eval_dataloaders.append(eval_dataloader)
    
    args.maxitcount = args.epochs*len(train_dataloader)
    
    net = get_model(args)
    
    
    init_weight(net.business_layer, nn.init.kaiming_normal_,
            SyncBatchNorm, 1e-5, 0.1,
            mode='fan_in', nonlinearity='relu')
    
    
    

    if args.has_apex: 
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
   
    net.cuda()  
        
    if not (args.distributed and dist.get_rank() != 0):
        tfboard_writer = SummaryWriter()  
        logger.log('Arguments : -------------------------------')
        for name, value in args._get_kwargs():
            logger.log('{:16} : {:}'.format(name, value))
        logger.log("=> network :\n {}".format(net)) 
        
    def group_weight(weight_group, module, norm_layer, lr, no_decay_lr=None):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, norm_layer) or isinstance(m, (
            nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
        assert len(list(module.parameters())) == len(group_decay) + len(
            group_no_decay)
        weight_group.append(dict(params=group_decay, lr=lr))
        lr = lr if no_decay_lr is None else no_decay_lr
        weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
        return weight_group    

        
        
    base_lr = args.LR    
    params_list = []
    params_list = group_weight(params_list, net.context_path,
                               SyncBatchNorm, base_lr)
    params_list = group_weight(params_list, net.spatial_path,
                               SyncBatchNorm, base_lr * 10)
    params_list = group_weight(params_list, net.global_context,
                               SyncBatchNorm, base_lr * 10)
    params_list = group_weight(params_list, net.arms,
                               SyncBatchNorm, base_lr * 10)
    params_list = group_weight(params_list, net.refines,
                               SyncBatchNorm, base_lr * 10)
    params_list = group_weight(params_list, net.heads,
                               SyncBatchNorm, base_lr * 10)
    params_list = group_weight(params_list, net.ffm,
                               SyncBatchNorm, base_lr * 10)    
        
    
    optimizer = get_optimizer(params_list, args)    
    # scheduler = get_scheduler(args, optimizer)
    scheduler = Scheduler(args, optimizer, 1e-2, 0.9, 5e-4, 1000, 1e-5, args.maxitcount, 0.9)
    criterion = get_criterion(args)
    
    
    if args.has_apex:        
        opt_level = 'O1' if args.use_fp16 else 'O0'
        net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)  
    
    last_info = logger.last_info()
    if last_info.exists():
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(last_info)
        start_epoch = last_info['epoch'] + 1
        checkpoint  = torch.load(last_info['last_checkpoint'])
        assert last_info['epoch'] == checkpoint['epoch'], 'Last-Info is not right {:} vs {:}'.format(last_info, checkpoint['epoch'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.log("=> load-ok checkpoint '{:}' (epoch {:}) done" .format(logger.last_info(), checkpoint['epoch']))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch = 0 
    
    if args.distributed:
        if args.has_apex:
            net = parallel.DistributedDataParallel(net, delay_allreduce=True)
        else:
            net = nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank, ], output_device = args.local_rank, find_unused_parameters=True)
         
    # net = nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank, ], output_device = args.local_rank)
    
    itcont = -1
    for epoch in range(start_epoch, args.epochs):
        net.train()
        
        train_loss_epoch = AverageMeter()
        for i, (image, label) in enumerate(train_dataloader):
            itcont += 1
            adjust_learning_rate(args, optimizer, itcont)
            
            image = image.cuda()
            label = label.cuda()
                        
            out8, out16, out32 = net(image)
            
            loss8 = criterion(out8, label)
            loss16 = criterion(out16, label)
            loss32 = criterion(out32, label)
            
            loss = loss8 + (loss16 + loss32) * args.loss_alpha
           
            train_loss_epoch.update(loss.item(), image.size(0))
            
            optimizer.zero_grad()            
            if args.has_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            
            if (i % args.print_freq == 0 or i+1 == len(train_dataloader)):
                logger.log('[train Info]: [epoch-{}-{}][{:04d}/{:04d}][LR:{:.6f}][Loss:{:.4f}][loss8:{:.4f}][loss16:{:.4f}][loss32:{:.4f}]'.format(epoch, args.epochs, i, len(train_dataloader), optimizer.param_groups[0]['lr'], loss.item(), loss8.item(), loss16.item(), loss32.item()))
                
        
        if dist.get_rank() == 0:
            logger.log('epoch {:02d} completed!'.format(epoch))
            logger.log('[train Info]: [epoch-{}-{}][Avg Loss:{:.6f}]'.format(epoch, args.epochs, train_loss_epoch.avg))
            tfboard_writer.add_scalar('Average Loss', train_loss_epoch.avg, epoch)
        
     
        # save checkpoint
        
        if dist.get_rank() == 0:
            filename = 'epoch-{}-{}.pth'.format(epoch, args.epochs)
            save_path = logger.path('model') / filename
            torch.save({
              'epoch': epoch,
              'args' : deepcopy(args),
              'state_dict': net.module.state_dict(),
              'scheduler' : scheduler.state_dict(),
              'optimizer' : optimizer.state_dict(),
                }, logger.path('model') / filename)  
            logger.log('save checkpoint into {}'.format(filename))
            last_info = torch.save({
              'epoch': epoch,
              'last_checkpoint': save_path
             }, logger.last_info())
        
        with torch.no_grad():
            net.eval()
            
            for edl_idx, edataloader in enumerate(eval_dataloaders):
                eval_loss_epoch = AverageMeter()
                cfm = ConfusionMatrix(args.nclass)
                for i, (image, label) in enumerate(edataloader):


                    image = image.cuda()
                    label = label.cuda()

                    out8, out16, out32 = net(image)

                    loss8 = criterion(out8, label)
                    loss16 = criterion(out16, label)
                    loss32 = criterion(out32, label)

                    loss = loss8 + (loss16 + loss32) * args.loss_alpha

                    eval_loss_epoch.update(loss.item(), image.size(0))

                    cfm.update(out8, label)

                    if (i % args.print_freq == 0 or i+1 == len(eval_dataloader)):
                        logger.log('[Eval Info]: [epoch-{}-{}][{:04d}/{:04d}][Loss:{:.4f}][loss8:{:.4f}][loss16:{:.4f}][loss32:{:.4f}]'.format(epoch, args.epochs, i, len(eval_dataloader), loss.item(), loss8.item(), loss16.item(), loss32.item()))
                        

                miou = cfm.computeiou()
                if dist.get_rank() == 0:
                    logger.log('[Eval Info]: [Eval:{}][epoch-{}-{}][Avg Loss:{:.6f}][mIOU:{:.6f}]'.format(edl_idx, epoch, args.epochs, eval_loss_epoch.avg, miou))
                    tfboard_writer.add_scalar('Average Loss/eval{}'.format(edl_idx), eval_loss_epoch.avg, epoch) 
                    tfboard_writer.add_scalar('mIOU/eval{}'.format(edl_idx), miou, epoch)

                    image = image.cuda()
                    label = label.cuda()
                    out, _, _ = net(image)

                    image_np = transform_tensor2img(image)
                    label_np = label.cpu().numpy()
                    out_np   = out.cpu().numpy().argmax(1)

                    samp_num = min(16, image.size()[0])
                    canvas = np.zeros((args.input_height*samp_num, args.input_width*3, 3))
                    for j in range(image.size()[0]):  
                        gt = show_seg(image_np[j], label_np[j], args.nclass)
                        prediction = show_seg(image_np[j], out_np[j], args.nclass)

                        canvas[j*args.input_height:(j+1)*args.input_height,0*args.input_width:1*args.input_width] = image_np[j]
                        canvas[j*args.input_height:(j+1)*args.input_height,1*args.input_width:2*args.input_width] = gt
                        canvas[j*args.input_height:(j+1)*args.input_height,2*args.input_width:3*args.input_width] = prediction

                    logger.save_images(canvas, 'eval{}-epoch-{:03d}.jpg'.format(edl_idx, epoch))
        
        
    logger.close()
            
    
    
    
  
if __name__ == '__main__':
    args = obtain_basic_args()
    train(args)