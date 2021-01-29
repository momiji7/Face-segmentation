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

def train(args):
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = torch.cuda.device_count(),
                            rank=args.local_rank)
        
    logname = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M'))
    logger = Logger(args.save_path, logname)

        
    train_dataloader = get_dataloader(args, 'train')
    # test_dataloader = get_dataloader(args, 'test')
    
    args.maxitcount = args.epochs*len(train_dataloader)
    
    net = get_model(args)
    net.cuda()  
        
    if dist.get_rank() == 0:
        tfboard_writer = SummaryWriter()  
        logger.log('Arguments : -------------------------------')
        for name, value in args._get_kwargs():
            logger.log('{:16} : {:}'.format(name, value))
        logger.log("=> network :\n {}".format(net))   
    
    optimizer = get_optimizer(filter(lambda p: p.requires_grad, net.parameters()), args)    
    scheduler = get_scheduler(args, optimizer)  
    criterion = SoftmaxLoss()
    
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
                image = image.cuda()
                label = label.cuda()
                out, _, _ = net(image)

                image_np = tensor2img(image)
                label_np = label.cpu().numpy()
                out_np   = out.cpu().numpy().argmax(0)

                samp_num = min(16, image.size()[0])
                canvas = np.zeros((args.input_height*samp_num, args.input_width*3, 3))
                for j in range(image.size()[0]):  
                    gt = show_seg(image_np[j], label_np[j], 19)
                    prediction = show_seg(image_np[j], out_np[j], 19)

                    canvas[j*args.input_height:(j+1)*args.input_height,0*args.input_width:1*args.input_width] = image_np[j]
                    canvas[j*args.input_height:(j+1)*args.input_height,1*args.input_width:2*args.input_width] = gt
                    canvas[j*args.input_height:(j+1)*args.input_height,2*args.input_width:3*args.input_width] = prediction
                    
            logger.save_images(canvas, 'epoch-{:03d}.jpg'.format(epoch))
        
        
    logger.close()
            
    
    
    
  
if __name__ == '__main__':
    args = obtain_basic_args()
    train(args)