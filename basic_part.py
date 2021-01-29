from model.bisenet import BiseNet
import torch


def get_model(args):
    if args.model_type == 'BiseNet':
        net = BiseNet(args)
    return net


def get_optimizer(params, args):
    if args.optimizer_type == 'sgd':
        opt = torch.optim.SGD(params, lr=args.LR, momentum=args.momentum, weight_decay=args.decay, nesterov=args.nesterov)
    elif args.optimizer_type == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr=args.LR, momentum=args.momentum, weight_decay=args.decay)
    elif args.optimizer_type == 'adam':
        opt = torch.optim.Adam(params, lr=args.LR)
        
        
    return opt



def get_scheduler(args, opt):
    
    
    if args.scheduler_type == 'LambdaLR':
        lbd = lambda itcount:  ((args.maxitcount - itcount)/(args.maxitcount - itcount + 1)) ** args.scheduler_power
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lbd)
        
    return scheduler


def adjust_learning_rate(args, optimizer, itcont):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr = args.LR * ((1 - itcont/args.maxitcount)**args.scheduler_power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

