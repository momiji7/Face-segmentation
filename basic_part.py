from model.bisenet import BiseNet
from model.bisenet_author import BiSeNet
import torch


def get_model(args):
    if args.model_type == 'BiseNet':
        # net = BiseNet(args)
        net = BiSeNet()
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
    for i in range(2):
        optimizer.param_groups[i]['lr'] = lr
    for i in range(2, len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr * 10

        
 
class Scheduler(object):
    def __init__(self, args,
                opt,
                lr0,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0     
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)
        self.optim = opt


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
                pg['lr'] = self.lr
        self.it += 1
            
            
    def state_dict(self):
        
        return {}


