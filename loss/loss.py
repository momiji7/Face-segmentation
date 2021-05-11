import torch.nn.functional as F
import torch.nn as nn
import torch


class SoftmaxLoss(nn.Module):
    def __init__(self, ignore_label = 255):
        super(SoftmaxLoss, self).__init__()     
        self.nll = nn.NLLLoss(ignore_index=ignore_label)

    def forward(self, logits, labels):
        log_score = F.log_softmax(logits, dim=1)
        loss = self.nll(log_score, labels)
        return loss
    
    
    
class OhemCELoss(nn.Module):

    def __init__(self, args, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        thresh = args.ohem_thresh
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criterion(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)
    
    
class FocalLoss(nn.Module):
    def __init__(self, args, ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.gamma = args.fl_gamma
        

    def forward(self, logits, labels):
        # logits: N*C*H*W, labels: N*H*W
        assert ignore_lb,255
        probs_softmax = F.softmax(logits, dim=1)
        lb = torch.unsqueeze(labels, dim = 1)
        probs = probs_softmax.gather(1, lb) # N*1*H*W
        print(probs.size())
        loss = - torch.pow(1 - probs, self.gamma) * torch.log(probs)
        
        return loss.mean()    
    
    
def get_criterion(args):
    
    if args.loss_type == 'softmax':
        criterion = SoftmaxLoss()
    elif args.loss_type == 'ohem':
        criterion = OhemCELoss(args)
    elif args.loos_type == 'focalloss':
        pass
    else:
        raise
    
    return criterion