import torch
import torch.distributed as dist


class ConfusionMatrix(object):
    
    def __init__(self, nclass, distributed = True, ignore_label = 255):
        
        self.hist = torch.zeros(nclass, nclass, dtype=torch.long).cuda()
        self.nclass = nclass
        self.ignore_label = ignore_label
        self.distributed = distributed
        
    def update(self, preds, label):        
        
        preds = torch.argmax(preds, dim=1)
        mask = label != self.ignore_label
        self.hist += torch.bincount(self.nclass * preds[mask] + label[mask], minlength=self.nclass ** 2).view(self.nclass, self.nclass)

    def computeiou(self):
        
        if self.distributed:
            dist.all_reduce(self.hist, dist.ReduceOp.SUM)
        ious = self.hist.diag().float() / (self.hist.sum(dim=0) + self.hist.sum(dim=1) - self.hist.diag()).float()
        miou = ious.mean()

        return miou.item()