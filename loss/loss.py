import torch.nn.functional as F
import torch.nn as nn


class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()     
        self.nll = nn.NLLLoss()

    def forward(self, logits, labels):
        log_score = F.log_softmax(logits, dim=1)
        loss = self.nll(log_score, labels)
        return loss