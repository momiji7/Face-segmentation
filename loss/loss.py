import torch.nn.functional as F
import torch.nn as nn


class SoftmaxLoss(nn.Module):
    def __init__(self, ignore_label = 255):
        super(SoftmaxLoss, self).__init__()     
        self.nll = nn.NLLLoss(ignore_index=ignore_label)

    def forward(self, logits, labels):
        log_score = F.log_softmax(logits, dim=1)
        loss = self.nll(log_score, labels)
        return loss