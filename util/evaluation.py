import numpy as np



class ConfusionMatrix(object):
    
    def __init__(nclass):
        
        self.hist = np.zeros(nclass, nclass)
        self.nclass = nclass
        
    def update(preds, label):        
        
        self.hist += np.bincount(self.nclass * preds.astype(int) + label, minlength=self.nclass ** 2).reshape(self.nclass, self.nclass)

    def computeiou():

        ious = self.hist.diag() / (self.hist.sum(dim=0) + self.hist.sum(dim=1) - self.hist.diag())
        miou = ious.mean()

        return miou