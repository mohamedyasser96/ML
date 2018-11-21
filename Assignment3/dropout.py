import numpy as np
from optimizer import Adam


class Dropout:
    def __init__(self,prob=0.5):
        self.prob = prob
        self.trainable_params = False
    
    def forward(self,X,mode="training"):
        if mode == "training":
            self.mask = np.random.binomial(1,self.prob,size=X.shape) / self.prob
            out = X * self.mask
            return out.reshape(X.shape)
        else:
            return X
    
    def backward(self,dout,mode="training"):
        if mode == "training":
            dout = dout * self.mask
        
        return dout