import numpy as np
class Relu:
    def __init__(self,alpha=0.01):
        self.input_cache = None
        self.name = "Relu Activation"
        self.alpha = alpha
        self.trainable_params = False
    
    def forward(self,X,mode="training"):
        
        out = np.ones_like(X)
        out[X < 0] = self.alpha 
        self.input_cache = X
        return out * X
    
    def backward(self,dout,mode="training"):
        X = self.input_cache
        dx = np.ones_like(X)
        dx[X < 0] = self.alpha
        return dx * dout
    
    