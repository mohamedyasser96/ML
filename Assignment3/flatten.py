import numpy as np
class Flatten:
    def __init__(self):
        self.name = "Flatten Layer"
        self.cache_input = None
        self.trainable_params = False
    
    def forward(self,X,mode="training"):
         output_sum = np.product(X.shape[1:])
         self.cache_input = X.shape
         return np.reshape(X,(X.shape[0],output_sum))
         
    
    def backward(self,dout,mode="training"):
        return np.reshape(dout,(self.cache_input))
    
