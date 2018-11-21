import numpy as np
from optimizer import Adam

class Dense:
    def __init__(self,input_shape, neurons, reg=1e-4):
        
        self.name = "Dense Layer"
        self.neurons = neurons
        self.input_shape = input_shape
        self.reg = reg
        
        self.Weight = np.random.randn(input_shape, neurons)/np.sqrt(input_shape/2.0)
        self.Bias = np.zeros(neurons)
        
        self.cache_input = None
        self.Weight_grad = None
        self.Bias_grad = None
        
        self.trainable_params = True
        self.best_params = None
        
        self.Weight_opt = Adam(self.Weight)
        self.Bias_opt = Adam(self.Bias)
        
    def save_params(self):
        self.best_params = (self.Weight, self.Bias)
        
    def load_best_params(self):
        if self.best_params is not None:
            self.Weight, self.Bias = self.best_params
            
    def forward(self,X,mode="training"):

        output = X.reshape(X.shape[0], self.Weight.shape[0]).dot(self.Weight) + self.Bias
        self.input_cache = X
        return output
    
    def backward(self,dout,mode="training"):
        X = self.input_cache
        
        dw = X.reshape(X.shape[0], self.Weight.shape[0]).T.dot(dout)
        db = np.sum(dout, axis=0)
        dx = dout.dot(self.Weight.T).reshape(X.shape)
        
        self.Weight_grad = dw + (self.reg * self.Weight)
        self.Bias_grad = db
        
        self.Weight = self.Weight_opt.update(self.Weight,self.Weight_grad)
        self.Bias = self.Bias_opt.update(self.Bias,self.Bias_grad)
        
        return dx