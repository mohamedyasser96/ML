import numpy as np
class Adam:
    def __init__(self,input_arr):
        
         self.name = "Adam Optimizer"
         self.learning_rate = 1e-3
         self.beta1 = 0.9
         self.beta2 = 0.999
         self.epsilon = 1e-8
         self.m = np.zeros_like(input_arr)
         self.v = np.zeros_like(input_arr)
         self.t = 0
    
    def update(self,x,dx):

        next_x = None
         
        
        learning_rate = self.learning_rate
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.epsilon
        m = self.m
        v = self.v
        t = self.t
          
        t += 1
        m = beta1 * m
        temp = (1 - beta1) * dx
        m = m + temp
          
        v = beta2 * v + (1 - beta2) * (dx**2)
        
        
        mb = m / (1 - beta1**t)
        vb = v / (1 - beta2**t)
            
        next_x = -learning_rate * mb / (np.sqrt(vb) + eps) + x
            
        self.m, self.v, self.t = m, v, t
        return next_x