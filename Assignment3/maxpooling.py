import numpy as np
from im2col_cython import col2im_cython, im2col_cython
from im2col_cython import col2im_6d_cython
from im2col import *
class MaxPooling:
    
    def __init__(self,stride = 2, pool_dims = [2,2]):
        self.stride = stride
        self.pool_dims = pool_dims
        self.input_cache = None
        self.name = "Maxpool Layer"
        self.trainable_params = False
    
    def forward(self,x,mode="training"):
        
        N, C, H, W = x.shape
        pool_height =  self.pool_dims[0]
        pool_width = self.pool_dims[1]
        stride = self.stride
        
        assert pool_height == pool_width == stride, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        
        x_reshaped = x.reshape(N, C, int(H / pool_height), pool_height, int(W / pool_width), pool_width)
        out = x_reshaped.max(axis=3).max(axis=4)
        
        
        self.input_cache = (x, x_reshaped,out)
        return out
    
    
    def backward(self,dout,mode="training"):

        x, x_reshaped, out = self.input_cache

        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)
        
        return dx
    