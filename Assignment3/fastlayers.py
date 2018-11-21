import numpy as np
from adam_optimizer import Adam

from im2col_cython import col2im_cython, im2col_cython
from im2col_cython import col2im_6d_cython
from im2col import *

class ConvolutionalLayer:
    
    def __init__(self,
                 input_shape=[3,32,32],
                 num_filters = 32,
                 filter_dims = [3,3,3], 
                 stride = 1,
                 padding = 1,
                 weight_scale = 1e-3):

        self.name = "Conv Layer"
        self.num_filters = num_filters #number of filters applied
        self.C_filter = filter_dims[0] #Channel dims
        self.F_filter = filter_dims[1] #Dimensions for Filter Window
        self.S_filter = stride         #Stride applied
        self.padding_input = padding   #Padding True/False
        self.C_input = input_shape[0]  #Input Channel dims (Must Match Filter Channel)
        self.N_input = input_shape[1] + 2*padding  #Dimensions of the Input 
        self.input_shape = input_shape #Input Shape
        self.trainable_params = True
        
        if(self.C_filter != self.C_input):
            print ('Error in Channel for CONV Layer')

        self.O = int((self.N_input-self.F_filter)/self.S_filter) + 1 
        self.output_shape = (self.num_filters,self.O,self.O) 
        
        print self.output_shape


        self.Weight = weight_scale * np.random.randn(self.num_filters,
                                                     self.C_input,
                                                     self.F_filter,
                                                     self.F_filter)
        
        self.Bias = np.zeros(self.num_filters)
        self.x_cols = None
        
        
        #Initializing Cache
        self.cache_input = None
        
        self.Weight_grad = None
        self.Bias_grad = None
        
        self.best_params = None
        
        self.Weight_opt = Adam(self.Weight)
        self.Bias_opt = Adam(self.Bias)
        
        
        
    def get_summary(self):
        print 'Input Shape: ', self.input_shape
        print 'Output Shape: ' , self.output_shape
        
    
    def save_params(self):
        self.best_params = (self.Weight, self.Bias)
        
    def load_best_params(self):
        if self.best_params is not None:
            self.Weight, self.Bias = self.best_params
        
    def forward(self,x,mode="training"):
        N, C, H, W = x.shape
        w = self.Weight
        b = self.Bias
        F, _, HH, WW = w.shape
        stride = self.S_filter
        pad = self.padding_input
        
        assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
        assert (H + 2 * pad - HH) % stride == 0, 'height does not work'   
        p = pad
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        H += 2 * pad
        W += 2 * pad
        out_h =  int((H - HH) / stride) + 1
        out_w =  int((W - WW) / stride) + 1
        shape = (C, HH, WW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded,
               shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * HH * WW, N * out_h * out_w)
        res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)
        res.shape = (F, N, out_h, out_w)
        out = res.transpose(1, 0, 2, 3)
        out = np.ascontiguousarray(out)
        self.cache_input = x
        self.x_cols = x_cols
        
  
        return out 

    def backward(self,dout,mode="training"):
        
        x = self.cache_input
        w = self.Weight
        b = self.Bias
        x_cols = self.x_cols
        stride = self.S_filter
        pad = self.padding_input
        
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, out_h, out_w = dout.shape
        
        db = np.sum(dout, axis=(0, 2, 3))
        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
        dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)
        dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
        dx_cols.shape = (C, HH, WW, N, out_h, out_w)
        dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)
        self.Weight_grad = dw
        self.Bias_grad = db
        self.Weight = self.Weight_opt.update(self.Weight,self.Weight_grad)
        self.Bias = self.Bias_opt.update(self.Bias,self.Bias_grad)
        
        return dx

  
    