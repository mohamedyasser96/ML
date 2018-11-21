import numpy as np
import sys
from data_utils import *
import time

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()



class Model:
    def __init__(self, X_train, y_train, X_val, y_val, num_epochs, batch_size):
        self.layers = np.array([])
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.augment = False
        self.best_val = 0
        
    def add_layer(self,layer):
        self.layers = np.append(self.layers,layer)
    
    
    
    def add_augmentation(self,rotation_range=0,
                         height_shift_range=0,
                         width_shift_range=0,
                         img_row_axis=1,
                         img_col_axis=2,
                         img_channel_axis=0,
                         horizontal_flip=False,
                         vertical_flip=False):
        
        self.rotation_range = float(rotation_range)
        self.height_shift_range = float(height_shift_range)
        self.width_shift_range= float(width_shift_range)
        self.img_row_axis= int(img_row_axis)
        self.img_col_axis= int(img_col_axis)
        self.img_channel_axis= int(img_channel_axis)
        self.horizontal_flip= bool(horizontal_flip)
        self.vertical_flip= bool(vertical_flip)
        self.augment = True
        
        
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
         # Maybe subsample the data
        N = X.shape[0]
        num_layers = self.layers.shape[0] 
        
        if num_samples is not None and N > num_samples:
          mask = np.random.choice(N, num_samples)
          N = num_samples
          X = X[mask]
          y = y[mask]
        
        # Compute predictions in batches
        num_batches = int(N / batch_size)
        if N % batch_size != 0:
          num_batches += 1
        y_pred = []
        for i in range(num_batches):
            
            start = i * batch_size
            end = (i + 1) * batch_size
          
            output = None
            for j in range(num_layers - 1):
                if j == 0:
                    output = self.layers[j].forward(X[start:end],mode="testing")
                else:
                    output = self.layers[j].forward(output,mode="testing")
        
            y_pred.append(np.argmax(output, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        
        return acc
    
    
    
    def step(self):
        
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        if self.augment:
            X_batch = augment_batch(X_batch,
                              rotation_range=self.rotation_range,
                              height_shift_range=self.rotation_range,
                              width_shift_range=self.width_shift_range,
                              img_row_axis=self.img_row_axis,
                              img_col_axis=self.img_col_axis,
                              img_channel_axis=self.img_channel_axis,
                              horizontal_flip=self.horizontal_flip,
                              vertical_flip=self.vertical_flip)
        
        num_layers = self.layers.shape[0] 
        
        output = None
        for i in range(num_layers - 1):
            if i == 0:
                output = self.layers[i].forward(X_batch)
            else:
                output = self.layers[i].forward(output)

                
        loss, dx = self.layers[num_layers-1].forward_backward(output,y_batch)
        self.loss_history.append(loss)
        
        for i in reversed(range(num_layers-1)):
            dx = self.layers[i].backward(dx)
           
    
    def save_best(self):
        num_layers = self.layers.shape[0]
        for i in range(num_layers):
            if self.layers[i].trainable_params:
                self.layers[i].save_params()
    
    def load_best(self):
        num_layers = self.layers.shape[0]
        for i in range(num_layers):
            if self.layers[i].trainable_params:
                self.layers[i].load_best_params()
        
    def train(self):
        
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)
        training_loss = 0
        training_loss_sum = 0 
        training_num = 0
        self.actual_training_loss = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.loss_history = []
        self.epoch = 0
        
        
        for t in range(num_iterations):
            self.step()
            
            training_num += 1
            training_loss_sum += self.loss_history[len(self.loss_history)-1]
            training_loss = float(training_loss_sum)/(training_num)
            
            
            progress(int(t-(iterations_per_epoch*self.epoch)), iterations_per_epoch, status='Loss: ' + str(training_loss))
            
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                training_num = 0
                training_loss_sum = 0
                self.actual_training_loss.append(training_loss)
                
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                print 'Epoch #: ' , '/',  self.epoch
                print 'Training Accuracy: ' , train_acc
                print 'Val Accuracy: ', val_acc
         
        
        self.load_best()
        

def classpredict(model, X, num_samples=None, batch_size=100):
        N = X.shape[0]
        num_layers = model.layers.shape[0] 
        
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        

        num_batches = int(N / batch_size)
        if N % batch_size != 0:
          num_batches += 1
        y_pred = []
        for i in range(num_batches):
            
            start = i * batch_size
            end = (i + 1) * batch_size
          
            output = None
            for j in range(num_layers - 1):
                if j == 0:
                    output = model.layers[j].forward(X[start:end])
                else:
                    output = model.layers[j].forward(output)
        
            y_pred.append(np.argmax(output, axis=1))
        y_pred = np.hstack(y_pred)
        
        return y_pred
        