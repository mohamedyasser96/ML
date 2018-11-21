# In[5]:


import cv2
import os
import argparse
import functools
from functools import cmp_to_key
from PIL import Image

import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model, load_model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf


# In[7]:


def video_to_frames(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    cap = cv2.VideoCapture(input_loc)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    length = video_length
    print ("Number of frames: ", video_length)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imwrite(output_loc + "/%#01d.jpg" % (count+1), frame)
        count = count + 1
        if (count > (video_length-1)):
            cap.release()
            print ("Done extracting frames.\n%d frames extracted" % count)
            break
    return length


# In[8]:


def isnum (num):
    try:
        int(num)
        return True
    except:
        return False

def image_sort (x,y):
    x = int(x.split(".")[0])
    y = int(y.split(".")[0])
    return x-y

def frames_to_videos(dir_path, ext, output, framerate, time, sort_type, visual):
    dir_path = dir_path
    ext = ext
    output = output
    framerate = framerate
    sort_type = sort_type
    time = time
    visual = False
    
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)

    print(len(images))
    
    if not isnum(framerate):
        framerate = 10
    else:
        framerate = int(framerate)

    if sort_type == "numeric":
        int_name = images[0].split(".")[0]
        if isnum(int_name):
            images = sorted(images, key=cmp_to_key(image_sort))
        else:
            print("Failed to sort numerically, switching to alphabetic sort")
            images.sort()
    elif sort_type == "alphabetic":
        images.sort()

    if isnum(time):
        framerate = int(len(images) / int(time))
        print("Adjusting framerate to " + str(framerate))
        
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    if visual:
        cv2.imshow('video',frame)
    regular_size = os.path.getsize(image_path)
    height, width, channels = frame.shape

    visual = visual

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, framerate, (width, height))

    for n, image in enumerate(images):
        image_path = os.path.join(dir_path, image)
        image_size = os.path.getsize(image_path)
        if image_size < regular_size / 1.5:
            print("Cancelled: " + image)
            continue

        frame = cv2.imread(image_path)
        out.write(frame) # Write out frame to video
        if visual:
            cv2.imshow('video', frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
        if n%100 == 0:
            print("Frame " + str(n))

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))


# In[9]:


def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    resized_image.save(output_image_path)


# In[10]:


input_loc = './BW2.MP4'
output_loc = './Frames'
ext = "jpg"
video_length = video_to_frames(input_loc, output_loc)


# In[11]:


for i in range(1, video_length+1):
    inpath = "./Frames/"+str(i)+"."+ext
    outpath = "./Testing/"+str(i)+"."+ext
    size= (256, 256)
    resize_image(inpath, outpath,size)
print("Image: ", i, "resized")


# In[12]:


inception = InceptionResNetV2(weights='inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5', include_top=True)
inception.graph = tf.get_default_graph()
print("Inception Loaded")

# In[13]:


model = load_model("weights.hdf5")
print("Model Loaded")

# In[14]:


def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


# In[ ]:


color_me = []
for filename in os.listdir('Testing/'):
    color_me.append(img_to_array(load_img('Testing/'+filename)))
color_me = np.array(color_me, dtype=float)
gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

print("Color Me Done")

# In[ ]:


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

print("Prediction Done")
# In[ ]:


# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("Results/"+str(i)+".jpg", lab2rgb(cur))

print("Prediction Saved")

# In[10]:


dir_path = "./Results"
ext = "jpg"
output = "./BW2 C.mp4"
framerate = 25.02
sort_type = "numeric"
time = "hi"
visual = False

frames_to_videos(dir_path, ext, output, framerate, time, sort_type, visual)

