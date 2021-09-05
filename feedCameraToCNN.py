import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import sys
import fnmatch
import re
import random
from datetime import datetime

import tensorflow as tf
from PIL import Image
from collections import defaultdict

from keras_unet.utils import get_augmented
from keras_unet.models import custom_unet
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

import jetson.inference
import jetson.utils
import cv2




#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

        
#(4, 256, 256, 1)


test_images_dir = "../cnn/"

#test_images_dir = "jpgs/"
#test_masks_dir = "pngs/"


#from keras_unet.losses import jaccard_distance

test_image_paths = sorted(
    [
        os.path.join(test_images_dir, fname)
        for fname in os.listdir(test_images_dir)
        if fname.startswith("rgb") and fname.endswith(".jpg")
    ]
)


imgs_list = []
masks_list = []



random.seed(datetime.now())
enough = random.randrange(5, len(test_image_paths) - 1)
startindex = enough - 5
counter = 0


count = 0
for image in test_image_paths:
    count+=1    
    
    if count > startindex and count < enough:
        i = Image.open(image).resize((256,256))
        i = i.convert('L') 
        imgs_list.append(np.array(i))


imgs_np = np.asarray(imgs_list)
x = np.asarray(imgs_np, dtype=np.float32)/255
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
input_shape = x[0].shape

print(input_shape)

#input_shape = (1, 256, 256, 1)

model = custom_unet(
    input_shape,
    use_batch_norm=True,
    num_classes=4,
    filters=32,
    dropout=0.2,
    output_activation='sigmoid'
)

model.summary()



sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("sim_vision_weights")])

#model_filename = 'chicken_training_varied.h5'

callback_checkpoint = ModelCheckpoint(
    sorted_weights[0], 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)


lr = random.uniform(0.001,0.0007)
model.compile(
    optimizer=Adam(learning_rate=lr), 
    #optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)


model.load_weights(sorted_weights[0])




camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput("rtp://192.168.101.127:1234","--headless") # 'my_video.mp4' for file


while True:
    cuda_img = camera.Capture()
    array = jetson.utils.cudaToNumpy(cuda_img)
    print(array.shape)

    x = cv2.resize(array, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    print(x.shape)
    
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    gray_expanded = gray[:, :, np.newaxis]
    print(gray_expanded.shape)
    
    
    enlisted = []
    enlisted.append(gray_expanded)
    #nplist = np.asarray(enlisted)
    cnn_sized_image = np.asarray(enlisted, dtype=np.float32)/255
    #cnn_sized_image = cnn_sized_image.reshape(cnn_sized_image.shape[0], cnn_sized_image.shape[1], 1)
    
    print(cnn_sized_image.max())
    print(cnn_sized_image.shape)
    y_pred = model.predict(cnn_sized_image)
    
    
    display.Render(jetson.utils.cudaFromNumpy(y_pred[0]))
    
    
    
    #from y_pred[0][:,:,0], y_pred[0][:,:,1], y_pred[0][:,:,2], y_pred[0][:,:,3]
    #to img
#         i = Image.open(image).resize((256,256))
#         i = i.convert('L') 
#         imgs_list.append(np.array(i))
    
    
    
#    display.Render(cuda_img)
#    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    if not camera.IsStreaming() or not display.IsStreaming():
        break






# from keras_unet.utils import plot_imgs

# plot_imgs(org_imgs=np.stack((x[0], x[0], x[0], x[0])),
#           mask_imgs=np.stack((y[0][:,:,0], y[0][:,:,1], y[0][:,:,2], y[0][:,:,3])),
#           pred_imgs=np.stack((y_pred[0][:,:,0], y_pred[0][:,:,1], y_pred[0][:,:,2], y_pred[0][:,:,3])),
#           nm_img_to_plot=4)


