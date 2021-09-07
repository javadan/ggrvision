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
    
    resized_img = jetson.utils.cudaAllocMapped(width=256, height=256, format='rgb8')
    jetson.utils.cudaResize(cuda_img, resized_img)
    
    
    
    
    PILim = Image.fromarray(jetson.utils.cudaToNumpy(resized_img))
    PILim = PILim.convert('L') 
    
    
    
    
    enlisted = []
    enlisted.append(np.array(PILim))
    nplist = np.asarray(enlisted)
    cnn_sized_image = np.asarray(nplist, dtype=np.float32)/255
    
    x = cnn_sized_image
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    
    #print(x)
#     print(x.shape)
    
    
#     print("Predicting")
    
    y_pred = model.predict(x)
    
    r = y_pred[0][:,:,2] #human
    g = y_pred[0][:,:,1] #chicken
    b = y_pred[0][:,:,0] #egg
    
    rgb_uint8 = (np.dstack((r,g,b)) * 255) .astype(np.uint8)
    
    
#     fiddyfiddy = random.choice([0, 1])
#     if fiddyfiddy == 1:
   # display.Render(resized_img)
#     if fiddyfiddy == 1:
    display.Render(jetson.utils.cudaFromNumpy(rgb_uint8) )
    
    
    
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



    #gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    #gray_expanded = gray[:, :, np.newaxis]
    #print(gray_expanded.shape)

    #from https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-examples.py
    #gray_img = convert_color(gray, "rgb8")

    #cv2.cvtColor() with cv2.COLOR_RGB2BGR
    
    
#     enlisted = []
#     enlisted.append(x)
#     #nplist = np.asarray(enlisted)
#     cnn_sized_image = np.asarray(enlisted, dtype=np.float32)/255
#     #cnn_sized_image = cnn_sized_image.reshape(cnn_sized_image.shape[0], cnn_sized_image.shape[1], 1)
    
#     print(cnn_sized_image.max())
#     print(cnn_sized_image.shape)
#     #y_pred = model.predict(cnn_sized_image)
    
    
#     imgInput = cuda_img# jetson.utils.loadImage('my_image.jpg')

#     allocate the output image, with the same dimensions as input
#     imgOutput = jetson.utils.cudaAllocMapped(width=imgInput.width, height=imgInput.height, format=imgInput.format)

    # normalize the image from [0,255] to [0,1]
    #jetson.utils.cudaNormalize(imgInput, (0,255), imgOutput, (0,1))
    

    # load the image into CUDA memory
    
    
    
    
    #rr, gg, bb = y_pred.split()
    #rr = y1.point(lambda p: 0 if p==0 else np.random.randint(256) )
    #gg = y2.point(lambda p: 0 if p==0 else np.random.randint(256) )
    #bb = y3.point(lambda p: 0 if p==0 else np.random.randint(256) )
    #new_image = Image.merge("RGB", (rr, gg, bb))
    
    
    
    #bands = [y1,y2,y3]
    #multi_layer_img = Image.merge("RGB", bands)
    

    
    #y_pred_img = Image.fromarray(y_pred[0][:,:,3])
    
#     new_image.paste(y_pred[0][:,:,2], box=(0, 0) + new_image.size)
#     new_image.paste(y_pred[0][:,:,1], box=(0, 0) + new_image.size)
#     new_image.paste(y_pred[0][:,:,0], box=(0, 0) + new_image.size)
    
    #new_image.paste(y_pred[0],(0,0))

    #display.Render(cv_img)
    
    #display.Render(rgb_img)
    
    
    

# #https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/depthnet_utils.py
# class depthBuffers:
#     def __init__(self, args):
#         self.args = args
#         self.depth = None
#         self.composite = None
        
#         self.use_input = "input" in args.visualize
#         self.use_depth = "depth" in args.visualize
            
#     def Alloc(self, shape, format):
#         depth_size = (shape[0] * self.args.depth_size, shape[1] * self.args.depth_size)
#         composite_size = [0,0]
        
#         if self.depth is not None and self.depth.height == depth_size[0] and self.depth.width == depth_size[1]:
#             return
            
#         if self.use_depth:
#             composite_size[0] = depth_size[0]
#             composite_size[1] += depth_size[1]
            
#         if self.use_input:
#             composite_size[0] = shape[0]
#             composite_size[1] += shape[1]

#         self.depth = jetson.utils.cudaAllocMapped(width=depth_size[1], height=depth_size[0], format=format)
#         self.composite = jetson.utils.cudaAllocMapped(width=composite_size[1], height=composite_size[0], format=format)
        
        
#     print('RGB image: ')
#     print(rgb_img)

#     # convert to BGR, since that's what OpenCV expects
#     bgr_img = jetson.utils.cudaAllocMapped(width=rgb_img.width,
#                                     height=rgb_img.height,
#                                     format='bgr8')

#     jetson.utils.cudaConvertColor(rgb_img, bgr_img)

#     print('BGR image: ')
#     print(bgr_img)

#     # make sure the GPU is done work before we convert to cv2
#     jetson.utils.cudaDeviceSynchronize()
    
#     # convert to cv2 image (cv2 images are numpy arrays)
#     cv_img = jetson.utils.cudaToNumpy(bgr_img)

#     print('OpenCV image size: ' + str(cv_img.shape))
#     print('OpenCV image type: ' + str(cv_img.dtype))
