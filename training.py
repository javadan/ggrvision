

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import fnmatch
import re

from sklearn.model_selection import train_test_split


from keras_unet.utils import get_augmented
import tensorflow as tf
from PIL import Image
from collections import defaultdict

from keras_unet.models import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded


#from keras_unet.losses import jaccard_distance



base_dir = "../../../"

train_images_dir = base_dir + "images/Chicken/train/"
validation_images_dir = base_dir + "images/Chicken/validation/"
test_images_dir = base_dir + "images/Chicken/test/"


train_masks_dir = base_dir + "masks/Chicken/train/"
validation_masks_dir = base_dir + "masks/Chicken/validation/"
test_masks_dir = base_dir + "masks/Chicken/test/"

masks_dir = base_dir + "masks/"

if not os.path.exists(train_masks_dir):
    os.makedirs(train_masks_dir)


train_image_paths = sorted(
    [
        os.path.join(train_images_dir, fname)
        for fname in os.listdir(train_images_dir)
        if fname.endswith(".jpg")
    ]
)

train_mask_paths = sorted(
    [
        os.path.join(train_masks_dir, fname)
        for fname in os.listdir(train_masks_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of Train images:", len(train_image_paths))
print("Number of Train masks:", len(train_mask_paths))


for fname in os.listdir(train_images_dir):
    if len(glob.glob(train_masks_dir + "*" + fname[:-4] + "*")) == 0:
        print(fname)
    

train_map = defaultdict(list)

for fname in os.listdir(train_images_dir):
    for mask in glob.glob(train_masks_dir + "*" + fname[:-4] + "*"):
        train_map[fname].append(mask)


def trainEpochWithNewData(offset, training_size, train_map, best_file_checkpoint):

    imgs_list = []
    masks_list = []


    counter = 0
    for mask in train_map:
        counter += 1
        if (counter > offset and counter <= offset + training_size):
            for value in train_map[mask]:
                i = Image.open(train_images_dir + mask).resize((256,256))
                i = i.convert('L') 
                imgs_list.append(np.array(i))

                m = Image.open(value).resize((256,256))
                m = m.convert('L') 
                masks_list.append(np.array(m))
        elif (counter > offset + training_size):
            break
            
    imgs_np = np.asarray(imgs_list)
    masks_np = np.asarray(masks_list)

    print(imgs_np.shape, masks_np.shape)
    print(imgs_np.max(), masks_np.max())

    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.asarray(masks_np, dtype=np.float32)/255

    print(x.max(), y.max())
    print(x.shape, y.shape)
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
    print(x.shape, y.shape)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    print(x.shape, y.shape)



    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)

    train_gen = get_augmented(
        x_train, y_train, batch_size=1,
        data_gen_args = dict(
            horizontal_flip=True
        ))



    input_shape = x_train[0].shape

    model = custom_unet(
        input_shape,
        use_batch_norm=False,
        num_classes=1,
        filters=32,
        dropout=0.2,
        output_activation='sigmoid'
    )

   # model.summary()

    model.compile(
        optimizer=Adam(),
        #optimizer=SGD(lr=0.01, momentum=0.99),
        loss='binary_crossentropy',
        #loss=jaccard_distance,
        metrics=[iou, iou_thresholded]
    )

    try:
        model.load_weights(best_file_checkpoint)
    except Exception:
        print("Not loading weights")

    fname = "weights-{val_loss:.4f}.hdf5"

    callback_checkpoint = ModelCheckpoint(
        fname, 
        verbose=1, 
        monitor='val_loss', 
        save_best_only=True
    )
    callbacks_list = [callback_checkpoint]

    
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=5,
        epochs=2,

        validation_data=(x_val, y_val),
        callbacks=callbacks_list
    )






try:
    sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
    print(sorted_weights)
    trainEpochWithNewData(0, 100, train_map, sorted_weights[0])
except Exception as inst:
    print(inst)
    sorted_weights = None
    trainEpochWithNewData(0, 100, train_map)

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(100, 200, train_map, sorted_weights[0])

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(200, 300, train_map, sorted_weights[0])

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(300, 400, train_map, sorted_weights[0])

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(400, 500, train_map, sorted_weights[0])

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(500, 600, train_map, sorted_weights[0])

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(600, 700, train_map, sorted_weights[0])

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(700, 800, train_map, sorted_weights[0])

sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])
print(sorted_weights)

trainEpochWithNewData(800, 900, train_map, sorted_weights[0])
