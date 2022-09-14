from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
import tensorflow as tf
import numpy as np
from preprocessing import processImage, processMask
from metrics import diceCoef, diceLoss, iou
from InceptionResNetV2 import inceptionResnetv2Unet
from unet import unet
from vgg19 import vgg19Unet

SEED = 24
W = 512
H = 512

imageFileNames = tf.data.Dataset.list_files(
    "../input/segmentation-full-body-tiktok-dancing-dataset/segmentation_full_body_tik_tok_2615_img/images/*.png",
    seed = SEED,
    shuffle= False,
)

maskFileNames = tf.data.Dataset.list_files(
    "../input/segmentation-full-body-tiktok-dancing-dataset/segmentation_full_body_tik_tok_2615_img/masks/*.png",
    seed = SEED,
    shuffle= False,
)

def loadData():
    for x, z in tf.data.Dataset.zip((imageFileNames.take(2515), maskFileNames.take(2515))).as_numpy_iterator():
        img = processImage(x)
        mask = processMask(z)

        yield(img , mask)

def ValLoadData():
    for x, z in tf.data.Dataset.zip((imageFileNames.skip(2515), maskFileNames.skip(2515))).as_numpy_iterator():
        img = processImage(x)
        mask = processMask(z)

        yield(img , mask)

traindata = tf.data.Dataset.from_generator(
    loadData,
    output_types = (tf.float32, tf.float32),
    output_shapes = ((512,512,3),(512,512,1))
)

ValidData = tf.data.Dataset.from_generator(
    ValLoadData,
    output_types = (tf.float32, tf.float32),
    output_shapes = ((512,512,3),(512,512,1))
)

model = inceptionResnetv2Unet((H, W, 3))
metrics = [diceCoef, iou, Recall(), Precision()]
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=diceLoss, 
    metrics=metrics
)

model.fit(
    traindata.batch(3),
    batch_size = 3,
    epochs=5,
    max_queue_size= 3,
    validation_data = ValidData.batch(3),
)

model.save("./segmodeRestNet_5.h5")