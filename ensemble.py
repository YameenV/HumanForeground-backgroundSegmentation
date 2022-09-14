from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
import tensorflow as tf
import numpy as np
from preprocessing import processImage, processMask
from metrics import diceCoef, diceLoss, iou
from InceptionResNetV2 import inceptionResnetv2Unet
from unet import unet
from vgg19 import vgg19Unet
from tensorflow.keras.models import load_model

model1 = load_model("../input/segmodel/segmode1_5.h5", compile=False)
model2 = load_model("../input/vgg16model/segmodevgg19_5.h5", compile=False)
model3 = load_model("../input/restnetmodel/segmodeRestNet_5(1).h5", compile=False)

weights = [0.1, 0.1]

pred1 = model1.predict(traget) > 0.5
pred2 = model2.predict(traget) > 0.5
pred3 = model3.predict(traget) > 0.5

# Only Vgg19 and InceptionResnetV2
preds = [pred2, pred3]

# Ensemble
ensemblePreds = np.tensordot(preds, weights, axes=((0),(0)))