import tensorflow as tf
# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


#model_name = "EfficientNetB0"
#model_name = "InceptionV3"
#model_name = "MobileNet"
model_name = "MobileNetV2"
#model_name = "ResNet50"
#model_name = "ResNet101"
#model_name = "ResNet152"
#model_name = "ResNet50V2"
#model_name = "ResNet101V2"
#model_name = "ResNet152V2"
#model_name = "VGG16"
#model_name = "VGG19"
#model_name = "Xception"

model=load_model(model_name+'/model.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(model_name+'/model.tflite', 'wb') as f:
  f.write(tflite_model)