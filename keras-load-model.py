'''
Program to load Deep Network (CNN) Keras mode and evaluation performance
Training Set Data        ---  600 images of digits with 64,64,3 dimensions 
Training Labels          ---  digits between 0/1
'''
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.models import model_from_json
from kt_utils import *
get_ipython().run_line_magic('matplotlib', 'inline')


# load data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T



#verify output
print(X_train.shape)

# Load model


# load json and create model
json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")

loaded_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

# run using test data
scores = loaded_model.evaluate(X_train, Y_train, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

# run using test data
scores = loaded_model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

# predict a smile/no smile from an image

img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) # x is now (1,64,64,3)

print(loaded_model.predict(x))