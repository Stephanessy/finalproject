import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import math
import copy
import pydotplus


train_path = 'data/train'
valid_path = 'data/validation'
test_path = 'data/test'
top_model_weights_path = 'model_weigh.h5'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 2

img_width, img_height = 224, 224

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1./255)


train_batches = train_datagen.flow_from_directory(train_path,
                                                  target_size=(img_width, img_height),
                                                  classes=None,
                                                  class_mode='categorical',
                                                  batch_size=batch_size,
                                                  shuffle=True)
valid_batches = valid_datagen.flow_from_directory(valid_path,
                                                  target_size=(img_width, img_height),
                                                  classes=None,
                                                  class_mode='categorical',
                                                  batch_size=batch_size,
                                                  shuffle=True)
test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                        target_size=(img_width, img_height),
                                                        classes=None,
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=False)

nb_train_samples = len(train_batches.filenames)  # get the size of the training set
nb_classes_train = len(train_batches.class_indices)  # get the number of classes
predict_size_train = int(math.ceil(nb_train_samples / batch_size))

nb_valid_samples = len(valid_batches.filenames)
nb_classes_valid = len(valid_batches.class_indices)
predict_size_validation = int(math.ceil(nb_valid_samples / batch_size))

nb_test_samples = len(test_batches.filenames)
nb_classes_test = len(test_batches.class_indices)
predict_size_test = int(math.ceil(nb_test_samples / batch_size))

# load json file and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# build fine tune vgg16 model
vgg16_model = keras.applications.vgg16.VGG16()
# vgg16_model.summary()
loaded_model = Sequential()
for layer in vgg16_model.layers[0: -2]:
    loaded_model.add(layer)


for layer in loaded_model.layers:
    layer.trainable = False

loaded_model.add(Dense(nb_classes_train, activation='softmax'))
loaded_model.layers[-1].trainable = True

# load weight into new model
loaded_model.load_weights(top_model_weights_path)
print("loaded model from disk")

#
loaded_model.compile(optimizer=Adam(lr=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

predictions = loaded_model.predict_generator(test_batches,
                                      steps=nb_test_samples / batch_size,
                                      verbose=0)
# print(predictions)

predicted_class_indices = np.argmax(predictions, axis=1)
print(predicted_class_indices)

labels = train_batches.class_indices
labels = dict((v, k) for k, v in labels.items())
final_predictions = [labels[k] for k in predicted_class_indices]
print(final_predictions)

# save as csv file
filenames = test_batches.filenames
results = pd.DataFrame({"Filename": filenames,
                        "Predictions": final_predictions})
results.to_csv("loaded_results.csv", index=False)

# evaluation test result
(test_loss, test_accuracy) = loaded_model.evaluate_generator(
    test_batches,
    steps=nb_train_samples // batch_size,
    verbose=1)
print("[INFO] test accuracy: {:.2f}%".format(test_accuracy * 100))
print("[INFO] test loss: {}".format(test_loss))

