import CNN 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())


EPOCHS = 25
LR = 0.001
BS = 32

# initialize the data and labels
print("loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	# get labels from path name
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "pizza" else 0
	labels.append(label)

#  raw pixel to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# One hot encoding
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

#  data augmentation prevents overfitting
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("compiling model...")
model = CNN.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=LR, decay=LR / EPOCHS) # combination of Momentum and RMSPROP optimization
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# training the network
print("training network...")

model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("Saving model...")
model.save(args["model"])