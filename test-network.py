import numpy as np
import argparse
import imutils
import cv2
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Dont like the annoying cpu message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


try:
	image = cv2.imread(args["image"])
	orig = image.copy()
except:	
	print ("THIS IMAGE DONT EXIST MATE")


image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


print("loading network...")
model = load_model(args["model"])

# classify the input image
(not_pizza, pizza) = model.predict(image)[0]

classes = model.predict_classes(image)

label = "Pizza" if classes == 1 else "Not Pizza"
probability = pizza if pizza > not_pizza else not_pizza
probability = round((probability * 100),2)
label = "{}: {}%".format(label,probability)


# Output image with probablity and label
output = imutils.resize(orig, width=400)
font_color = (0, 255, 0) if pizza > not_pizza else (0, 0, 255)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, font_color, 2)

cv2.imshow("Output", output)
cv2.waitKey(0)