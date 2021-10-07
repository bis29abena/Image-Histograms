# Computing the gray scale image intensities
# Usage
# python main.py --image image-name
# import the necessary packages
from matplotlib import pyplot as plt
import cv2 as cv
import argparse

# Construct an argparser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input file", required=True, type=str)
args = vars(ap.parse_args())

# load the input image and converte it to grayscale
image = cv.imread(args["image"])
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# cv.imshow("gray", image)

# compute the grayscale histogram
hist = cv.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# display the image in matplotlib
# matplotlib experts RGB images so we visualize using the .gray function
plt.figure()
plt.axis("off")
plt.imshow(image)
plt.gray()

# plt the hitogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0, 256])

# normalize the histogram
hist /= hist.sum()
plt.figure()
plt.title("Normalize Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
