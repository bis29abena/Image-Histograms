# Usage
# python main.py --image image-name
# import the necessary packages
import imutils
from matplotlib import pyplot as plt
import cv2 as cv
import argparse

# Construct an argparser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input file", required=True, type=str)
args = vars(ap.parse_args())

# load the input image and converte it to grayscale
image = cv.imread(args["image"])

# split the the image into its respective channels, then initialize
# the tuple of channel names along with our figure for plotting
chans = cv.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")

# loop over the image channels
for (chan, color) in zip(chans, colors):
    # Create a histogram for the current channel and plot it
    hist = cv.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

# create a new figure and create a 2D color histogram for the green and blue channel
fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color Histogram for G and B")
plt.colorbar(p)

# plot a 2D color histogram for the green and red
ax = fig.add_subplot(132)
hist = cv.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color Histogram for G and R")
plt.colorbar(p)

# plot a 2D color histogram for the blue and red
ax = fig.add_subplot(133)
hist = cv.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color Histogram for B and R")
plt.colorbar(p)

# finally we display the dimensionality of one of the 2D histograms
print(f"2D Histogram Shape: {hist.shape}, with {hist.flatten().shape[0]} values")

# our 2D histogram could only take into account 2 out of the 3 channels in the image
# so now lets build a 3D color histogram with 8 Bins in each direction
# we cant plot a 3D histogram, but the theory is exactly like that of a 2D histogram
# so we will just show the shape of the histogram
hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print(f"3D histogram Shape: {hist.shape}, with {hist.flatten().shape[0]} values")

# display the original input image
plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(image))

plt.show()