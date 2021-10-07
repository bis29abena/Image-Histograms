# Usage
# python main.py --image image-name
# import the necessary packages
from matplotlib import pyplot as plt
import cv2 as cv
import argparse
import numpy as np

def plot_mask_image(image, title, mask=None):
    # Split the images to it respectives channels and we
    # and we initialize our colors with our corresponding
    # channel images
    chans = cv.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")

    # we loop over the channels and colors
    for (chan, color) in zip(chans, colors):
        # we create a histogram for the current channel and plot it
        hist = cv.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


# Construct an argparser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input file", required=True, type=str)
args = vars(ap.parse_args())

# load the input image and converte it to grayscale
# we plot a histogram for that image
image = cv.imread(args["image"])
plot_mask_image(image, "Histogram for original image")
cv.imshow("Original", image)

# we construct a mask for image: our mask will be "black"
# for regions we want to ignore and "white" for regions we want to examine
mask = np.zeros(image.shape[:2], dtype="uint8")
cv.rectangle(mask, (95, 60), (450, 450), 255, -1)
cv.imshow("Mask", mask)

# Display the mask region on the image
masked = cv.bitwise_and(image, image, mask=mask)
cv.imshow("Appling The mask", masked)

# compute the histogram for our image, but we only include the mask region
plot_mask_image(image, "Histogram of masked region", mask=mask)
plt.show()
cv.waitKey(0)