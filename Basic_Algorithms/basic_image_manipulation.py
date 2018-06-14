# Author: Raoul Malm
# Description: Basic Image Manipulation with openCV in Python.

import numpy as np
import cv2
from matplotlib import pyplot as plt

print('openCV version: {}'.format(cv2.__version__))

# Load an color image in grayscale
img = cv2.imread('neuron.jpg',0)
print("image shape: {}".format(img.shape))
print("data type: {}".format(img.dtype))

# Show the image with matplotlib
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# Show the image with opencv
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Write image
#img = cv2.imwrite('neuron-gray.jpg', img)

# show all color flags
#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#print(flags) 



