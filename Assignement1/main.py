from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import numpy as np
import cv2

"""
#   Gaussian Blur
"""
fig = plt.figure()

plt.gray()  # show the blured image in grayscale

#   Reading the image then applying a gaussian blur with a 5x5 kernel
img = cv2.imread('Lena-Gray-3.png', cv2.IMREAD_UNCHANGED)
result = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)

#   Creating the subplot to compare the two images
ax1 = fig.add_subplot(121)  #
ax2 = fig.add_subplot(122)

#   Plotting the two images
ax1.imshow(img)
ax2.imshow(result)
plt.show()

"""
#   Convolution function ( https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/ )
"""
def convolve(image, kernel):
    #   Getting the dimentions of both kernel and image
    (imgHeight, imageWidth) = image.shape[:2]
    (kernelHeight, kernelWidth) = kernel.shape[:2]
    #   Create a copy of the image but with padded border to handle edge of the image
    pad = (kernelWidth - 1) // 2
    imgCopy = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((imgHeight, imageWidth), dtype="float32")

    #   Iterating "sliding" the kernel accross all the image
    for y in np.arange(pad, imgHeight + pad):
        for x in np.arange(pad, imageWidth + pad):
            #   get the region of interest at the current coordinates
            roi = imgCopy[y - pad:y + pad + 1, x - pad:x + pad + 1]
            #   Convolute (multiplicateROI * Kernel) then summing the matrix
            k = (roi * kernel).sum()
            #   Place the output on the right spot in the output image
            output[y - pad, x - pad] = k
    #   rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

# Applying Laplacian Operator on the image filtered with Gaussian blur
fig = plt.figure()
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side

laplacianKernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

result_laplace = convolve(result,laplacianKernel)

ax1.imshow(img)

ax2.imshow(result_laplace)

plt.show()

# Applying Laplacian Operator on the image without the Gaussian blur filter
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
result_laplace = convolve(img,laplacianKernel)
ax1.imshow(img)
ax2.imshow(result_laplace)
plt.show()

# Sources:
## https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/