import cv2
import numpy as np


def GetSobelBinary(channel, orient='x', sobelKernel=3, thresh=(0, 255)):
    """ Applies the Sobel-function to a channel image and generates the binary image.

    param: channel: 2D array of the image channel
    param: orient: Axis for the derivative, 'x' or 'y'
    param: sobelKernel: Sobel kernel size, must be odd
    param: thresh: Tuple of min, max threshold
    returns: Binary image
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    else:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobelKernel)
    # Take the absolute value of the derivative or gradient
    absSobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaledSobel = np.uint8(255 * absSobel / np.max(absSobel))
    # Create a mask of 1's where the scaled gradient magnitude is >= thresh_min and =< thresh_max
    outBinary = np.zeros(scaledSobel.shape, dtype=np.uint8)
    outBinary[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1
    return outBinary

def GetMagnitudeBinary(channel, sobelKernel=3, thresh=(0, 255)):
    """ Computes the magnitude of the gradient of a channel image and generates the binary image.

    param: channel: 2D array of the image channel
    param: sobelKernel: Sobel kernel size, must be odd
    param: thresh: Tuple of min, max threshold
    returns: Binary image
    """
    # Take the gradient in x and y separately
    sobelX = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    sobelY = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobelKernel)
    # Calculate the magnitude 
    magSobel = np.sqrt(np.add(np.square(sobelX), np.square(sobelY)))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaledSobel = np.uint8(255 * magSobel / np.max(magSobel))
    # Create a binary mask where mag thresholds are met
    outBinary = np.zeros(scaledSobel.shape, dtype=np.uint8)
    outBinary[(scaledSobel >= thresh[0]) & (scaledSobel <= thresh[1])] = 1
    return outBinary

def GetDirectionBinary(channel, sobelKernel=3, thresh=(0, np.pi/2)):
    """ Computes the direction of the gradient of a channel image and generates the binary image.

    param: channel: 2D array of the image channel
    param: sobelKernel: Sobel kernel size, must be odd
    param: thresh: Tuple of min, max threshold in radians
    returns: Binary image
    """
    # Take the gradient in x and y separately
    sobelX = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    sobelY = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobelKernel)
    # Take the absolute value of the x and y gradients
    absSobelX = np.absolute(sobelX)
    absSobelY = np.absolute(sobelY)
    # Use np.arctan2(absSobelY, absSobelX) to calculate the direction of the gradient 
    gradientDirection = np.arctan2(absSobelY, absSobelX)
    # Create a binary mask where direction thresholds are met
    outBinary = np.zeros(gradientDirection.shape, dtype=np.uint8)
    outBinary[(gradientDirection > thresh[0]) & (gradientDirection < thresh[1])] = 1
    return outBinary

def GetChannelBinary(channel, thresh=(0, 255)):
    """ Computes the threshold of a channel image and generates the binary image.

    param: channel: 2D array of the image channel
    param: thresh: Tuple of min, max threshold
    returns: Binary image
    """
    outBinary = np.zeros(channel.shape, dtype=np.uint8)
    outBinary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return outBinary

def GetThresholdedBinary(img):
    """ Applies a combination of the function above to generate the binary image revealing lane lines.

    param: channel: 2D array of the image channel
    returns: Binary image
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    channelS = hls[:,:,2]
    channelBinaryS = GetChannelBinary(channelS, thresh=(90, 255))
    channelH = hls[:,:,0]
    channelBinaryH = GetChannelBinary(channelH, thresh=(15, 100))
    channelR = img[:,:,0]
    channelBinaryR = GetChannelBinary(channelR, thresh=(200, 255))
    combinedColors = np.zeros_like(channelBinaryS)
    # S-channel can capture shadows on the lane, so we should use it in conjunction with H-channel, which captures same color regions.
    # R-channel capures white lines only and can form a union with S and H.
    combinedColors[((channelBinaryS == 1) & (channelBinaryH == 1)) | (channelBinaryR == 1)] = 1
    # Use Y-channel providing the most information about lane lines as the base channel for gradients
    baseChannel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradX = GetSobelBinary(baseChannel, orient='x', sobelKernel=5, thresh=(35, 200))
    gradY = GetSobelBinary(baseChannel, orient='y', sobelKernel=5, thresh=(35, 200))
    magBinary = GetMagnitudeBinary(baseChannel, sobelKernel=7, thresh=(50, 250))
    dirBinary = GetDirectionBinary(baseChannel, sobelKernel=9, thresh=(0.7, 1.3))
    combinedGradients = np.zeros_like(dirBinary)
    combinedGradients[((gradX == 1) & (gradY == 1)) | ((magBinary == 1) & (dirBinary == 1))] = 1
    combinedBinary = np.zeros_like(combinedGradients)
    combinedBinary[(combinedGradients == 1) | (combinedColors == 1)] = 1
    return combinedBinary
