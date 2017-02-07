import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np


def SaveBinaryImage(img, fileName):
#    outImg = np.array(inImg * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(fileName, img)

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def GetSobelBinary(channel, orient='x', sobelKernel=3, thresh=(0, 255)):
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

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def GetMagnitudeBinary(channel, sobelKernel=3, thresh=(0, 255)):
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

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def GetDirectionBinary(channel, sobelKernel=3, thresh=(0, np.pi/2)):
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
    outBinary = np.zeros(channel.shape, dtype=np.uint8)
    outBinary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return outBinary

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Binary Image")
    argParser.add_argument("in_img", type=str, help="Path to an image file")
    argParser.add_argument("out_plot",
                           type=str,
                           help="Path to the plot file")
    args = argParser.parse_args()

    img = cv2.imread(args.in_img)
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

    # Use S-channel providing the most information about lane lines as the base channel for gradients
    baseChannel = channelS
    gradX = GetSobelBinary(baseChannel, orient='x', sobelKernel=5, thresh=(20, 200))
    gradY = GetSobelBinary(baseChannel, orient='y', sobelKernel=5, thresh=(20, 200))
    magBinary = GetMagnitudeBinary(baseChannel, sobelKernel=7, thresh=(50, 240))
    dirBinary = GetDirectionBinary(baseChannel, sobelKernel=9, thresh=(0.7, 1.3))
    combinedGradients = np.zeros_like(dirBinary)
    combinedGradients[((gradX == 1) & (gradY == 1)) | ((magBinary == 1) & (dirBinary == 1))] = 1

    combinedBinary = np.zeros_like(combinedGradients)
    combinedBinary[(combinedGradients == 1) | (combinedColors == 1)] = 1

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14) = plt.subplots(14, 1, figsize=(10, 82))
    fig.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image", fontsize=20)
    ax2.imshow(cv2.cvtColor(channelS, cv2.COLOR_GRAY2RGB))
    ax2.set_title("S-channel", fontsize=20)
    ax3.imshow(cv2.cvtColor(channelBinaryS * 255, cv2.COLOR_GRAY2RGB))
    ax3.set_title("S-channel Binary", fontsize=20)
    ax4.imshow(cv2.cvtColor(channelH, cv2.COLOR_GRAY2RGB))
    ax4.set_title("H-channel", fontsize=20)
    ax5.imshow(cv2.cvtColor(channelBinaryH * 255, cv2.COLOR_GRAY2RGB))
    ax5.set_title("H-channel Binary", fontsize=20)
    ax6.imshow(cv2.cvtColor(channelR, cv2.COLOR_GRAY2RGB))
    ax6.set_title("R-channel", fontsize=20)
    ax7.imshow(cv2.cvtColor(channelBinaryR * 255, cv2.COLOR_GRAY2RGB))
    ax7.set_title("R-channel Binary", fontsize=20)
    ax8.imshow(cv2.cvtColor(combinedColors * 255, cv2.COLOR_GRAY2RGB))
    ax8.set_title("Combined Colors", fontsize=20)
    ax9.imshow(cv2.cvtColor(gradX * 255, cv2.COLOR_GRAY2RGB))
    ax9.set_title("X-gradient", fontsize=20)
    ax10.imshow(cv2.cvtColor(gradY * 255, cv2.COLOR_GRAY2RGB))
    ax10.set_title("Y-gradient", fontsize=20)
    ax11.imshow(cv2.cvtColor(magBinary * 255, cv2.COLOR_GRAY2RGB))
    ax11.set_title("Gradient Magnitude", fontsize=20)
    ax12.imshow(cv2.cvtColor(dirBinary * 255, cv2.COLOR_GRAY2RGB))
    ax12.set_title("Gradient Direction", fontsize=20)
    ax13.imshow(cv2.cvtColor(combinedGradients * 255, cv2.COLOR_GRAY2RGB))
    ax13.set_title("Combined Gradients", fontsize=20)
    ax14.imshow(cv2.cvtColor(combinedBinary * 255, cv2.COLOR_GRAY2RGB))
    ax14.set_title("Combined Binary", fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.995, bottom=0.005)
    fig.savefig(args.out_plot)
