import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from binary_thresholding import GetThresholdedBinary, GetSobelBinary, GetMagnitudeBinary, GetDirectionBinary, GetChannelBinary
from lane_tracking import TLaneTracker
from lens_correction import TLensCorrector
from perspective_transform import TPerspectiveTransformer


CAMERA_CALIBRATION_DIR = "camera_calibration"


# The following code is only used for debugging and generating test images
#------------------------------------------------------------------------------
if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Test Pipeline Components")
    argParser.add_argument("type",
                           choices=["lens_correction", "binary_thresholding", "perspective_transform", "lane_tracking"])
    argParser.add_argument("in_img",
                           type=str,
                           help="Path to the original image file")
    argParser.add_argument("out_img",
                           type=str,
                           help="Path to the plot file of a side-by-side comparison of the distorted and undistorted images")
    args = argParser.parse_args()

    img = cv2.imread(args.in_img)
    lensCorrector = TLensCorrector(CAMERA_CALIBRATION_DIR)
    undistortedImg = lensCorrector.Undistort(img)

    if args.type == "lens_correction":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))
        fig.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image", fontsize=20)
        ax2.imshow(cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2RGB))
        ax2.set_title("Undistorted Image", fontsize=20)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.995, bottom=0.005)
        fig.savefig(args.out_img)

    elif args.type == "binary_thresholding":
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
        fig.savefig(args.out_img)

    elif args.type == "perspective_transform":
        perspectiveTransformer = TPerspectiveTransformer((undistortedImg.shape[1], undistortedImg.shape[0]))
        warpedImg = perspectiveTransformer.Warp(undistortedImg)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.tight_layout()
        ax1.imshow(cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2RGB))
        ax1.set_title("Undistorted Image", fontsize=20)
        ax2.imshow(cv2.cvtColor(warpedImg, cv2.COLOR_BGR2RGB))
        ax2.set_title("Warped Image", fontsize=20)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.995, bottom=0.005)
        fig.savefig(args.out_img)

    elif args.type == "lane_tracking":
        perspectiveTransformer = TPerspectiveTransformer((undistortedImg.shape[1], undistortedImg.shape[0]))
        thresholdedBinary = GetThresholdedBinary(undistortedImg)
        warpedBinary = perspectiveTransformer.Warp(thresholdedBinary)
        laneTracker = TLaneTracker((warpedBinary.shape[1], warpedBinary.shape[0]))
        leftCoefficients, rightCoefficients, curveRad, deviation = laneTracker.ProcessLaneImage(warpedBinary)
        # Generate x and y values for plotting
        plotY = np.linspace(0, warpedBinary.shape[0] - 1, warpedBinary.shape[0])
        leftPlotX = leftCoefficients[0] * plotY**2 + leftCoefficients[1] * plotY + leftCoefficients[2]
        rightPlotX = rightCoefficients[0] * plotY**2 + rightCoefficients[1] * plotY + rightCoefficients[2]
        # Fill the lane surface
        laneImg = np.zeros_like(undistortedImg)
        # Recast the x and y points into usable format for cv2.fillPoly()
        leftPoints = np.array([np.transpose(np.vstack([leftPlotX, plotY]))])
        rightPoints = np.array([np.flipud(np.transpose(np.vstack([rightPlotX, plotY])))])
        lanePoints = np.hstack((leftPoints, rightPoints))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(laneImg, np.int_([lanePoints]), (255, 0, 0))
        # Draw lines
        cv2.polylines(laneImg, np.int_([leftPoints]), isClosed=False, color=(0, 0, 255), thickness=32)
        cv2.polylines(laneImg, np.int_([rightPoints]), isClosed=False, color=(0, 255, 0), thickness=32)
        unwarpedLane = perspectiveTransformer.Unwarp(laneImg)
        outImg = cv2.addWeighted(undistortedImg, 1, unwarpedLane, 0.3, 0)
        if deviation < 0:
            deviationDirection = "left"
        else:
            deviationDirection = "right"
        deviation = np.absolute(deviation)
        cv2.putText(outImg, "Curvature radius: %dm" % (curveRad),
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        cv2.putText(outImg, "Deviation: %.2fm %s" % (deviation, deviationDirection),
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))
        fig.tight_layout()
        ax1.imshow(cv2.cvtColor(undistortedImg, cv2.COLOR_BGR2RGB))
        ax1.set_title("Undistorted Image", fontsize=20)
        ax2.imshow(cv2.cvtColor(warpedBinary * 255, cv2.COLOR_GRAY2RGB))
#        cv2.imwrite("test_images/warped.png", cv2.cvtColor(warpedBinary * 255, cv2.COLOR_GRAY2RGB))
        plt.rcParams['lines.linewidth'] = 7
        ax2.plot(leftPlotX, plotY, color='red')
        ax2.plot(rightPlotX, plotY, color='green')
        ax2.set_title("Warped Binary with Detected Lines", fontsize=20)
        ax3.imshow(cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB))
        ax3.set_title("Undistorted with Overlay", fontsize=20)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        fig.savefig(args.out_img)
