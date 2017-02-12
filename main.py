import argparse
import cv2
import numpy as np

from binary_thresholding import GetThresholdedBinary
from lane_tracking import TLaneTracker
from lens_correction import TLensCorrector
from moviepy.editor import VideoFileClip
from perspective_transform import TPerspectiveTransformer

# Constants -------------------------------------------------------------------
CAMERA_CALIBRATION_DIR = "camera_calibration"

# Global Variables ------------------------------------------------------------
LensCorrector = TLensCorrector(CAMERA_CALIBRATION_DIR)
LaneTracker = TLaneTracker()

# FIXME: remove
FRAME_RANGE = (1, 100)
FrameNumber = 0

# Functions ------------------------------------------------------------
def ProcessImage(img):
    # FIXME: remove
#    global FRAME_RANGE
#    global FrameNumber
#    FrameNumber += 1
#    if FrameNumber not in range(FRAME_RANGE[0], FRAME_RANGE[1]):
#        return img

    undistortedImg = LensCorrector.Undistort(img)
    perspectiveTransformer = TPerspectiveTransformer((undistortedImg.shape[1], undistortedImg.shape[0]))
    thresholdedBinary = GetThresholdedBinary(undistortedImg)
    warpedBinary = perspectiveTransformer.Warp(thresholdedBinary)
    leftCoefficients, rightCoefficients, curveRad, deviation = LaneTracker.ProcessLaneImage(warpedBinary)
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
    cv2.fillPoly(laneImg, np.int_([lanePoints]), (0, 0, 255))
    # Draw lines
    cv2.polylines(laneImg, np.int_([leftPoints]), isClosed=False, color=(255, 0, 0), thickness=32)
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
    return outImg

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Main Pipeline")
    argParser.add_argument("in_clip", type=str, help="Path to the original clip")
    argParser.add_argument("out_clip", type=str, help="Path to the clip with the lane overlay")
    args = argParser.parse_args()
    inClip = VideoFileClip(args.in_clip)
    outClip = inClip.fl_image(ProcessImage)
    outClip.write_videofile(args.out_clip, audio=False)
