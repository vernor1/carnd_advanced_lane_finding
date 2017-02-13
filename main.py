import argparse
import cv2
import numpy as np

from binary_thresholding import GetThresholdedBinary
from lane_tracking import TLaneTracker
from lens_correction import TLensCorrector
from moviepy.editor import VideoFileClip
from perspective_transform import TPerspectiveTransformer

# Global Variables ------------------------------------------------------------
LensCorrector = TLensCorrector("camera_calibration")
LaneTracker = TLaneTracker()
PerspectiveTransformer = TPerspectiveTransformer(1280, 720)

# Functions ------------------------------------------------------------
def ProcessImage(img):
    """ Processes an RGB image by detecting the lane lines, radius of curvature and course deviation.
        The information is added to the undistorded original image in overlay.

    param: img: Image to process
    returns: Processed RGB image
    """
    # Convert the RGB image of MoviePy to BGR format of OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Undistort the image
    undistortedImg = LensCorrector.Undistort(img)
    # Transform
    thresholdedBinary = GetThresholdedBinary(undistortedImg)
    # Generate a bird's eye view
    warpedBinary = PerspectiveTransformer.Warp(thresholdedBinary)
    # Detect the lane lines, radius of curvature and course deviation
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
    cv2.fillPoly(laneImg, np.int_([lanePoints]), (255, 0, 0))
    # Draw the lane lines
    cv2.polylines(laneImg, np.int_([leftPoints]), isClosed=False, color=(0, 0, 255), thickness=32)
    cv2.polylines(laneImg, np.int_([rightPoints]), isClosed=False, color=(0, 255, 0), thickness=32)
    # Convert the lane image from the bird's eye view to the original perspective
    unwarpedLane = PerspectiveTransformer.Unwarp(laneImg)
    # Add the lane lines overlay
    outImg = cv2.addWeighted(undistortedImg, 1, unwarpedLane, 0.3, 0)
    # Add the radius of curvature overlay
    cv2.putText(outImg, "Curvature radius: %dm" % (curveRad),
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    # Add the course deviation overlay
    if deviation < 0:
        deviationDirection = "left"
    else:
        deviationDirection = "right"
    deviation = np.absolute(deviation)
    cv2.putText(outImg, "Deviation: %.2fm %s" % (deviation, deviationDirection),
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    # Convert the processed image back to the RGB format comatible with MoviePy
    outImg = cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB)
    return outImg

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Main Pipeline")
    argParser.add_argument("in_clip", type=str, help="Path to the original clip")
    argParser.add_argument("out_clip", type=str, help="Path to the clip with the lane overlay")
    args = argParser.parse_args()
    inClip = VideoFileClip(args.in_clip)
    outClip = inClip.fl_image(ProcessImage)
    outClip.write_videofile(args.out_clip, audio=False)
