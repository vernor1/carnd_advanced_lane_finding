import argparse
import cv2
import numpy as np

from lens_correction import TLensCorrector
from binary_thresholding import GetThresholdedBinary
from perspective_transform import TPerspectiveTransformer


CAMERA_CALIBRATION_DIR = "camera_calibration"


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Binary Image")
    argParser.add_argument("in_img", type=str, help="Path to an image file")
    argParser.add_argument("out_img",
                           type=str,
                           help="Path to the test file")
    args = argParser.parse_args()
    img = cv2.imread(args.in_img)

    lensCorrector = TLensCorrector(CAMERA_CALIBRATION_DIR)
    undistortedImg = lensCorrector.Undistort(img)

    thresholdedImg = GetThresholdedBinary(undistortedImg) * 255

    perspectiveTransformer = TPerspectiveTransformer((img.shape[1], img.shape[0]))
    warped = perspectiveTransformer.Warp(thresholdedImg)

    cv2.imwrite(args.out_img, warped)