import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from lens_correction import TLensCorrector


CAMERA_CALIBRATION_DIR = "camera_calibration"


class TPerspectiveTransformer():
    # Constants ---------------------------------------------------------------
    LEFT_BOTTOM = (193, 719)
    LEFT_TOP = (579, 460)
    RIGHT_TOP = (703, 460)
    RIGHT_BOTTOM = (1122, 719)

    # Public Members ----------------------------------------------------------
    def __init__(self, imgSize):
        src = np.float32([self.LEFT_BOTTOM, self.LEFT_TOP, self.RIGHT_TOP, self.RIGHT_BOTTOM])
        dst = np.float32([[imgSize[0]/4, imgSize[1]-1], [imgSize[0]/4, 0], [imgSize[0]*3/4, 0], [imgSize[0]*3/4, imgSize[1]-1]])
        self.M = cv2.getPerspectiveTransform(src, dst)

    def Warp(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


# The following code is only used for debugging and generating test images
#------------------------------------------------------------------------------
if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="Perspective Transform")
    argParser.add_argument("in_img", type=str, help="Path to an image file")
    argParser.add_argument("out_plot",
                           type=str,
                           help="Path to the plot file of a side-by-side comparison of the original and warped images")
    args = argParser.parse_args()
    img = cv2.imread(args.in_img)

    lensCorrector = TLensCorrector(CAMERA_CALIBRATION_DIR)
    undistortedImg = lensCorrector.Undistort(img)

    perspectiveTransformer = TPerspectiveTransformer((img.shape[1], img.shape[0]))
    warped = perspectiveTransformer.Warp(undistortedImg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.tight_layout()
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image", fontsize=20)
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax2.set_title("Undistorted and Warped Image", fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.995, bottom=0.005)
    fig.savefig(args.out_plot)
