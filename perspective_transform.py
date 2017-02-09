import cv2
import numpy as np


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
