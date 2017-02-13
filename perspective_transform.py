import cv2
import numpy as np


class TPerspectiveTransformer():
    """ Perspective transformer class.
    """
    # Constants ---------------------------------------------------------------
    # Manually captured on straight_lines1.jpg
    LEFT_BOTTOM = (193, 719)
    LEFT_TOP = (595, 449)
    RIGHT_TOP = (685, 449)
    RIGHT_BOTTOM = (1122, 719)

    # Public Members ----------------------------------------------------------
    def __init__(self, imgSizeX, imgSizeY):
        """ TPerspectiveTransformer ctor.

        param: imgSizeX: X-dimension of the image to be transformed
        param: imgSizeX: Y-dimension of the image to be transformed
        """
        src = np.float32([self.LEFT_BOTTOM, self.LEFT_TOP, self.RIGHT_TOP, self.RIGHT_BOTTOM])
        dst = np.float32([[imgSizeX/4, imgSizeY-1], [imgSizeX/4, 0], [imgSizeX*3/4, 0], [imgSizeX*3/4, imgSizeY-1]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def Warp(self, img):
        """ Warps an image.

        param: img: Image to warp
        returns: Warped image
        """
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def Unwarp(self, img):
        """ Unwarps an image.

        param: img: Image to unwarp
        returns: Unwarped image
        """
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
