import cv2
import numpy as np


class TLaneTracker():
    class TLine():
        # Constants -----------------------------------------------------------
        NUMBER_OF_SLIDING_WINDOWS = 9
        SLIDING_WINDOWS_WIDTH = 200
        MIN_NUMBER_OF_PIXELS = 50
        # Width of lane is 12ft or 3.7m
        M_PER_PIX_HORIZONTAL = 3.7 / 615
        # Length of dashed lane line is 10ft or 3m
        M_PER_PIX_VERTICAL = 3 / 100

        # Public Members ------------------------------------------------------
        def __init__(self, imgSize, offsetX=0):
            self.ImgSize = imgSize
            self.OffsetX = offsetX

        def ProcessLineImage(self, img):
            assert img.shape[0] == self.ImgSize[1] and img.shape[1] == self.ImgSize[0]
            # Take a histogram of the bottom half of the image
            histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
            # Find the peak of the histogram - it will be the starting point for the line
            baseX = np.argmax(histogram)
            # Set height of windows
            windowHeight = img.shape[0] // self.NUMBER_OF_SLIDING_WINDOWS
            # Identify the x and y positions of all nonzero pixels in the image
            nonZero = img.nonzero()
            nonZeroY = np.array(nonZero[0])
            nonZeroX = np.array(nonZero[1])
            # Current position to be updated for each window
            currentX = baseX
            # Create empty list to receive line pixel indices
            lineIndices = []
            # Step through the windows one by one
            for windowNbr in range(self.NUMBER_OF_SLIDING_WINDOWS):
                # Identify window boundaries in x and y (and right and left)
                windowLowY = img.shape[0] - (windowNbr + 1) * windowHeight
                windowHighY = img.shape[0] - windowNbr * windowHeight
                windowLowX = currentX - self.SLIDING_WINDOWS_WIDTH // 2
                windowHighX = currentX + self.SLIDING_WINDOWS_WIDTH // 2
                # Identify the nonzero pixels in x and y within the window
                windowLineIndices = ((nonZeroY >= windowLowY)
                                   & (nonZeroY < windowHighY)
                                   & (nonZeroX >= windowLowX)
                                   & (nonZeroX < windowHighX)).nonzero()[0]
                # Append these indices to the list
                lineIndices.append(windowLineIndices)
                # If you found >= minpix pixels, recenter next window on their mean position
                if len(windowLineIndices) >= self.MIN_NUMBER_OF_PIXELS:
                    currentX = np.int(np.mean(nonZeroX[windowLineIndices]))
            # Concatenate the arrays of indices
            lineIndices = np.concatenate(lineIndices)
            # Extract line points
            pointsX = nonZeroX[lineIndices]
            pointsY = nonZeroY[lineIndices]
            # Fit a second order polynomial
            pxCoefficients = np.polyfit(pointsY, pointsX + self.OffsetX, 2)
            # Fit new polynomials to x,y in world space
            mCoefficients = np.polyfit(pointsY * self.M_PER_PIX_VERTICAL, (pointsX + self.OffsetX) * self.M_PER_PIX_HORIZONTAL, 2)
            # Define y-position where we want radius of curvature
            curveY = img.shape[0]
            # Calculate radius of curvature
            curveRad = ((1 + (2 * mCoefficients[0] * curveY * self.M_PER_PIX_VERTICAL + mCoefficients[1]) ** 2) ** 1.5) \
                       / np.absolute(2 * mCoefficients[0])
            return pxCoefficients, curveRad, baseX + self.OffsetX

    # Public Members ----------------------------------------------------------
    def __init__(self, imgSize):
        self.ImgSize = imgSize
        self.LeftLine = self.TLine((imgSize[0]//2, imgSize[1]))
        self.RightLine = self.TLine((imgSize[0]//2, imgSize[1]), imgSize[0]//2)

    def ProcessLaneImage(self, img):
        assert img.shape[0] == self.ImgSize[1] and img.shape[1] == self.ImgSize[0]
        leftLineCoefficients, leftCurveRad, leftX = self.LeftLine.ProcessLineImage(img[:, :img.shape[1]//2])
        rightLineCoefficients, rightCurveRad, rightX = self.RightLine.ProcessLineImage(img[:, img.shape[1]//2:])
        deviation = (img.shape[1] // 2 - (leftX + rightX) // 2) * self.TLine.M_PER_PIX_HORIZONTAL
        return leftLineCoefficients, rightLineCoefficients, (leftCurveRad + rightCurveRad) / 2, deviation
