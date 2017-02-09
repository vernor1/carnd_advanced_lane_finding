import cv2
import numpy as np


class TLineDetector():
    # Constants ---------------------------------------------------------------
    NUMBER_OF_SLIDING_WINDOWS = 9
    SLIDING_WINDOWS_WIDTH = 200
    MIN_NUMBER_OF_PIXELS = 50

    # Public Members ----------------------------------------------------------
    def __init__(self):
        print("Initialized")

    def ProcessLaneImage(self, img):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        centerX = histogram.shape[0]//2
        baseLeftX = np.argmax(histogram[:centerX])
        baseRightX = np.argmax(histogram[centerX:]) + centerX

        # Set height of windows
        windowHeight = img.shape[0] // self.NUMBER_OF_SLIDING_WINDOWS

        # Identify the x and y positions of all nonzero pixels in the image
        nonZero = img.nonzero()
        nonZeroY = np.array(nonZero[0])
        nonZeroX = np.array(nonZero[1])

        # Current positions to be updated for each window
        currentLeftX = baseLeftX
        currentRightX = baseRightX

        # Create empty lists to receive left and right lane pixel indices
        leftLineIndices = []
        rightLineIndices = []

        # Step through the windows one by one
        for windowNbr in range(self.NUMBER_OF_SLIDING_WINDOWS):
            # Identify window boundaries in x and y (and right and left)
            windowLowY = img.shape[0] - (windowNbr + 1) * windowHeight
            windowHighY = img.shape[0] - windowNbr * windowHeight
            leftWindowLowX = currentLeftX - self.SLIDING_WINDOWS_WIDTH // 2
            leftWindowHighX = currentLeftX + self.SLIDING_WINDOWS_WIDTH // 2
            rightWindowLowX = currentRightX - self.SLIDING_WINDOWS_WIDTH // 2
            rightWindowHighX = currentRightX + self.SLIDING_WINDOWS_WIDTH // 2

            # Identify the nonzero pixels in x and y within the window
            windowLeftLineIndices = ((nonZeroY >= windowLowY)
                                   & (nonZeroY < windowHighY)
                                   & (nonZeroX >= leftWindowLowX)
                                   & (nonZeroX < leftWindowHighX)).nonzero()[0]
            windowRightLineIndices = ((nonZeroY >= windowLowY)
                                    & (nonZeroY < windowHighY)
                                    & (nonZeroX >= rightWindowLowX)
                                    & (nonZeroX < rightWindowHighX)).nonzero()[0]

            # Append these indices to the lists
            leftLineIndices.append(windowLeftLineIndices)
            rightLineIndices.append(windowRightLineIndices)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(windowLeftLineIndices) >= self.MIN_NUMBER_OF_PIXELS:
                currentLeftX = np.int(np.mean(nonZeroX[windowLeftLineIndices]))
            if len(windowRightLineIndices) >= self.MIN_NUMBER_OF_PIXELS:        
                currentRightX = np.int(np.mean(nonZeroX[windowRightLineIndices]))

        # Concatenate the arrays of indices
        leftLineIndices = np.concatenate(leftLineIndices)
        rightLineIndices = np.concatenate(rightLineIndices)

        # Extract left and right line pixel positions
        leftX = nonZeroX[leftLineIndices]
        leftY = nonZeroY[leftLineIndices]
        rightX = nonZeroX[rightLineIndices]
        rightY = nonZeroY[rightLineIndices]

        # Fit a second order polynomial to each
        leftFit = np.polyfit(leftY, leftX, 2)
        rightFit = np.polyfit(rightY, rightX, 2)

        return leftFit, rightFit
