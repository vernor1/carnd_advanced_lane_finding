import cv2
import numpy as np


class TLaneTracker():
    """ Lane tracker class.
    """
    class TLine():
        """ Line tracker class.
        """
        # Constants -----------------------------------------------------------
        NUMBER_OF_SLIDING_WINDOWS = 9
        SLIDING_WINDOWS_WIDTH = 200
        MIN_NUMBER_OF_PIXELS = 50
        # Width of lane is 12ft or 3.7m
        M_PER_PIX_HORIZONTAL = 3.7 / 615
        # Length of dashed lane line is 10ft or 3m
        M_PER_PIX_VERTICAL = 3 / 70
        MAX_HISTORY_LENGTH = 7
        MAX_COEFFICIENTS_DEVIATION = (6e-4, 4e-1, 1e+2)

        # Public Members ------------------------------------------------------
        def __init__(self, isRight=False):
            """ TLine ctor.

            param: isRight: Flag indicating whether the line is right or left
            """
            self.IsRight = isRight
            # History of polynomial coefficients in pixel space
            self.CoefficientsHistory = []
            # History of curvature measurements in meters
            self.CurvatureHistory = []

        def ProcessLineImage(self, binImg):
            """ Processes half of image containing a single lane line.

            param: binImg: Binary image of the line
            returns: Tuple of polynomial coefficients of the line in pixel space, radius of curvature in meters, base x-position of line
            """
            # Define y-position where we want radius of curvature and center offset
            baseY = binImg.shape[0]
            # Find the line points
            if len(self.CoefficientsHistory) == 0:
                # Extract new points
                self.UpdateHistoryWithNewPoints(self.ExtractNewPoints(binImg), baseY)
            else:
                # Update line points
                pointsX, pointsY = self.ExtractUpdatedPoints(binImg)
                coefficients = np.polyfit(pointsY, pointsX, 2)
                # Validate the coefficients
                averageCoefficients = np.average(self.CoefficientsHistory, axis=0)
                diffCoefficients = np.absolute(coefficients - averageCoefficients)
                if diffCoefficients[0] < self.MAX_COEFFICIENTS_DEVIATION[0] \
                        and diffCoefficients[1] < self.MAX_COEFFICIENTS_DEVIATION[1] \
                        and diffCoefficients[2] < self.MAX_COEFFICIENTS_DEVIATION[2]:
                    self.UpdateHistoryWithNewPoints((pointsX, pointsY), baseY)
                else:
                    if self.IsRight:
                        side = "right"
                    else:
                        side = "left"
                    print("Discarding new %s line coefficients because they are off limits: %s" % (side, diffCoefficients))
                    # Truncate the history and count down
                    self.CoefficientsHistory.pop(0)
                    if len(self.CoefficientsHistory) == 0:
                        print("Extracting new line points")
                        # Fall back to extracting new points
                        self.UpdateHistoryWithNewPoints(self.ExtractNewPoints(binImg), baseY)
            # Truncate the history if the max length reached
            if len(self.CoefficientsHistory) > self.MAX_HISTORY_LENGTH:
                self.CoefficientsHistory.pop(0)
            # Calculate average coefficients over the recent history
            averageCoefficients = np.average(self.CoefficientsHistory, axis=0)
            if len(self.CurvatureHistory) > self.MAX_HISTORY_LENGTH:
                self.CurvatureHistory.pop(0)
            # Calculate base x-position of the line
            baseX = averageCoefficients[0] * baseY**2 + averageCoefficients[1] * baseY + averageCoefficients[2]
            return averageCoefficients, np.average(self.CurvatureHistory), baseX

        # Private Members -----------------------------------------------------
        def ExtractNewPoints(self, binImg):
            """ Base method of extracting line points.

            param: binImg: Binary image of the line
            returns: x,y-points of the line
            """
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binImg[binImg.shape[0]//2:, :], axis=0)
            # Set height of windows
            windowHeight = binImg.shape[0] // self.NUMBER_OF_SLIDING_WINDOWS
            # Identify the x and y positions of all nonzero pixels in the image
            nonZero = binImg.nonzero()
            nonZeroY = np.array(nonZero[0])
            nonZeroX = np.array(nonZero[1])
            # Find the peak of the histogram - it will be the starting point for the line, and will update for each window
            currentX = np.argmax(histogram)
            # Create empty list to receive line pixel indices
            lineIndices = []
            # Step through the windows one by one
            for windowNbr in range(self.NUMBER_OF_SLIDING_WINDOWS):
                # Identify window boundaries in x and y (and right and left)
                windowLowY = binImg.shape[0] - (windowNbr + 1) * windowHeight
                windowHighY = binImg.shape[0] - windowNbr * windowHeight
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
            return nonZeroX[lineIndices] + self.IsRight * binImg.shape[1], nonZeroY[lineIndices]

        def ExtractUpdatedPoints(self, binImg):
            """ Method of extracting line points relying on existing polynomial coefficients.

            param: binImg: Binary image of the line
            returns: x,y-points of the line
            """
            # Identify the x and y positions of all nonzero pixels in the image
            nonZero = binImg.nonzero()
            nonZeroY = np.array(nonZero[0])
            nonZeroX = np.array(nonZero[1]) + self.IsRight * binImg.shape[1]
            # Calculate average coefficients over the recent history
            averageCoefficients = np.average(self.CoefficientsHistory, axis=0)
            # Identify the nonzero pixels in x and y
            lineIndices = ((nonZeroX > (averageCoefficients[0]*(nonZeroY**2)
                                        + averageCoefficients[1]*nonZeroY
                                        + averageCoefficients[2] - self.SLIDING_WINDOWS_WIDTH // 2))
                         & (nonZeroX < (averageCoefficients[0]*(nonZeroY**2)
                                        + averageCoefficients[1]*nonZeroY
                                        + averageCoefficients[2] + self.SLIDING_WINDOWS_WIDTH // 2)))
            pointsX = nonZeroX[lineIndices]
            pointsY = nonZeroY[lineIndices]
            return nonZeroX[lineIndices], nonZeroY[lineIndices]

        def UpdateHistoryWithNewPoints(self, points, baseY):
            """ Helper method for updating history of polynomial coefficients and curvature.

            param: points: x,y points of the line
            param: baseY: y-position where we want radius of curvature
            """
            # Fit a second order polynomial to x,y in pixel and world space
            coefficientsP = np.polyfit(points[1], points[0], 2)
            coefficientsM = np.polyfit(points[1] * self.M_PER_PIX_VERTICAL, points[0] * self.M_PER_PIX_HORIZONTAL, 2)
            # Store coefficients in the history
            self.CoefficientsHistory.append(coefficientsP)
            # Calculate radius of curvature
            curveRad = ((1 + (2 * coefficientsM[0] * baseY * self.M_PER_PIX_VERTICAL + coefficientsM[1])**2)**1.5) \
                       / np.absolute(2 * coefficientsM[0])
            self.CurvatureHistory.append(curveRad)

    # Public Members ----------------------------------------------------------
    def __init__(self):
        """ TLane ctor.
        """
        self.LeftLine = self.TLine()
        self.RightLine = self.TLine(True)

    def ProcessLaneImage(self, binImg):
        """ Processes an image containing lane lines.

        param: binImg: Binary image of the lane
        returns: Tuple of polynomial coefficients of two lines in pixel space, radius of curvature in meters, course deviation in meters
        """
        leftLineCoefficients, leftCurveRad, leftX = self.LeftLine.ProcessLineImage(binImg[:, :binImg.shape[1]//2])
        rightLineCoefficients, rightCurveRad, rightX = self.RightLine.ProcessLineImage(binImg[:, binImg.shape[1]//2:])
        deviation = (binImg.shape[1] // 2 - (leftX + rightX) // 2) * self.TLine.M_PER_PIX_HORIZONTAL
        return leftLineCoefficients, rightLineCoefficients, (leftCurveRad + rightCurveRad) / 2, deviation
