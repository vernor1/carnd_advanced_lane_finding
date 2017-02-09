import cv2
import glob
import numpy as np
import os.path
import pickle


class TLensCorrector():
    # Constants ---------------------------------------------------------------
    CHESSBOARD_COLUMNS = 9
    CHESSBOARD_ROWS = 6

    # Public Members ----------------------------------------------------------
    def __init__(self, calibrationDir):
        self.CameraMatrix, self.DistCoeffs = self.GetCalibrationData(calibrationDir)

    def Undistort(self, img):
        return cv2.undistort(img, self.CameraMatrix, self.DistCoeffs, None, self.CameraMatrix)

    # Private Members ---------------------------------------------------------
    def GetCalibrationData(self, calibrationDir):
        calibrationDataFile = calibrationDir + "/calibration_data.p"
        calibrationData = {}
        if os.path.isfile(calibrationDataFile):
            print("Loading camera calibration data")
            calibrationData = pickle.load(open(calibrationDataFile, "rb"))
            cameraMatrix = calibrationData["cameraMatrix"]
            distCoeffs = calibrationData["distCoeffs"]
        else:
            print("Collecting camera calibration data")
            objPoints, imgPoints, imgSize = self.GetCalibrationPoints(calibrationDir)
            retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, imgSize, None, None)

            # Save the camera calibration result for fast access
            calibrationData["cameraMatrix"] = cameraMatrix
            calibrationData["distCoeffs"] = distCoeffs
            pickle.dump(calibrationData, open(calibrationDataFile, "wb"))

        return cameraMatrix, distCoeffs

    def GetCalibrationPoints(self, calibrationDir):
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (6,5,0)
        singleObjPoints = np.zeros((self.CHESSBOARD_COLUMNS * self.CHESSBOARD_ROWS, 3), np.float32)
        singleObjPoints[:, :2] = np.mgrid[0:self.CHESSBOARD_COLUMNS, 0:self.CHESSBOARD_ROWS].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images
        # 3D points in real world space
        objPoints = []

        # 2D points in image plane
        imgPoints = []

        # Make a list of calibration images
        images = glob.glob("%s/*.jpg" % (calibrationDir))

        imgSize = None
        # Step through the list and search for chessboard corners
        for idx, fileName in enumerate(images):
            img = cv2.imread(fileName)
            if not imgSize:
                imgSize = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            areFound, corners = cv2.findChessboardCorners(gray, (self.CHESSBOARD_COLUMNS, self.CHESSBOARD_ROWS), None)

            # If found, add object points, image points
            if areFound:
                print("Found corners of %s" % (fileName))
                objPoints.append(singleObjPoints)
                imgPoints.append(corners)

        return objPoints, imgPoints, imgSize
