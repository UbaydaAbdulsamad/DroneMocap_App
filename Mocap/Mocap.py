import datetime
from threading import Thread
from numpy import linalg
import cv2
import glob
import numpy as np
import pickle


def load_stereo(path, stream1, stream2):
    with open(path, 'rb') as f:
        stereo_object = pickle.load(f)
        assert isinstance(stereo_object, StereoCalibrate)
        assert isinstance(stream1, VideoStreaming)
        assert isinstance(stream2, VideoStreaming)

        c1_frame_size = stereo_object.getCameraObj1.getFrameSize
        c2_frame_size = stereo_object.getCameraObj2.getFrameSize

        # Calculating parameters of different resolutions
        stereo_object.getCameraObj1.getIntMatrix[0,
                                                 0] *= (stream1.get(3) / c1_frame_size[0])
        stereo_object.getCameraObj1.getIntMatrix[1,
                                                 1] *= (stream1.get(4) / c1_frame_size[1])
        stereo_object.getCameraObj1.getIntMatrix[0,
                                                 2] *= (stream1.get(3) / c1_frame_size[0])
        stereo_object.getCameraObj1.getIntMatrix[1,
                                                 2] *= (stream1.get(4) / c1_frame_size[1])

        stereo_object.getCameraObj2.getIntMatrix[0,
                                                 0] *= (stream2.get(3) / c2_frame_size[0])
        stereo_object.getCameraObj2.getIntMatrix[1,
                                                 1] *= (stream2.get(4) / c2_frame_size[1])
        stereo_object.getCameraObj2.getIntMatrix[0,
                                                 2] *= (stream2.get(3) / c2_frame_size[0])
        stereo_object.getCameraObj2.getIntMatrix[1,
                                                 2] *= (stream2.get(4) / c2_frame_size[1])
        #
        # # Calculating projection matrices
        # stereo_object.getProjMat1 = np.concatenate((stereo_object.getCameraObj1.getIntMatrix, np.zeros((3, 1))), axis=1)
        # stereo_object.getProjMat2 = np.dot(stereo_object.getCameraObj2.getIntMatrix,
        #                                    np.concatenate((stereo_object.getRotMat, stereo_object.getTransVec), axis=1))

        return stereo_object


def load_camera(path):
    with open(path, 'rb') as f:
        object = pickle.load(f)
        assert isinstance(object, SingleCameraCalibrate)
        return object


def show_circles(frames, points, radius, **kwargs):
    thickness = kwargs.get('thickness', 2)
    for marker_index, marker in enumerate(points):
        for frame_index, frame in enumerate(marker):
            cv2.circle(frames[frame_index], (int(frame[0]), int(frame[1])), int(
                radius[frame_index, marker_index]), (0, 255, 0), thickness)


def triangulate_point(x1, x2, p1, p2):
    M = np.zeros((6, 6))
    M[:3, :4] = p1
    M[3:, :4] = p2
    M[:3, 4] = [-x1[0], -x1[1], -1]
    M[3:, 5] = [-x2[0], -x2[1], -1]
    U, S, V = linalg.svd(M)
    X = V[-1, :4]
    return X[:3] / X[3]


def triangulate_points(points, stereo, kwargs):
    assert isinstance(stereo, StereoCalibrate)
    stream_num = kwargs.get("streamNum", 2)
    marker_num = kwargs.get("MarkerNum", 2)
    points3d = np.zeros([marker_num], tuple)

    for index, marker in enumerate(points):
        points3d[index] = triangulate_point(
            marker[0], marker[1], stereo.getProjMat1, stereo.getProjMat2)

    return points3d


def calculate_distance_two_points(points3d, **kwargs):
    precision = kwargs.get('precision', 2)
    unit = kwargs.get('unit', 10)
    squared_dist = np.sum((points3d[0] - points3d[1]) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    distance = round(dist / unit, precision)
    return distance


class VideoStreaming:
    def __init__(self, src, _width, _height, **kwargs):
        self._width = _width
        self._height = _height
        self._stream = cv2.VideoCapture(src)
        self._stream.set(3, self._width)
        self._stream.set(4, self._height)
        self._grabbed, self._frame = self._stream.read()
        self._stopped = False
        self._paused = False

        # Calculating frame rate
        self._framesToCount = kwargs.get('framesNum', 30)
        self._numFrames = 0
        self._fpsStart = None
        self._fps = 0

    def start(self):
        Thread(target=self.update, args=()).start()
        Thread(target=self.FPS, args=()).start()
        return self

    def update(self):
        while True:
            if self._stopped:
                self._stream.release()
                return
            elif self._paused:
                continue
            else:
                self._grabbed, self._frame = self._stream.read()
                self._fps = self.FPS()

    def read(self, **kwargs):
        size = kwargs.get('size', (self._width, self._height))

        if size == (self._width, self._height):
            return self._frame
        else:
            return cv2.resize(self._frame, size)

    def stop(self):
        self._stopped = True

    def get(self, x):
        if x == 'width' or x == 3:
            return int(self._stream.get(3))

        elif x == 'height' or x == 4:
            return int(self._stream.get(4))

        elif x == 'resolution':
            return int(self._stream.get(3)), int(self._stream.get(4))
        elif x == 'fps':
            return self._fps

    def set(self, _width, _height):
        self._stream.set(_width, _height)

    def isOpened(self):
        return self._stream.isOpened()

    def FPS(self):

        if self._numFrames == 0:
            self._fpsStart = datetime.datetime.now()
            self._numFrames += 1

        elif self._numFrames == self._framesToCount:
            self._numFrames = -1

            try:
                self._fps = self._framesToCount / \
                    (datetime.datetime.now() - self._fpsStart).total_seconds()
            except ZeroDivisionError:
                print("Video stream has been ended")
                self.stop()

        else:
            self._numFrames += 1

        return self._fps

    def pause(self):
        self._paused = not self._paused

    @property
    def get_paused(self):
        return self._paused

    @property
    def get_grabbed(self):
        return self._grabbed

    @property
    def get_frame(self):
        return self._frame

    @property
    def fps(self):
        return self._fps


class ShowVideo:
    def __init__(self, name, frame):
        self._stopped = False
        self._name = name
        self._frame = frame

    def startShow(self):
        Thread(target=self.show(), args=()).start()
        return self

    def show(self):
        while True:
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                return
            cv2.imshow(self._name, self._frame)

    def update(self, frame):
        self._frame = frame

    def stop(self):
        self._stopped = True


class SingleCameraCalibrate:
    """"
    Calibrate a Camera using chessboard images taken by it.

    Args:
    _name (str): name of the camera
    _path (str): path of the images to be imported
    _chessboard_size(tuple): number of inside corners of the chessboard
    _showPeriod (int): the time period an image of calibration will stay on screen
    _extension (str): extension of the images to be imported
    _showResolution: resolution of the displayed images
    Returns:
    Camera object with intrinsic parameters
    """

    def __init__(self, _name: str, _path: str, _chessboard_size: tuple, **kwargs):

        self._name = _name
        self._imagesPath = _path
        self._chessboardSize = _chessboard_size
        self._frameSize = None
        self._cameraMatrix = None
        self._newCameraMatrix = None
        self._distortionCoef = None
        self._meanError = 0

        self._showPeriod = kwargs.get('showPeriod', 1500)
        self._extension = kwargs.get('extension', 'png')
        self._showResolution = kwargs.get('showResolution', (1280, 720))

        self.Calibrate()

    def Calibrate(self):

        # termination criteria:
        criteria = (cv2.TermCriteria_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... (6,5,0)
        objp = np.zeros(
            (self._chessboardSize[0] * self._chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self._chessboardSize[0],
                               0:self._chessboardSize[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images
        objPoints = []  # 3d point in real world space
        imgPoints = []  # 2d points in image plane

        images = glob.glob(self._imagesPath + '\\*.' + self._extension)
        for image in images:
            print(image)
            cv2.im
            img = cv2.imread(image)
            self._frameSize = [img.shape[1], img.shape[0]]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray, self._chessboardSize, None)

            # If found, add object points, image points (after refining them), Uncomment code block to show the operation
            if ret:
                objPoints.append(objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgPoints.append(corners)

                # draw and display the the corners
                cv2.drawChessboardCorners(
                    img, self._chessboardSize, corners2, ret)
                imgLow = cv2.resize(img, self._showResolution)
                cv2.imshow('imgLow', imgLow)
                cv2.waitKey(self._showPeriod)
        cv2.destroyAllWindows()

        # Calibrating using data
        ret, self._cameraMatrix, self._distortionCoef, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints,
                                                                                          self._frameSize, None, None)
        self._newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self._cameraMatrix, self._distortionCoef,
                                                                   self._frameSize, 1, self._frameSize)

        # Calculating mean error
        for i in range(len(objPoints)):
            imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], self._cameraMatrix,
                                              self._distortionCoef)
            error = cv2.norm(imgPoints[i], imgPoints2,
                             cv2.NORM_L2) / len(imgPoints2)
            self._meanError += error

    def Save(self, _save_path):
        with open("{}\\{}.pickle".format(_save_path, self._name), 'wb') as f:
            pickle.dump(self, f)

    @property
    def getName(self):
        return self._name

    @property
    def getIntMatrix(self):
        return self._cameraMatrix

    @property
    def getDistortion(self):
        return self._distortionCoef

    @property
    def getFrameSize(self):
        return self._frameSize


class StereoCalibrate:
    '''
    Calibration of set of two calibrated cameras (getting the extrinsic parameters)

    Args:
        name(str): name of the calibrated set
        images_path(str): path of the images to be imported
        camera1(SingleCameraCalibrate): object of the first camera
        camera2(SingleCameraCalibrate): object of the second camera
        chessboard_size(tuple): number of all internal corners of the chessboard (x:int, y:int)
        chessboard_distance: dimension of the chessboard's squares in mm
        showPeriod(int): the time period one image is displayed on the screen
        _cxExtension: the extension of imported images
        _showResolution: resolution of the displayed images
        _c1FileName: name of the file saved using save function
        _c2FileName: name of the file saved using save function

    '''

    def __init__(self, name, images_path, camera1, camera2, chessboard_size, chessboard_distance, **kwargs):
        self._name = name
        self._imagesPath = images_path
        self._camera1 = camera1
        self._camera2 = camera2
        self._chessboardSize = chessboard_size
        self._chessboardDistance = chessboard_distance
        self._pairImagesNumber = 0
        self._frameSizeC1 = None
        self._frameSizeC2 = None
        self._projMatC1 = None
        self._projMatC2 = None
        self._rotationMat = None
        self._transVector = None

        self.showPeriod = kwargs.get('showPeriod', 1000)
        self._c1Extension = kwargs.get('Camera1Ext', 'png')
        self._c2Extension = kwargs.get('Camera2Ext', 'png')
        self._showResolution = kwargs.get('showResolution', (640, 360))
        self._c1FileName = kwargs.get('camera1FileName', self._camera1.getName)
        self._c2FileName = kwargs.get('camera2FileName', self._camera2.getName)

        assert isinstance(camera1, SingleCameraCalibrate)
        assert isinstance(camera2, SingleCameraCalibrate)

        self.Calibrate()

    def Calibrate(self):
        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0) , (1,0,0), (2,0,0), ... , (6,5,0)
        objp = np.zeros(
            (self._chessboardSize[0] * self._chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self._chessboardSize[0],
                               0:self._chessboardSize[1]].T.reshape(-1, 2)
        objp = objp * self._chessboardDistance

        # Arrays to store object points and image points from all the images
        objpoints = []  # 3d point in real space.
        imgpointsL = []  # 2d points in image plane.
        imgpointsR = []  # 2d points in image plane.

        imagesFirst = glob.glob(
            self._imagesPath + '\\{}*.{}'.format(self._c1FileName, self._c1Extension))
        imagesSecond = glob.glob(
            self._imagesPath + '\\{}*.{}'.format(self._c2FileName, self._c2Extension))

        for imgFirst, imgSecond in zip(imagesFirst, imagesSecond):
            imgL = cv2.imread(imgFirst)
            imgR = cv2.imread(imgSecond)
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            retL, cornersL = cv2.findChessboardCorners(
                grayL, self._chessboardSize, None)
            retR, cornersR = cv2.findChessboardCorners(
                grayR, self._chessboardSize, None)

            # If found, add object points, image points (after refining them)
            if retL and retR:
                self._pairImagesNumber += 1
                objpoints.append(objp)

                cornersL = cv2.cornerSubPix(
                    grayL, cornersL, (11, 11), (-1, -1), criteria)
                imgpointsL.append(cornersL)
                cornersR = cv2.cornerSubPix(
                    grayR, cornersR, (11, 11), (-1, -1), criteria)
                imgpointsR.append(cornersR)

                # Draw and display the corners
                cv2.drawChessboardCorners(
                    imgL, self._chessboardSize, cornersL, retL)
                cv2.drawChessboardCorners(
                    imgR, self._chessboardSize, cornersR, retR)
                imgLresized = cv2.resize(imgL, self._showResolution)
                imgRresized = cv2.resize(imgR, self._showResolution)
                cv2.imshow(self._camera1.getName, imgLresized)
                cv2.imshow(self._camera2.getName, imgRresized)
                print(imgFirst.split('\\')[-1], imgSecond.split('\\')[-1])
                cv2.waitKey(self.showPeriod)

            elif retL and not retR:
                print("couldn't detect corners on {}'s photo.".format(
                    self._camera1.getName))

            elif not retL and retR:
                print("couldn't detect corners on {}'s photo.".format(
                    self._camera2.getName))

            else:
                print("couldn't detect corners on both photos.".format(imgSecond))
        print("Number of pair of photos Used In Calibration: ",
              self._pairImagesNumber)
        cv2.destroyAllWindows()

        # Stereo Calibration
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                           cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        retStereo, newCameraMatrixWebcam, distFirst, newCameraMatrixJ6, distSecond, self._rotationMat, self._transVector, essentialMatrix, fundamentalMatrix = \
            cv2.stereoCalibrate(objpoints, imgpointsL, imgpointsR, self._camera1.getIntMatrix,
                                self._camera1.getDistortion, self._camera2.getIntMatrix, self._camera2.getDistortion,
                                grayL.shape[::-1], criteria_stereo, flags)

        # Calculating Projection Matrices
        self._projMatC1 = np.concatenate(
            (self._camera1.getIntMatrix, np.zeros((3, 1))), axis=1)
        self._projMatC2 = np.dot(self._camera1.getIntMatrix,
                                 np.concatenate((self._rotationMat, self._transVector), axis=1))

    def Save(self, _save_path):
        with open("{}\\{}.pickle".format(_save_path, self._name), 'wb') as f:
            pickle.dump(self, f)
        print('file has been saved to {}'.format(_save_path))

    @property
    def getProjMat1(self):
        return self._projMatC1

    @getProjMat1.setter
    def getProjMat1(self, newProjMat):
        self._projMatC1 = newProjMat

    @property
    def getProjMat2(self):
        return self._projMatC2

    @getProjMat2.setter
    def getProjMat2(self, newProjMat):
        self._projMatC2 = newProjMat

    @property
    def getCameraObj1(self):
        assert isinstance(self._camera1, SingleCameraCalibrate)
        return self._camera1

    @property
    def getCameraObj2(self):
        assert isinstance(self._camera2, SingleCameraCalibrate)
        return self._camera2

    @getCameraObj1.setter
    def getCameraObj1(self, newMat):
        self._camera1 = newMat

    @getCameraObj2.setter
    def getCameraObj2(self, newMat):
        self._camera2 = newMat

    @property
    def getRotMat(self):
        return self._rotationMat

    @property
    def getTransVec(self):
        return self._transVector


class Triangulate:
    def __init__(self, stereo, **kwargs):
        self._stereo = stereo
        self._stream_num = kwargs.get("streamNum", 2)
        self._marker_num = kwargs.get("MarkerNum", 2)
        self._points3d = np.zeros([self._marker_num], tuple)

        self._precision = kwargs.get('precision', 2)
        self._unit = kwargs.get('unit', 10)

        self._distance = None
        assert isinstance(self._stereo, StereoCalibrate)

    def triangulate_point(self, x1, x2, p1, p2):
        M = np.zeros((6, 6))
        M[:3, :4] = p1
        M[3:, :4] = p2
        M[:3, 4] = [-x1[0], -x1[1], -1]
        M[3:, 5] = [-x2[0], -x2[1], -1]
        U, S, V = linalg.svd(M)
        X = V[-1, :4]
        return X[:3] / X[3]

    def triangulate_points(self, points):
        for index, marker in enumerate(points):
            self._points3d[index] = self.triangulate_point(
                marker[0], marker[1], self._stereo.getProjMat1, self._stereo.getProjMat2)

    @property
    def points3d(self):
        return self._points3d

    def calculate_distance(self):
        squared_dist = np.sum(
            (self.points3d[0] - self.points3d[1]) ** 2, axis=0)
        dist = np.sqrt(squared_dist)
        self._distance = round(dist / self._unit, self._precision)
        return self._distance


class GetThresholdingData:
    def __init__(self, stream_num, marker_num, file_path):
        # inputs
        self._stream_num = stream_num
        self._marker_num = marker_num
        self._file_path = file_path

        # private data
        self._trackbars_data = np.zeros(
            [self._stream_num, self._marker_num, 8], int)
        self._frame_index = 0
        self._marker_index = 0

        # outputs
        self._boundaries_array = np.empty(
            [self._stream_num, self._marker_num, 6], int)

        # load last thresholding data file
        self.load_file()
        self.create_trackbars()

    def create_trackbars(self):
        cv2.namedWindow("Thresholding")

        cv2.createTrackbar('Stream', 'Thresholding',
                           self._trackbars_data[0, 0, 6], self._stream_num - 1, self.change_frame_index)
        cv2.createTrackbar('Marker', 'Thresholding',
                           self._trackbars_data[0, 0, 7], self._marker_num - 1, self.change_marker_index)
        cv2.createTrackbar('Min Hue', 'Thresholding',
                           self._trackbars_data[0, 0, 0], 255, self.nothing)
        cv2.createTrackbar('Max Hue', 'Thresholding',
                           self._trackbars_data[0, 0, 3], 255, self.nothing)
        cv2.createTrackbar('Min Sat', 'Thresholding',
                           self._trackbars_data[0, 0, 1], 255, self.nothing)
        cv2.createTrackbar('Max Sat', 'Thresholding',
                           self._trackbars_data[0, 0, 4], 255, self.nothing)
        cv2.createTrackbar('Min Brightness', 'Thresholding',
                           self._trackbars_data[0, 0, 2], 255, self.nothing)
        cv2.createTrackbar('Max Brightness', 'Thresholding',
                           self._trackbars_data[0, 0, 5], 255, self.nothing)

    def nothing(self, x):
        pass

    def change_frame_index(self, new_frame_index):
        self._frame_index = new_frame_index
        cv2.setTrackbarPos('Min Hue', 'Thresholding',
                           self._trackbars_data[new_frame_index, self._marker_index, 0])
        cv2.setTrackbarPos('Min Sat', 'Thresholding',
                           self._trackbars_data[new_frame_index, self._marker_index, 1])
        cv2.setTrackbarPos('Min Brightness', 'Thresholding',
                           self._trackbars_data[new_frame_index, self._marker_index, 2])
        cv2.setTrackbarPos('Max Hue', 'Thresholding',
                           self._trackbars_data[new_frame_index, self._marker_index, 3])
        cv2.setTrackbarPos('Max Sat', 'Thresholding',
                           self._trackbars_data[new_frame_index, self._marker_index, 4])
        cv2.setTrackbarPos('Max Brightness', 'Thresholding',
                           self._trackbars_data[new_frame_index, self._marker_index, 5])

    def change_marker_index(self, new_marker_index):
        cv2.setTrackbarPos('Min Hue', 'Thresholding',
                           self._trackbars_data[self._frame_index, new_marker_index, 0])
        cv2.setTrackbarPos('Min Sat', 'Thresholding',
                           self._trackbars_data[self._frame_index, new_marker_index, 1])
        cv2.setTrackbarPos('Min Brightness', 'Thresholding',
                           self._trackbars_data[self._frame_index, new_marker_index, 2])
        cv2.setTrackbarPos('Max Hue', 'Thresholding',
                           self._trackbars_data[self._frame_index, new_marker_index, 3])
        cv2.setTrackbarPos('Max Sat', 'Thresholding',
                           self._trackbars_data[self._frame_index, new_marker_index, 4])
        cv2.setTrackbarPos('Max Brightness', 'Thresholding',
                           self._trackbars_data[self._frame_index, new_marker_index, 5])

    def get_trackbars_position(self):
        self._frame_index = cv2.getTrackbarPos('Stream', 'Thresholding')
        self._marker_index = cv2.getTrackbarPos('Marker', 'Thresholding')

        self._trackbars_data[self._frame_index, self._marker_index,
                             0] = cv2.getTrackbarPos('Min Hue', 'Thresholding')
        self._trackbars_data[self._frame_index, self._marker_index,
                             1] = cv2.getTrackbarPos('Min Sat', 'Thresholding')
        self._trackbars_data[self._frame_index, self._marker_index,
                             2] = cv2.getTrackbarPos('Min Brightness', 'Thresholding')

        self._trackbars_data[self._frame_index, self._marker_index,
                             3] = cv2.getTrackbarPos('Max Hue', 'Thresholding')
        self._trackbars_data[self._frame_index, self._marker_index,
                             4] = cv2.getTrackbarPos('Max Sat', 'Thresholding')
        self._trackbars_data[self._frame_index, self._marker_index,
                             5] = cv2.getTrackbarPos('Max Brightness', 'Thresholding')

        self._trackbars_data[self._frame_index, self._marker_index,
                             6] = cv2.getTrackbarPos('Stream', 'Thresholding')
        self._trackbars_data[self._frame_index, self._marker_index,
                             7] = cv2.getTrackbarPos('Marker', 'Thresholding')

    def define_boundary(self):
        for stream_index in range(self._stream_num):
            for marker_index in range(self._marker_num):
                self._boundaries_array[stream_index,
                                       marker_index] = self._trackbars_data[stream_index, marker_index, :6]

    def load_obj(self):
        with open(self._file_path, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)

    def load_file(self):
        try:
            self.load_obj()

        except FileNotFoundError:
            print("couldn't load the thresholding properties' file, new file has been "
                  "generated in the path:\n", self._file_path)
            self.save_obj()
        finally:
            self.save_obj()
            print(
                "Couldn't load meta file, new file has been created at:\n", self._file_path)

    def save_obj(self):
        with open(self._file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def update(self):
        self.get_trackbars_position()
        self.define_boundary()
        self.save_obj()

    @property
    def boundaries_array(self):
        return self._boundaries_array

    @property
    def current_mask(self):
        return self._frame_index, self._marker_index


class LocateObjects:
    def __init__(self, streams_num, markers_num, pause_contour_calculations):
        # inputs
        self._frames_num = streams_num
        self._markers_num = markers_num
        self._frames = np.empty([streams_num], np.ndarray)
        self._boundary_arrays = np.empty(
            [streams_num, markers_num], np.ndarray)
        self._pause = pause_contour_calculations

        # private data
        self._resolution = self._frames.shape[:2]

        # outputs
        self._radius = np.zeros([self._frames_num, self._markers_num], float)
        self._points = np.zeros(
            [self._markers_num, self._frames_num, 2], float)
        self._masks = np.empty([self._frames_num, self._markers_num], object)

    def update_masks(self):
        for frame_index in range(self._frames_num):
            for marker_index in range(self._markers_num):
                hsv = cv2.cvtColor(
                    self._frames[frame_index], cv2.COLOR_BGR2HLS)
                self._masks[frame_index, marker_index] = \
                    cv2.inRange(hsv, self._boundary_arrays[frame_index, marker_index]
                                [:3], self._boundary_arrays[frame_index, marker_index][3:6])

    def locate(self):
        for mask_index in range(self._frames_num):
            for marker_index in range(self._markers_num):
                contours, _ = cv2.findContours(
                    self._masks[mask_index, marker_index], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                if len(contours) > 0:
                    (self._points[marker_index, mask_index, 0], self._points[marker_index, mask_index, 1]), self._radius[mask_index, marker_index] = \
                        cv2.minEnclosingCircle(
                            max(contours, key=cv2.contourArea))

    def update(self, frames, boundaries):
        self._frames = frames
        self._boundary_arrays = boundaries
        self.update_masks()
        if not self._pause:
            self.locate()

    @property
    def masks(self):
        return self._masks

    @property
    def points(self):
        return self._points

    @property
    def radius(self):
        return self._radius
