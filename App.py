import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QComboBox, QRadioButton, QSlider, QStatusBar
from PyQt5.QtWidgets import QCheckBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2 as cv
from DroneMocap_DataLink import CamLink

class MoCapGUI(QMainWindow):
    change_pixmap_signal_1 = pyqtSignal(np.ndarray)
    change_pixmap_signal_2 = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        uic.loadUi('Mocap.ui', self)
        self.showMaximized()

        self.statusBar().showMessage("Frame = 0")

        self.cap1 = cv.VideoCapture()
        self.cap2 = cv.VideoCapture()

        # defining widgets
        self.stream1_label = self.findChild(QLabel, 'stream1_label')
        isinstance(self.stream1_label, QLabel)

        self.stream2_label = self.findChild(QLabel, 'stream2_label')
        self.stream3_label = self.findChild(QLabel, 'stream3_label')
        self.stream4_label = self.findChild(QLabel, 'stream4_label')

        self.streamResolution_label = self.findChild(QLabel, 'StreamFrameRate_Label')
        self.streamFrameRate_Label = self.findChild(QLabel, 'streamFrameRate_Label')
        self.camera1_CheckBox = self.findChild(QCheckBox, 'camera1_CheckBox')
        self.camera2_CheckBox = self.findChild(QCheckBox, 'camera2_CheckBox')
        self.camera3_CheckBox = self.findChild(QCheckBox, 'camera3_CheckBox')
        self.camera4_CheckBox = self.findChild(QCheckBox, 'camera4_CheckBox')
        self.loadVideo1_Button = self.findChild(QPushButton, 'loadVideo1_Button')
        isinstance(self.loadVideo1_Button, QPushButton)
        self.loadVideo2_Button = self.findChild(QPushButton, 'loadVideo2_Button')
        isinstance(self.loadVideo2_Button, QPushButton)
        self.offlineStereo_ComboBox = self.findChild(QComboBox, 'offlineStereo_ComboBox')
        isinstance(self.offlineStereo_ComboBox, QComboBox)
        self.activeMaskVideo1_Radio = self.findChild(QRadioButton, 'activeMaskVideo1_Radio')
        isinstance(self.activeMaskVideo1_Radio, QRadioButton)
        self.activeMaskVideo2_Radio = self.findChild(QRadioButton, 'activeMaskVideo2_Radio')
        isinstance(self.activeMaskVideo2_Radio, QRadioButton)
        self.channelRed_Radio = self.findChild(QRadioButton, 'channelRed_Radio')
        isinstance(self.channelRed_Radio, QRadioButton)
        self.channelGreen_Radio = self.findChild(QRadioButton, 'channelGreen_Radio')
        isinstance(self.channelGreen_Radio, QRadioButton)
        self.channelBlue_Radio = self.findChild(QRadioButton, 'channelBlue_Radio')
        isinstance(self.channelBlue_Radio, QRadioButton)
        self.hueMin_Slider = self.findChild(QSlider, 'hueMin_Slider')
        isinstance(self.hueMin_Slider, QSlider)
        self.hueMax_Slider = self.findChild(QSlider, 'hueMax_Slider')
        isinstance(self.hueMax_Slider, QSlider)
        self.saturationMin_Slider = self.findChild(QSlider, 'saturationMin_Slider')
        isinstance(self.saturationMin_Slider, QSlider)
        self.saturationMax_Slider = self.findChild(QSlider, 'saturationMax_Slider')
        isinstance(self.saturationMax_Slider, QSlider)
        self.brightnessMin_Slider = self.findChild(QSlider, 'brightnessMin_Slider')
        isinstance(self.brightnessMin_Slider, QSlider)
        self.brightnessMax_Slider = self.findChild(QSlider, 'brightnessMax_Slider')
        isinstance(self.brightnessMax_Slider, QSlider)
        self.video_horizontalSlider = self.findChild(QSlider, 'video_horizontalSlider')
        isinstance(self.video_horizontalSlider, QSlider)

        # Linking signals to widgets
        self.change_pixmap_signal_1.connect(self.update_image1)
        self.change_pixmap_signal_2.connect(self.update_image2)
        self.video_horizontalSlider.valueChanged.connect(self.value_changed)
        self.loadVideo1_Button.clicked.connect(self.load_video1)
        self.loadVideo2_Button.clicked.connect(self.load_video2)

    def load_video1(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Video load", ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        self.cap1 = cv.VideoCapture(fileName)
        self.cap1.set(cv.CAP_PROP_POS_FRAMES, 300)
        frame_count = self.cap1.get(cv.CAP_PROP_FRAME_COUNT)

        self.video_horizontalSlider.setRange(1, int(frame_count))

        ret, cv_img = self.cap1.read()
        if ret:
            self.change_pixmap_signal_1.emit(cv_img)

    def load_video2(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Video load", ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        self.cap2 = cv.VideoCapture(fileName)
        self.cap2.set(cv.CAP_PROP_POS_FRAMES, 300)
        frame_count = self.cap2.get(cv.CAP_PROP_FRAME_COUNT)

        self.video_horizontalSlider.setRange(1, int(frame_count))

        ret, cv_img = self.cap2.read()
        if ret:
            self.change_pixmap_signal_2.emit(cv_img)

    def value_changed(self, frame_index):
        self.statusBar().showMessage("Frame " + str(frame_index))
        if self.cap1:
            self.cap1.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ret, cv_img = self.cap1.read()
            if ret:
                self.change_pixmap_signal_1.emit(cv_img)

        if self.cap2:
            self.cap2.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ret, cv_img = self.cap2.read()
            if ret:
                self.change_pixmap_signal_2.emit(cv_img)

    @pyqtSlot(np.ndarray)
    def update_image1(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        p = qt_img.scaledToHeight(480, Qt.FastTransformation)
        self.stream1_label.setPixmap(p)

    @pyqtSlot(np.ndarray)
    def update_image2(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        p = qt_img.scaledToHeight(480, Qt.FastTransformation)
        self.stream2_label.setPixmap(p)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        p = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(p)

    def update_GUI_labels(self):
        pass
        # image = QImage(self.camera1._frame_info, self.camera1._resolution_w_info, self.camera1._resolution_h_info, QImage.Format_Grayscale8)
        # pic = QPixmap(image)
        # self.stream1_label.setPixmap(pic)
        # self.streamFrameRate_Label.setText(str(self.camera1._frame_rate_info))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MoCapGUI()

    demo.show()
    sys.exit(app.exec_())

#Test
