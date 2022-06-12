import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QComboBox, QRadioButton, QSlider, QStatusBar, QButtonGroup
from PyQt5.QtWidgets import QCheckBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2 as cv

class MoCapGUI(QMainWindow):
    new_frame_sig = pyqtSignal(tuple)
    mask_params_changed_sig = pyqtSignal()
    masks_generated_sig = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        uic.loadUi('Mocap.ui', self)
        self.showMaximized()

        self.statusBar().showMessage("Frame = 0")
        self.cap1 = cv.VideoCapture()
        self.cap2 = cv.VideoCapture()
        self.masks_parameters = np.zeros((2, 3, 2, 3), dtype=int)

        # defining widgets
        self.stream1_label = self.findChild(QLabel, 'stream1_label')
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
        self.loadVideo2_Button = self.findChild(QPushButton, 'loadVideo2_Button')
        self.offlineStereo_ComboBox = self.findChild(QComboBox, 'offlineStereo_ComboBox')
        self.activeMaskVideo1_Radio = self.findChild(QRadioButton, 'activeMaskVideo1_Radio')
        assert isinstance(self.activeMaskVideo1_Radio, QRadioButton)
        self.activeMaskVideo2_Radio = self.findChild(QRadioButton, 'activeMaskVideo2_Radio')
        self.channelRed_Radio = self.findChild(QRadioButton, 'channelRed_Radio')
        self.channelGreen_Radio = self.findChild(QRadioButton, 'channelGreen_Radio')
        self.channelBlue_Radio = self.findChild(QRadioButton, 'channelBlue_Radio')
        self.hueMin_Slider = self.findChild(QSlider, 'hueMin_Slider')
        self.hueMax_Slider = self.findChild(QSlider, 'hueMax_Slider')
        self.saturationMin_Slider = self.findChild(QSlider, 'saturationMin_Slider')
        self.saturationMax_Slider = self.findChild(QSlider, 'saturationMax_Slider')
        self.brightnessMin_Slider = self.findChild(QSlider, 'brightnessMin_Slider')
        self.brightnessMax_Slider = self.findChild(QSlider, 'brightnessMax_Slider')
        self.video_buttonGroup = self.findChild(QButtonGroup, 'video_buttonGroup')
        assert isinstance(self.video_buttonGroup, QButtonGroup)
        self.video_buttonGroup.setId(self.activeMaskVideo1_Radio, 0)
        self.video_buttonGroup.setId(self.activeMaskVideo2_Radio, 1)
        self.rgb_buttonGroup = self.findChild(QButtonGroup, 'rgb_buttonGroup')
        assert isinstance(self.rgb_buttonGroup, QButtonGroup)
        self.rgb_buttonGroup.setId(self.channelRed_Radio, 0)
        self.rgb_buttonGroup.setId(self.channelBlue_Radio, 1)
        self.rgb_buttonGroup.setId(self.channelGreen_Radio, 2)
        self.video_horizontal_slider = self.findChild(QSlider, 'video_horizontalSlider')

        # Linking signals to widgets
        self.loadVideo1_Button.clicked.connect(self.load_video1)
        self.loadVideo2_Button.clicked.connect(self.load_video2)
        self.video_horizontal_slider.valueChanged.connect(self.video_seeker_slider_changed)
        self.new_frame_sig.connect(self.display_frame)
        self.new_frame_sig.connect(self.apply_mask)
        self.masks_generated_sig.connect(self.display_mask)
        self.mask_params_changed_sig.connect(self.get_current_frame)
        self.hueMin_Slider.valueChanged.connect(self.update_mask_parameters)
        self.hueMax_Slider.valueChanged.connect(self.update_mask_parameters)
        self.saturationMin_Slider.valueChanged.connect(self.update_mask_parameters)
        self.saturationMax_Slider.valueChanged.connect(self.update_mask_parameters)
        self.brightnessMin_Slider.valueChanged.connect(self.update_mask_parameters)
        self.brightnessMax_Slider.valueChanged.connect(self.update_mask_parameters)
        self.rgb_buttonGroup.buttonClicked.connect(self.update_mask_sliders)
        self.video_buttonGroup.buttonClicked.connect(self.update_mask_sliders)

    def load_video1(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Video load", ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        self.cap1 = cv.VideoCapture(fileName)
        self.cap1.set(cv.CAP_PROP_POS_FRAMES, 300)
        frame_count = self.cap1.get(cv.CAP_PROP_FRAME_COUNT)

        self.video_horizontal_slider.setRange(1, int(frame_count))

        ret, cv_img = self.cap1.read()
        if ret:
            self.new_frame_sig.emit((cv_img, 0))
            print("img1 emitted")

    def load_video2(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Video load", ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        self.cap2 = cv.VideoCapture(fileName)
        self.cap2.set(cv.CAP_PROP_POS_FRAMES, 300)
        frame_count = self.cap2.get(cv.CAP_PROP_FRAME_COUNT)

        self.video_horizontal_slider.setRange(1, int(frame_count))

        ret, cv_img = self.cap2.read()
        if ret:
            self.new_frame_sig.emit((0, cv_img))

    def video_seeker_slider_changed(self, frame_index):
        self.statusBar().showMessage("Frame " + str(frame_index))

        if self.cap1:
            self.cap1.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ret, cv_img = self.cap1.read()
            if ret:
                self.new_frame_sig.emit((cv_img, 0))

        if self.cap2:
            self.cap2.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ret, cv_img = self.cap2.read()
            if ret:
                self.new_frame_sig.emit((0, cv_img))

    def update_mask_parameters(self):
        stream = self.video_buttonGroup.checkedId()
        channel = self.rgb_buttonGroup.checkedId()

        low = (self.hueMin_Slider.sliderPosition(),
               self.saturationMin_Slider.sliderPosition(),
               self.brightnessMin_Slider.sliderPosition())

        high = (self.hueMax_Slider.sliderPosition(),
                self.saturationMax_Slider.sliderPosition(),
                self.brightnessMax_Slider.sliderPosition())

        self.masks_parameters[stream, channel] = (low, high)
        self.mask_params_changed_sig.emit()

    def apply_mask(self, cv_images):
        cv_img1, cv_img2 = cv_images

        if type(cv_img1) is not int:
            hsv = cv.cvtColor(cv_img1, cv.COLOR_BGR2HLS)

            low_r, high_r = self.masks_parameters[0, 0]
            low_g, high_g = self.masks_parameters[0, 1]
            low_b, high_b = self.masks_parameters[0, 2]

            mask_r = cv.inRange(hsv, low_r, high_r)
            mask_g = cv.inRange(hsv, low_g, high_g)
            mask_b = cv.inRange(hsv, low_b, high_b)

            self.masks_generated_sig.emit(((mask_r, mask_g, mask_b), 0))

        if type(cv_img2) is not int:
            hsv = cv.cvtColor(cv_img2, cv.COLOR_BGR2HLS)

            low_r, high_r = self.masks_parameters[1, 0]
            low_g, high_g = self.masks_parameters[1, 1]
            low_b, high_b = self.masks_parameters[1, 2]

            mask_r = cv.inRange(hsv, low_r, high_r)
            mask_g = cv.inRange(hsv, low_g, high_g)
            mask_b = cv.inRange(hsv, low_b, high_b)

            self.masks_generated_sig.emit((0, (mask_r, mask_g, mask_b)))

    def get_current_frame(self):
        """get the same frame the video stream is stopping on and sends a new frame signal"""
        stream = self.video_buttonGroup.checkedId()

        frame_number = self.cap1.get(cv.CAP_PROP_POS_FRAMES)-1
        self.cap1.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret1, cv_img1 = self.cap1.read()

        frame_number = self.cap2.get(cv.CAP_PROP_POS_FRAMES)-1
        self.cap2.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret2, cv_img2 = self.cap2.read()

        if ret1 and stream == 0:
            self.new_frame_sig.emit((cv_img1, 0))

        if ret2 and stream == 1:
            self.new_frame_sig.emit((0, cv_img2))

        else:
            print("No stream is available")

    @pyqtSlot(np.ndarray)
    def display_frame(self, images):
        """Updates the image_label with a new opencv image"""
        cv_img1, cv_img2 = images

        if type(cv_img1) is not int:
            qt_img = self.convert_cv_qt(cv_img1)
            p = qt_img.scaledToHeight(480, Qt.FastTransformation)
            self.stream1_label.setPixmap(p)

        if type(cv_img2) is not int:
            qt_img = self.convert_cv_qt(cv_img2)
            p = qt_img.scaledToHeight(480, Qt.FastTransformation)
            self.stream2_label.setPixmap(p)

    @pyqtSlot(np.ndarray)
    def display_mask(self, masks):
        """Updates the image_label with a new opencv image"""
        mask1, mask2 = masks

        current_id = self.rgb_buttonGroup.checkedId()

        if type(mask1) is not int:
            cv_img = mask1[current_id]
            qt_img = self.convert_cv_qt(cv_img)
            p = qt_img.scaledToHeight(480, Qt.FastTransformation)
            self.stream3_label.setPixmap(p)

        if type(mask2) is not int:
            cv_img = mask2[current_id]
            qt_img = self.convert_cv_qt(cv_img)
            p = qt_img.scaledToHeight(480, Qt.FastTransformation)
            self.stream4_label.setPixmap(p)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        p = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(p)

    def update_mask_sliders(self):
        stream = self.video_buttonGroup.checkedId()
        channel = self.rgb_buttonGroup.checkedId()

        LH, LS, LI = self.masks_parameters[stream, channel, 0]
        HH, HS, HI = self.masks_parameters[stream, channel, 1]

        self.hueMin_Slider.setSliderPosition(LH)
        self.saturationMin_Slider.setSliderPosition(LS)
        self.brightnessMin_Slider.setSliderPosition(LH)
        self.hueMax_Slider.setSliderPosition(HH)
        self.saturationMax_Slider.setSliderPosition(HS)
        self.brightnessMax_Slider.setSliderPosition(HI)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MoCapGUI()

    demo.show()
    sys.exit(app.exec_())

#Test
