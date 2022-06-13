import pickle
import sys
import os
from Mocap.Mocap import SingleCameraCalibrate, StereoCalibrate
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QComboBox, QRadioButton, QSlider, QButtonGroup, QToolButton
from PyQt5.QtWidgets import QCheckBox, QFileDialog, QListWidget, QListWidgetItem, QTabWidget, QLineEdit, QSpinBox, QDoubleSpinBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
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
        self.browse_images_pushButton = self.findChild(QPushButton, 'browse_images_pushButton')
        self.images_path_comboBox = self.findChild(QComboBox, 'images_path_comboBox')
        self.loaded_images_list = self.findChild(QListWidget, 'loaded_images_list')
        self.main_tabWidget = self.findChild(QTabWidget, 'main_tabWidget')
        self.delete_image_pushButton = self.findChild(QPushButton, 'delete_image_pushButton')
        self.reload_pushButton = self.findChild(QPushButton, 'reload_pushButton')
        self.delete_all_images_pushButton = self.findChild(QPushButton, 'delete_all_images_pushButton')
        self.single_calibrate_pushButton = self.findChild(QPushButton, 'single_calibrate_pushButton')
        self.camera_id_lineEdit = self.findChild(QLineEdit, 'camera_id_lineEdit')
        self.chess_size_x_spinBox = self.findChild(QSpinBox, 'chess_size_x_spinBox')
        self.chess_size_y_spinBox = self.findChild(QSpinBox, 'chess_size_y_spinBox')
        self.img_disp_time_doubleSpinBox = self.findChild(QDoubleSpinBox, 'img_disp_time_doubleSpinBox')
        self.camera_intrinsic_list = self.findChild(QListWidget, 'camera_intrinsic_list')
        self.export_intrinsic_text_pushButton = self.findChild(QPushButton, 'export_intrinsic_text_pushButton')
        self.export_intrinsic_mocap_pushButton = self.findChild(QPushButton, 'export_intrinsic_mocap_pushButton')
        self.disp_time_pushButton = self.findChild(QPushButton, 'disp_time_pushButton')
        self.browse_intrinsic1_toolButton = self.findChild(QToolButton, 'browse_intrinsic1_toolButton')
        self.browse_intrinsic2_toolButton = self.findChild(QToolButton, 'browse_intrinsic2_toolButton')
        self.browse_stereo_img1_toolButton = self.findChild(QToolButton, 'browse_stereo_img1_toolButton')
        self.browse_stereo_img2_toolButton = self.findChild(QToolButton, 'browse_stereo_img2_toolButton')
        self.intrinsic1_comboBox = self.findChild(QComboBox, 'intrinsic1_comboBox')
        self.intrinsic2_comboBox = self.findChild(QComboBox, 'intrinsic2_comboBox')
        self.stereo_img1_comboBox = self.findChild(QComboBox, 'stereo_img1_comboBox')
        self.stereo_img2_comboBox = self.findChild(QComboBox, 'stereo_img2_comboBox')
        self.stereo_img1_list = self.findChild(QListWidget, 'stereo_img1_list')
        self.stereo_img2_list = self.findChild(QListWidget, 'stereo_img2_list')
        self.stereo_disp_time_pushButton = self.findChild(QPushButton, 'stereo_disp_time_pushButton')
        self.stereo_setup_ID_lineEdit = self.findChild(QLineEdit, 'stereo_setup_ID_lineEdit')
        self.stereo_chess_x_spinBox = self.findChild(QSpinBox, 'stereo_chess_x_spinBox')
        self.stereo_chess_y_spinBox = self.findChild(QSpinBox, 'stereo_chess_y_spinBox')
        self.stereo_chess_length_doubleSpinBox = self.findChild(QDoubleSpinBox, 'stereo_chess_length_doubleSpinBox')
        self.stereo_disp_time_doubleSpinBox = self.findChild(QDoubleSpinBox, 'stereo_disp_time_doubleSpinBox')
        self.stereo_calibrate_pushButton = self.findChild(QPushButton, 'stereo_calibrate_pushButton')
        self.stereo_calibrated_setups_list = self.findChild(QListWidget, 'stereo_calibrated_setups_list')


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
        self.browse_images_pushButton.clicked.connect(self.browse_single_images)
        assert isinstance(self.main_tabWidget, QTabWidget)
        self.main_tabWidget.currentChanged.connect(self.main_tab_changed)
        assert isinstance(self.loaded_images_list, QListWidget)
        self.loaded_images_list.itemClicked.connect(self.image_item_clicked)
        self.delete_image_pushButton.clicked.connect(self.remove_image_item)
        self.reload_pushButton.clicked.connect(self.reload_loaded_images_list)
        self.single_calibrate_pushButton.clicked.connect(self.single_calibrate)
        self.export_intrinsic_mocap_pushButton.clicked.connect(self.export_intrinsic_mocap)
        self.export_intrinsic_text_pushButton.clicked.connect(self.export_intrinsic_txt)
        self.browse_intrinsic2_toolButton.clicked.connect(self.load_intrinsic2)
        self.browse_intrinsic1_toolButton.clicked.connect(self.load_intrinsic1)
        self.browse_stereo_img1_toolButton.clicked.connect(self.browse_stereo_img1)
        self.browse_stereo_img2_toolButton.clicked.connect(self.browse_stereo_img2)
        assert isinstance(self.stereo_img1_comboBox, QComboBox)
        self.stereo_img1_comboBox.currentTextChanged.connect(self.populate_stereo_img_1_list)
        self.stereo_img2_comboBox.currentTextChanged.connect(self.populate_stereo_img_2_list)
        self.stereo_calibrate_pushButton.clicked.connect(self.stereo_calibrate)

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

        frame_number = self.cap1.get(cv.CAP_PROP_POS_FRAMES) - 1
        self.cap1.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret1, cv_img1 = self.cap1.read()

        frame_number = self.cap2.get(cv.CAP_PROP_POS_FRAMES) - 1
        self.cap2.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret2, cv_img2 = self.cap2.read()

        if ret1 and stream == 0:
            self.new_frame_sig.emit((cv_img1, 0))

        if ret2 and stream == 1:
            self.new_frame_sig.emit((0, cv_img2))

        else:
            print("No stream is available")

    def main_tab_changed(self, index):

        if index == 0:
            self.stream1_label.show()
            self.stream2_label.show()
            self.stream3_label.show()
            self.stream4_label.show()

        if index == 1:
            self.stream1_label.show()
            self.stream2_label.hide()
            self.stream3_label.hide()
            self.stream4_label.hide()

    def populate_single_images_list(self, folder_path):
        self.loaded_images_list.clear()
        images = os.listdir(folder_path)

        for image in images:
            path = os.path.join(folder_path, image)

            icon = QIcon()
            icon.addPixmap(QPixmap(path), QIcon.Normal, QIcon.Off)

            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.UserRole, cv.imread(path))
            self.loaded_images_list.addItem(item)

    def browse_stereo_img1(self):
        file = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        if not file:
            return
        assert isinstance(self.stereo_img1_comboBox, QComboBox)
        self.stereo_img1_comboBox.addItem(file)
        self.populate_stereo_img_1_list(file)

    def populate_stereo_img_1_list(self, folder_path):
        self.stereo_img1_list.clear()
        images = os.listdir(folder_path)

        for image in images:
            path = os.path.join(folder_path, image)

            icon = QIcon()
            icon.addPixmap(QPixmap(path), QIcon.Normal, QIcon.Off)

            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.UserRole, cv.imread(path))
            self.stereo_img1_list.addItem(item)

    def populate_stereo_img_2_list(self, folder_path):
        self.stereo_img2_list.clear()
        images = os.listdir(folder_path)

        for image in images:
            path = os.path.join(folder_path, image)

            icon = QIcon()
            icon.addPixmap(QPixmap(path), QIcon.Normal, QIcon.Off)

            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.UserRole, cv.imread(path))
            self.stereo_img2_list.addItem(item)

    def browse_stereo_img2(self):
        file = QFileDialog.getExistingDirectory(self, "Select Images Directory")
        if not file:
            return
        assert isinstance(self.stereo_img2_comboBox, QComboBox)
        self.stereo_img2_comboBox.addItem(file)
        self.populate_stereo_img_2_list(file)

    def stereo_calibrate(self):
        images1 = list()
        items = self.stereo_img1_list.findItems('', Qt.MatchRegExp)
        for item in items:
            img = item.data(Qt.UserRole)
            images1.append(img)

        images2 = list()
        items = self.stereo_img2_list.findItems('', Qt.MatchRegExp)
        for item in items:
            img = item.data(Qt.UserRole)
            images2.append(img)

        stereo_id = QLineEdit()
        stereo_id = self.stereo_setup_ID_lineEdit.text()
        cameraObj1 = self.intrinsic1_comboBox.currentData(Qt.UserRole)
        cameraObj2 = self.intrinsic2_comboBox.currentData(Qt.UserRole)
        x = self.stereo_chess_x_spinBox.value()
        y = self.stereo_chess_y_spinBox.value()
        chess_length = self.stereo_chess_length_doubleSpinBox.value()
        time = None
        if self.stereo_disp_time_pushButton.isChecked():
            time = int(self.stereo_disp_time_doubleSpinBox.value() * 1000)

        stereo = StereoCalibrate(stereo_id, images1, images2, cameraObj1, cameraObj2, (x, y), chess_length, showPeriod=time)
        stereo.clear_images()

        item = QListWidgetItem()
        item.setText(stereo.getID)
        item.setData(Qt.UserRole, stereo)
        self.stereo_calibrated_setups_list.addItem(item)

    def image_item_clicked(self):
        item = self.loaded_images_list.currentItem()
        image = item.data(Qt.UserRole)
        q_image = self.convert_cv_qt(image)
        q_image = q_image.scaledToWidth(1500)
        self.stream1_label.setPixmap(QPixmap(q_image))

    def remove_image_item(self):
        row = self.loaded_images_list.currentRow()
        self.loaded_images_list.takeItem(row)

    def reload_loaded_images_list(self):
        assert isinstance(self.images_path_comboBox, QComboBox)
        path = self.images_path_comboBox.currentText()
        self.populate_single_images_list(path)

    def browse_single_images(self):
        file = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not file:
            return
        assert isinstance(self.images_path_comboBox, QComboBox)
        self.images_path_comboBox.addItem(file)
        self.populate_single_images_list(file)

    def single_calibrate(self):
        camera_id = self.camera_id_lineEdit.text()
        x = self.chess_size_x_spinBox.value()
        y = self.chess_size_y_spinBox.value()
        time = None
        if self.disp_time_pushButton.isChecked():
            time = int(self.img_disp_time_doubleSpinBox.value() * 1000)

        images = list()
        items = self.loaded_images_list.findItems('', Qt.MatchRegExp)
        for item in items:
            img = item.data(Qt.UserRole)
            images.append(img)

        camera = SingleCameraCalibrate(camera_id, images, (x, y), showPeriod=time)
        self.loaded_images_list.clear()
        for img in camera.getImagesDrawn:
            icon = QIcon()
            qt_img = self.convert_cv_qt(img)
            icon.addPixmap(QPixmap(qt_img), QIcon.Normal, QIcon.Off)

            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.UserRole, img)
            self.loaded_images_list.addItem(item)

        camera.clear_images()
        item = QListWidgetItem()
        item.setText(camera.getID)
        item.setData(Qt.UserRole, camera)
        self.camera_intrinsic_list.addItem(item)

    def export_intrinsic_txt(self):
        item = self.camera_intrinsic_list.currentItem()
        assert isinstance(item, QListWidgetItem)
        cameraObj = item.data(Qt.UserRole)
        assert isinstance(cameraObj, SingleCameraCalibrate)
        data = "Camera ID = " + cameraObj.getID + "\n\n" \
               + "Images frame size = " + str(cameraObj.getFrameSize) + "\n\n" \
               + "Intrinsic Matrix:\n" + str(cameraObj.getIntMatrix) + "\n\n" \
               + "Distortion Coefficients:\n" + str(cameraObj.getDistortion)

        fileName, _ = QFileDialog.getSaveFileName(self, "Save Intrinsic Parameters", cameraObj.getID, "Text (*.txt)")
        if not fileName:
            return
        with open(fileName, 'w') as f:
            f.write(data)

    def export_intrinsic_mocap(self):
        item = self.camera_intrinsic_list.currentItem()
        cameraObj = item.data(Qt.UserRole)
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Intrinsic Parameters", cameraObj.getID, "Intsc (*.intsc)")
        if not fileName:
            return
        with open(fileName, 'wb') as f:
            pickle.dump(cameraObj, f)

    def load_intrinsic1(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Video Intrinsic Parameters", ".", "Intrinsic (*.intsc)")
        if not fileName:
            return
        with open(fileName, 'rb') as f:
            cameraObj = pickle.load(f)

        icon = QIcon()
        item = QListWidgetItem()
        item.setIcon(icon)
        item.setData(Qt.UserRole, cameraObj)
        self.intrinsic1_comboBox.addItem(cameraObj.getID, cameraObj)

    def load_intrinsic2(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Video Intrinsic Parameters", ".", "Intrinsic (*.intsc)")
        if not fileName:
            return
        with open(fileName, 'rb') as f:
            cameraObj = pickle.load(f)

        icon = QIcon()
        item = QListWidgetItem()
        item.setIcon(icon)
        item.setData(Qt.UserRole, cameraObj)
        self.intrinsic2_comboBox.addItem(cameraObj.getID, cameraObj)

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
