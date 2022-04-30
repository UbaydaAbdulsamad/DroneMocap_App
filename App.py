import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QMenuBar, QStatusBar, QApplication, QLabel, QPushButton
from PyQt5.QtWidgets import QCheckBox
from PyQt5 import uic
from DroneMocap_DataLink import CamLink

class MoCapGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('Mocap.ui', self)

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

        # initialize connection
        self.camera1 = CamLink('')
        self.camera1.connect_to_camera()

        # Linking signals to widgets
        self.camera1._signal.connect(self.update_GUI_labels)
        self.camera1.listen_to_camera()

    def update_GUI_labels(self):
        print(self.camera1._frame_info)
        # self.streamFrameRate_Label.setText(str(self.camera1._frame_rate_info))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MoCapGUI()

    demo.show()
    sys.exit(app.exec_())
