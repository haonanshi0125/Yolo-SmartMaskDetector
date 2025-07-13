# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'yoloside6.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QProgressBar, QPushButton, QSizePolicy,
    QSlider, QTextBrowser, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1328, 918)
        MainWindow.setStyleSheet(u"\n"
"border-radius:3px;\n"
"qproperty-alignment: 'AlignCenter';\n"
"font-size: 14pt;  \n"
"font-weight: 500; ")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalFrame = QFrame(self.centralwidget)
        self.verticalFrame.setObjectName(u"verticalFrame")
        self.verticalFrame.setStyleSheet(u"border:1px solid #6babfa;")
        self.verticalLayout = QVBoxLayout(self.verticalFrame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.page_up = QHBoxLayout()
        self.page_up.setObjectName(u"page_up")
        self.title = QLabel(self.verticalFrame)
        self.title.setObjectName(u"title")
        self.title.setStyleSheet(u"font-size: 20pt;  \n"
"border:1px solid #6ebee8;\n"
"background:#64bee4;\n"
"border-top-left-radius: 0px;\n"
"border-top-right-radius: 0px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.page_up.addWidget(self.title)


        self.verticalLayout.addLayout(self.page_up)

        self.page_down = QHBoxLayout()
        self.page_down.setSpacing(2)
        self.page_down.setObjectName(u"page_down")
        self.left = QHBoxLayout()
        self.left.setSpacing(2)
        self.left.setObjectName(u"left")
        self.left_page = QVBoxLayout()
        self.left_page.setSpacing(8)
        self.left_page.setObjectName(u"left_page")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalFrame_4 = QFrame(self.verticalFrame)
        self.verticalFrame_4.setObjectName(u"verticalFrame_4")
        self.verticalLayout_5 = QVBoxLayout(self.verticalFrame_4)
        self.verticalLayout_5.setSpacing(4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(4, 4, 4, 4)
        self.label_2 = QLabel(self.verticalFrame_4)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setStyleSheet(u"border:2px solid #CFEBD2;\n"
"background:#f68c1f;\n"
"border-top-left-radius: 4px;\n"
"border-top-right-radius: 4px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.verticalLayout_5.addWidget(self.label_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalFrame_2 = QFrame(self.verticalFrame_4)
        self.horizontalFrame_2.setObjectName(u"horizontalFrame_2")
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalFrame_2)
        self.horizontalLayout_4.setSpacing(4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setSpacing(6)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(-1, 0, 0, 0)
        self.model_name = QLineEdit(self.horizontalFrame_2)
        self.model_name.setObjectName(u"model_name")
        self.model_name.setMinimumSize(QSize(0, 4))
        self.model_name.setStyleSheet(u"padding-left:10px;\n"
"padding-top:10px;\n"
"border:1px solid skyblue;\n"
"border-radius:1px;")

        self.horizontalLayout_7.addWidget(self.model_name)

        self.model_select = QPushButton(self.horizontalFrame_2)
        self.model_select.setObjectName(u"model_select")
        icon = QIcon()
        icon.addFile(u"icons/modelmanager.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.model_select.setIcon(icon)
        self.model_select.setIconSize(QSize(32, 32))

        self.horizontalLayout_7.addWidget(self.model_select)


        self.horizontalLayout_4.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")

        self.horizontalLayout_4.addLayout(self.horizontalLayout_6)


        self.horizontalLayout_3.addWidget(self.horizontalFrame_2)


        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.verticalLayout_5.setStretch(0, 3)
        self.verticalLayout_5.setStretch(1, 2)

        self.verticalLayout_11.addWidget(self.verticalFrame_4)


        self.left_page.addLayout(self.verticalLayout_11)

        self.select = QVBoxLayout()
        self.select.setObjectName(u"select")
        self.verticalFrame_model = QFrame(self.verticalFrame)
        self.verticalFrame_model.setObjectName(u"verticalFrame_model")
        self.verticalLayout_8 = QVBoxLayout(self.verticalFrame_model)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(4, 4, 4, 4)
        self.label_5 = QLabel(self.verticalFrame_model)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setStyleSheet(u"border:2px solid #CFEBD2;\n"
"background:#f68c1f;\n"
"border-top-left-radius: 4px;\n"
"border-top-right-radius: 4px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.verticalLayout_8.addWidget(self.label_5)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalFrame_5 = QFrame(self.verticalFrame_model)
        self.horizontalFrame_5.setObjectName(u"horizontalFrame_5")
        self.horizontalLayout_9 = QHBoxLayout(self.horizontalFrame_5)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.image = QPushButton(self.horizontalFrame_5)
        self.image.setObjectName(u"image")
        icon1 = QIcon()
        icon1.addFile(u"icons/image.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.image.setIcon(icon1)
        self.image.setIconSize(QSize(32, 32))

        self.horizontalLayout_9.addWidget(self.image)

        self.video = QPushButton(self.horizontalFrame_5)
        self.video.setObjectName(u"video")
        icon2 = QIcon()
        icon2.addFile(u"icons/video.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video.setIcon(icon2)
        self.video.setIconSize(QSize(32, 32))

        self.horizontalLayout_9.addWidget(self.video)

        self.camera = QPushButton(self.horizontalFrame_5)
        self.camera.setObjectName(u"camera")
        icon3 = QIcon()
        icon3.addFile(u"icons/camera.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.camera.setIcon(icon3)
        self.camera.setIconSize(QSize(32, 32))

        self.horizontalLayout_9.addWidget(self.camera)

        self.dirs = QPushButton(self.horizontalFrame_5)
        self.dirs.setObjectName(u"dirs")
        icon4 = QIcon()
        icon4.addFile(u"icons/dir.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.dirs.setIcon(icon4)
        self.dirs.setIconSize(QSize(32, 32))

        self.horizontalLayout_9.addWidget(self.dirs)


        self.horizontalLayout_8.addWidget(self.horizontalFrame_5)


        self.verticalLayout_8.addLayout(self.horizontalLayout_8)

        self.verticalLayout_8.setStretch(0, 3)
        self.verticalLayout_8.setStretch(1, 2)

        self.select.addWidget(self.verticalFrame_model)


        self.left_page.addLayout(self.select)

        self.conf = QVBoxLayout()
        self.conf.setObjectName(u"conf")
        self.horizontalFrame_6 = QFrame(self.verticalFrame)
        self.horizontalFrame_6.setObjectName(u"horizontalFrame_6")
        self.horizontalLayout_11 = QHBoxLayout(self.horizontalFrame_6)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(4, 4, 4, 4)
        self.label_6 = QLabel(self.horizontalFrame_6)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setStyleSheet(u"border:2px solid #CFEBD2;\n"
"background:#f68c1f;\n"
"border-top-left-radius: 4px;\n"
"border-top-right-radius: 4px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.verticalLayout_10.addWidget(self.label_6)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.conf_num = QDoubleSpinBox(self.horizontalFrame_6)
        self.conf_num.setObjectName(u"conf_num")

        self.horizontalLayout_15.addWidget(self.conf_num)

        self.conf_slider = QSlider(self.horizontalFrame_6)
        self.conf_slider.setObjectName(u"conf_slider")
        self.conf_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_15.addWidget(self.conf_slider)


        self.verticalLayout_10.addLayout(self.horizontalLayout_15)

        self.verticalLayout_10.setStretch(0, 3)
        self.verticalLayout_10.setStretch(1, 2)

        self.horizontalLayout_11.addLayout(self.verticalLayout_10)


        self.conf.addWidget(self.horizontalFrame_6)

        self.conf.setStretch(0, 2)

        self.left_page.addLayout(self.conf)

        self.IOU = QVBoxLayout()
        self.IOU.setObjectName(u"IOU")
        self.horizontalFrame_51 = QFrame(self.verticalFrame)
        self.horizontalFrame_51.setObjectName(u"horizontalFrame_51")
        self.horizontalLayout_10 = QHBoxLayout(self.horizontalFrame_51)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_9.setContentsMargins(4, 4, 4, 4)
        self.label_7 = QLabel(self.horizontalFrame_51)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setStyleSheet(u"border:2px solid #CFEBD2;\n"
"background:#f68c1f;\n"
"border-top-left-radius: 4px;\n"
"border-top-right-radius: 4px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.verticalLayout_9.addWidget(self.label_7)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.iou_number = QDoubleSpinBox(self.horizontalFrame_51)
        self.iou_number.setObjectName(u"iou_number")

        self.horizontalLayout_12.addWidget(self.iou_number)

        self.iou_slider = QSlider(self.horizontalFrame_51)
        self.iou_slider.setObjectName(u"iou_slider")
        self.iou_slider.setOrientation(Qt.Orientation.Horizontal)

        self.horizontalLayout_12.addWidget(self.iou_slider)


        self.verticalLayout_9.addLayout(self.horizontalLayout_12)

        self.verticalLayout_9.setStretch(0, 3)
        self.verticalLayout_9.setStretch(1, 2)

        self.horizontalLayout_10.addLayout(self.verticalLayout_9)


        self.IOU.addWidget(self.horizontalFrame_51)

        self.IOU.setStretch(0, 2)

        self.left_page.addLayout(self.IOU)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalFrame_7 = QFrame(self.verticalFrame)
        self.horizontalFrame_7.setObjectName(u"horizontalFrame_7")
        self.horizontalLayout_16 = QHBoxLayout(self.horizontalFrame_7)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setContentsMargins(4, 4, 4, 4)
        self.label_9 = QLabel(self.horizontalFrame_7)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setStyleSheet(u"border:none\n"
"")

        self.horizontalLayout_17.addWidget(self.label_9)

        self.save_data = QCheckBox(self.horizontalFrame_7)
        self.save_data.setObjectName(u"save_data")
        self.save_data.setStyleSheet(u"border:none")

        self.horizontalLayout_17.addWidget(self.save_data)

        self.horizontalLayout_17.setStretch(0, 3)
        self.horizontalLayout_17.setStretch(1, 1)

        self.horizontalLayout_16.addLayout(self.horizontalLayout_17)


        self.verticalLayout_6.addWidget(self.horizontalFrame_7)


        self.left_page.addLayout(self.verticalLayout_6)

        self.verticalLayout_voice = QVBoxLayout()
        self.verticalLayout_voice.setObjectName(u"verticalLayout_voice")
        self.horizontalFrame_voice = QFrame(self.verticalFrame)
        self.horizontalFrame_voice.setObjectName(u"horizontalFrame_voice")
        self.horizontalLayout_voice = QHBoxLayout(self.horizontalFrame_voice)
        self.horizontalLayout_voice.setObjectName(u"horizontalLayout_voice")
        self.horizontalLayout_voice_setting = QHBoxLayout()
        self.horizontalLayout_voice_setting.setObjectName(u"horizontalLayout_voice_setting")
        self.horizontalLayout_voice_setting.setContentsMargins(4, 4, 4, 4)
        self.label_voice = QLabel(self.horizontalFrame_voice)
        self.label_voice.setObjectName(u"label_voice")
        self.label_voice.setStyleSheet(u"border:none")

        self.horizontalLayout_voice_setting.addWidget(self.label_voice)

        self.tts_enable_checkbox = QCheckBox(self.horizontalFrame_voice)
        self.tts_enable_checkbox.setObjectName(u"tts_enable_checkbox")
        self.tts_enable_checkbox.setStyleSheet(u"border:none")
        self.tts_enable_checkbox.setChecked(False)

        self.horizontalLayout_voice_setting.addWidget(self.tts_enable_checkbox)

        self.horizontalLayout_voice_setting.setStretch(0, 3)
        self.horizontalLayout_voice_setting.setStretch(1, 1)

        self.horizontalLayout_voice.addLayout(self.horizontalLayout_voice_setting)


        self.verticalLayout_voice.addWidget(self.horizontalFrame_voice)


        self.left_page.addLayout(self.verticalLayout_voice)

        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.logo = QLabel(self.verticalFrame)
        self.logo.setObjectName(u"logo")
        self.logo.setMaximumSize(QSize(500, 300))
        self.logo.setPixmap(QPixmap(u"icons/icon-model.png"))
        self.logo.setScaledContents(True)

        self.verticalLayout_7.addWidget(self.logo)


        self.left_page.addLayout(self.verticalLayout_7)

        self.left_page.setStretch(0, 3)
        self.left_page.setStretch(1, 3)
        self.left_page.setStretch(2, 3)
        self.left_page.setStretch(3, 3)
        self.left_page.setStretch(4, 1)
        self.left_page.setStretch(5, 1)
        self.left_page.setStretch(6, 4)

        self.left.addLayout(self.left_page)


        self.page_down.addLayout(self.left)

        self.mid = QHBoxLayout()
        self.mid.setObjectName(u"mid")
        self.verticalFrame_mid = QFrame(self.verticalFrame)
        self.verticalFrame_mid.setObjectName(u"verticalFrame_mid")
        self.verticalLayout_19 = QVBoxLayout(self.verticalFrame_mid)
        self.verticalLayout_19.setObjectName(u"verticalLayout_19")
        self.verticalLayout_23 = QVBoxLayout()
        self.verticalLayout_23.setObjectName(u"verticalLayout_23")
        self.finall_result = QLabel(self.verticalFrame_mid)
        self.finall_result.setObjectName(u"finall_result")
        self.finall_result.setMinimumSize(QSize(700, 700))
        self.finall_result.setMaximumSize(QSize(1200, 700))
        self.finall_result.setScaledContents(True)

        self.verticalLayout_23.addWidget(self.finall_result)


        self.verticalLayout_19.addLayout(self.verticalLayout_23)

        self.verticalLayout_22 = QVBoxLayout()
        self.verticalLayout_22.setSpacing(0)
        self.verticalLayout_22.setObjectName(u"verticalLayout_22")
        self.verticalFrame_16 = QFrame(self.verticalFrame_mid)
        self.verticalFrame_16.setObjectName(u"verticalFrame_16")
        self.verticalFrame_16.setStyleSheet(u"border:1px solid #6aaafa;\n"
"background:#6babfa;\n"
"border-top-left-radius: 0px;\n"
"border-top-right-radius: 0px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")
        self.verticalLayout_24 = QVBoxLayout(self.verticalFrame_16)
        self.verticalLayout_24.setSpacing(2)
        self.verticalLayout_24.setObjectName(u"verticalLayout_24")
        self.verticalLayout_24.setContentsMargins(4, 4, 4, 4)
        self.yolo_start = QPushButton(self.verticalFrame_16)
        self.yolo_start.setObjectName(u"yolo_start")
        font = QFont()
        font.setPointSize(14)
        font.setWeight(QFont.Medium)
        self.yolo_start.setFont(font)
        self.yolo_start.setStyleSheet(u"border:1px solid #6aaafa;\n"
"background:#6babfa;\n"
"border-top-left-radius: 0px;\n"
"border-top-right-radius: 0px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.verticalLayout_24.addWidget(self.yolo_start)


        self.verticalLayout_22.addWidget(self.verticalFrame_16)


        self.verticalLayout_19.addLayout(self.verticalLayout_22)

        self.verticalLayout_21 = QVBoxLayout()
        self.verticalLayout_21.setObjectName(u"verticalLayout_21")
        self.horizontalFrame_12 = QFrame(self.verticalFrame_mid)
        self.horizontalFrame_12.setObjectName(u"horizontalFrame_12")
        self.horizontalLayout_21 = QHBoxLayout(self.horizontalFrame_12)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.video_start = QPushButton(self.horizontalFrame_12)
        self.video_start.setObjectName(u"video_start")
        icon5 = QIcon()
        icon5.addFile(u"icons/start.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_start.setIcon(icon5)
        self.video_start.setIconSize(QSize(32, 32))

        self.horizontalLayout_21.addWidget(self.video_start)

        self.video_stop = QPushButton(self.horizontalFrame_12)
        self.video_stop.setObjectName(u"video_stop")
        icon6 = QIcon()
        icon6.addFile(u"icons/stop.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_stop.setIcon(icon6)
        self.video_stop.setIconSize(QSize(32, 32))

        self.horizontalLayout_21.addWidget(self.video_stop)

        self.video_progressBar = QProgressBar(self.horizontalFrame_12)
        self.video_progressBar.setObjectName(u"video_progressBar")
        self.video_progressBar.setStyleSheet(u"\n"
"QProgressBar{ \n"
"height:5px;\n"
"color: #8EC5FC; \n"
"text-align:center; \n"
"border:3px solid rgb(255, 255, 255);\n"
"border-radius: 5px; \n"
"background-color: rgba(215, 215, 215,100);\n"
"} \n"
"\n"
"QProgressBar:chunk{ \n"
"border-radius:0px; \n"
"background:  #6BABFA;\n"
"border-radius: 7px;\n"
"}")
        self.video_progressBar.setValue(24)

        self.horizontalLayout_21.addWidget(self.video_progressBar)

        self.video_termination = QPushButton(self.horizontalFrame_12)
        self.video_termination.setObjectName(u"video_termination")
        icon7 = QIcon()
        icon7.addFile(u"icons/end.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.video_termination.setIcon(icon7)
        self.video_termination.setIconSize(QSize(32, 32))

        self.horizontalLayout_21.addWidget(self.video_termination)


        self.verticalLayout_21.addWidget(self.horizontalFrame_12)


        self.verticalLayout_19.addLayout(self.verticalLayout_21)

        self.verticalLayout_19.setStretch(0, 10)
        self.verticalLayout_19.setStretch(1, 1)
        self.verticalLayout_19.setStretch(2, 1)

        self.mid.addWidget(self.verticalFrame_mid)


        self.page_down.addLayout(self.mid)

        self.right = QHBoxLayout()
        self.right.setSpacing(2)
        self.right.setObjectName(u"right")
        self.verticalFrame_10 = QFrame(self.verticalFrame)
        self.verticalFrame_10.setObjectName(u"verticalFrame_10")
        self.verticalLayout_14 = QVBoxLayout(self.verticalFrame_10)
        self.verticalLayout_14.setObjectName(u"verticalLayout_14")
        self.verticalLayout_15 = QVBoxLayout()
        self.verticalLayout_15.setObjectName(u"verticalLayout_15")
        self.verticalLayout_20 = QVBoxLayout()
        self.verticalLayout_20.setObjectName(u"verticalLayout_20")
        self.upload_image = QLabel(self.verticalFrame_10)
        self.upload_image.setObjectName(u"upload_image")
        self.upload_image.setMinimumSize(QSize(300, 300))
        self.upload_image.setMaximumSize(QSize(450, 300))

        self.verticalLayout_20.addWidget(self.upload_image)


        self.verticalLayout_15.addLayout(self.verticalLayout_20)

        self.horizontalLayout_20 = QHBoxLayout()
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.verticalFrame_12 = QFrame(self.verticalFrame_10)
        self.verticalFrame_12.setObjectName(u"verticalFrame_12")
        self.verticalLayout_18 = QVBoxLayout(self.verticalFrame_12)
        self.verticalLayout_18.setObjectName(u"verticalLayout_18")
        self.horizontalLayout_19 = QHBoxLayout()
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.label_3 = QLabel(self.verticalFrame_12)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setStyleSheet(u"background:rgb(255, 170, 127)")

        self.horizontalLayout_19.addWidget(self.label_3)


        self.verticalLayout_18.addLayout(self.horizontalLayout_19)

        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setSpacing(0)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.label_8 = QLabel(self.verticalFrame_12)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setStyleSheet(u"border:1px solid #6ebee8;\n"
"background:#9478f4;\n"
"border-top-left-radius: 0px;\n"
"border-top-right-radius: 0px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.horizontalLayout_24.addWidget(self.label_8)

        self.detection_quantity = QLabel(self.verticalFrame_12)
        self.detection_quantity.setObjectName(u"detection_quantity")
        self.detection_quantity.setStyleSheet(u"border:1px solid #6ebee8;\n"
"background:#64bee2;\n"
"border-top-left-radius: 0px;\n"
"border-top-right-radius: 0px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.horizontalLayout_24.addWidget(self.detection_quantity)

        self.horizontalLayout_24.setStretch(0, 3)
        self.horizontalLayout_24.setStretch(1, 2)

        self.verticalLayout_18.addLayout(self.horizontalLayout_24)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setSpacing(0)
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_11 = QLabel(self.verticalFrame_12)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setStyleSheet(u"border:1px solid #6ebee8;\n"
"background:#9478f4;\n"
"border-top-left-radius: 0px;\n"
"border-top-right-radius: 0px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.horizontalLayout_23.addWidget(self.label_11)

        self.detection_time = QLabel(self.verticalFrame_12)
        self.detection_time.setObjectName(u"detection_time")
        self.detection_time.setStyleSheet(u"border:1px solid #6ebee8;\n"
"background:#64bee2;\n"
"border-top-left-radius: 0px;\n"
"border-top-right-radius: 0px;\n"
"border-bottom-left-radius: 0;\n"
"border-bottom-right-radius: 0px;")

        self.horizontalLayout_23.addWidget(self.detection_time)

        self.horizontalLayout_23.setStretch(0, 3)
        self.horizontalLayout_23.setStretch(1, 2)

        self.verticalLayout_18.addLayout(self.horizontalLayout_23)


        self.horizontalLayout_20.addWidget(self.verticalFrame_12)


        self.verticalLayout_15.addLayout(self.horizontalLayout_20)

        self.verticalLayout_16 = QVBoxLayout()
        self.verticalLayout_16.setObjectName(u"verticalLayout_16")
        self.detection_result = QTextBrowser(self.verticalFrame_10)
        self.detection_result.setObjectName(u"detection_result")

        self.verticalLayout_16.addWidget(self.detection_result)


        self.verticalLayout_15.addLayout(self.verticalLayout_16)

        self.verticalLayout_15.setStretch(0, 4)
        self.verticalLayout_15.setStretch(1, 3)
        self.verticalLayout_15.setStretch(2, 4)

        self.verticalLayout_14.addLayout(self.verticalLayout_15)


        self.right.addWidget(self.verticalFrame_10)


        self.page_down.addLayout(self.right)

        self.page_down.setStretch(0, 2)
        self.page_down.setStretch(1, 5)
        self.page_down.setStretch(2, 2)

        self.verticalLayout.addLayout(self.page_down)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 9)

        self.gridLayout.addWidget(self.verticalFrame, 0, 0, 1, 1)


        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.title.setText(QCoreApplication.translate("MainWindow", u"\u57fa\u4e8eYOLO&Pyside6\u7684\u667a\u80fd\u53e3\u7f69\u68c0\u6d4b\u7cfb\u7edf", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u6a21\u578b\u9009\u62e9", None))
        self.model_select.setText("")
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u65b9\u5f0f\u9009\u62e9", None))
        self.image.setText("")
        self.video.setText("")
        self.camera.setText("")
        self.dirs.setText("")
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"\u8bbe\u7f6e\u7f6e\u4fe1\u5ea6", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"\u8bbe\u5b9aIOU", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"\u662f\u5426\u5b58\u50a8\u6570\u636e", None))
        self.save_data.setText("")
        self.label_voice.setText(QCoreApplication.translate("MainWindow", u"\u662f\u5426\u6253\u5f00\u8bed\u97f3\u63d0\u9192", None))
        self.tts_enable_checkbox.setText("")
        self.logo.setText("")
        self.finall_result.setText(QCoreApplication.translate("MainWindow", u"\u7ed3\u679c\u533a\u57df", None))
        self.yolo_start.setText(QCoreApplication.translate("MainWindow", u"\u70b9\u51fb\u5f00\u59cb\u68c0\u6d4b", None))
        self.video_start.setText("")
        self.video_stop.setText("")
        self.video_termination.setText("")
        self.upload_image.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4f20\u7684\u56fe\u50cf", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u7ed3\u679c", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"\u672a\u4f69\u6234\u53e3\u7f69\u4eba\u6570", None))
        self.detection_quantity.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u8017\u65f6", None))
        self.detection_time.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.detection_result.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:14pt; font-weight:500; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:'\u82f9\u65b9 \u7c97\u4f53'; font-size:12pt; font-weight:700;\">Author</span><span style=\" font-family:'\u82f9\u65b9 \u7c97\u4f53'; font-size:12pt;\">: </span><span style=\" font-family:'\u82f9\u65b9 \u7c97\u4f53'; font-size:12pt; font-weight:700;\">\u7b2c\u4e94\u7ec4</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-le"
                        "ft:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:'\u82f9\u65b9 \u7c97\u4f53'; font-size:12pt;\"><br /></p></body></html>", None))
    # retranslateUi

