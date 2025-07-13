import sys
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QApplication, QFileDialog, QLabel, QMessageBox
from PySide6.QtGui import QIcon, QPixmap, QImage
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QSpinBox, QHBoxLayout, QPushButton
import cv2
import numpy as np
import logging
import argparse
from PySide6.QtWidgets import QTextEdit

from yoloside6 import Ui_MainWindow
from SafeYolo.yolo_server.utils.infer_stream import stream_inference
from SafeYolo.yolo_server.utils.config_utils import load_yaml_config, merger_configs
from SafeYolo.yolo_server.utils.beautify import calculate_beautify_params
from SafeYolo.yolo_server.utils.tts_utils import init_tts, process_tts_detection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SafeUI")

class InferenceThread(QThread):

    frame_ready = Signal(np.ndarray, np.ndarray, object)
    progress_updated = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, source, weights, project_args, main_window, yolo_args, tts_engine=None, chinese_voice=None,
                 english_voice=None, tts_state=None):
        super().__init__()
        self.source = source
        self.weights = weights
        self.project_args = project_args
        self.main_window = main_window
        self.yolo_args = yolo_args
        self.running = True
        self.paused = False
        self.is_camera = ((source == "0") or (source == "1"))
        self.is_image = Path(source).is_file() and source.lower().endswith(('.jpg', '.jpeg', '.png'))
        self.is_directory = Path(source).is_dir()
        self.cap = None
        self.tts_engine = tts_engine
        self.chinese_voice = chinese_voice
        self.english_voice = english_voice
        self.tts_state = tts_state or {'no_mask_start_time': None, 'last_tts_time': None}

    def run(self):
        try:
            frame_interval = 1000 if (self.is_image or self.is_directory) else None
            if self.is_camera or (not self.is_image and not self.is_directory):
                self.cap = cv2.VideoCapture(int(self.source) if self.is_camera else self.source)
                if not self.cap.isOpened():
                    self.error_occurred.emit(f"无法打开{'摄像头' if self.is_camera else '视频'}")
                    return
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_camera else 0
                fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                frame_interval = 1000 / fps
            else:
                source_path = Path(self.source)
                image_files = [source_path] if self.is_image else sorted(source_path.glob("*.[jp][pn][gf]"))
                total_frames = len(image_files)
                if total_frames == 0:
                    self.error_occurred.emit("目录中无图片文件")
                    return

            yolo_infer_kwargs = vars(self.yolo_args).copy()
            yolo_infer_kwargs['conf'] = self.main_window.ui.conf_num.value()
            yolo_infer_kwargs['iou'] = self.main_window.ui.iou_number.value()
            yolo_infer_kwargs['stream'] = True
            yolo_infer_kwargs['show'] = False
            yolo_infer_kwargs['save_txt'] = getattr(self.project_args, 'save_txt', False)
            yolo_infer_kwargs['save_conf'] = getattr(self.project_args, 'save_conf', False)
            yolo_infer_kwargs['save_crop'] = getattr(self.project_args, 'save_crop', False)

            project_config_for_stream = vars(self.project_args).copy()
            idx = 0
            for raw_frame, annotated_frame, result in stream_inference(
                weights=self.weights,
                source=self.source,
                project_args=project_config_for_stream,
                yolo_args=yolo_infer_kwargs,
                pause_callback=lambda: self.paused or not self.running
            ):
                if not self.running:
                    break
                if self.paused:
                    logger.debug("InferenceThread 暂停")
                    self.msleep(100)
                    continue
                self.frame_ready.emit(raw_frame, annotated_frame, result)

                # 语音播报参数可从project_args或yaml_config获取
                tts_enable = getattr(self.project_args, 'tts_enable', True)
                tts_duration = getattr(self.project_args, 'tts_duration', 10)
                tts_interval = getattr(self.project_args, 'tts_interval', 10)
                tts_text_cn = getattr(self.project_args, 'tts_text_cn', "请规范佩戴口罩！！！")
                tts_text_en = getattr(self.project_args, 'tts_text_en', "Please wear your mask properly!!!")

                # 仅在摄像头或视频输入时启用 TTS
                if self.is_camera or (not self.is_image and not self.is_directory):
                    process_tts_detection(
                        result,
                        tts_enable,
                        tts_duration,
                        tts_interval,
                        self.tts_engine,
                        self.tts_state,
                        tts_text_cn,
                        tts_text_en,
                        self.chinese_voice,
                        self.english_voice
                    )

                if not self.is_camera:
                    idx += 1
                    progress = int(idx / total_frames * 100) if total_frames > 0 else 0
                    self.progress_updated.emit(progress)
                self.msleep(int(frame_interval) if frame_interval else 10)
        except Exception as e:
            self.error_occurred.emit(f"推理失败: {str(e)}")
            logger.error(f"InferenceThread 错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
            logger.info("InferenceThread 已清理")

    def get_yolo_args(self):

        return {
            'conf': self.main_window.ui.conf_num.value(),
            'iou': self.main_window.ui.iou_number.value(),
            'imgsz': getattr(self.project_args, 'img_width', 640),
            'stream': True,
            'save_txt': getattr(self.project_args, 'save_txt', False),
            'save_conf': getattr(self.project_args, 'save_conf', False),
            'save_crop': getattr(self.project_args, 'save_crop', False),
        }

    def terminate(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        super().terminate()

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_ui()
        self.yolo_args = argparse.Namespace()
        self.project_args = argparse.Namespace()
        self.yaml_config = {}
        self.inference_thread = None
        self.source = None
        self.model_path = None
        self.is_camera = False
        self.is_image = False
        self.is_directory = False
        self.resolution_map = {
            "360": (640, 360),
            "480": (640, 480),
            "720": (1280, 720),
            "1080": (1920, 1080),
            "1440": (2560, 1440),
        }
        self.yaml_config = load_yaml_config(config_type="infer")
        initial_ui_args_namespace = argparse.Namespace(
            conf=self.ui.conf_num.value(),
            iou=self.ui.iou_number.value(),
            save=self.ui.save_data.isChecked(),
            model_path=None,
            source_path=None,
            use_yaml=True
        )
        self.yolo_args, self.project_args = merger_configs(
            initial_ui_args_namespace,
            self.yaml_config,
            mode="infer"
        )
        self.tts_engine, self.chinese_voice, self.english_voice = init_tts()
        self.tts_state = {'no_mask_start_time': None, 'last_tts_time': None}

        display_size_key = str(getattr(self.project_args, 'display_size', '720'))
        display_width, display_height = self.resolution_map.get(display_size_key, self.resolution_map["720"])
        setattr(self.project_args, 'display_width', display_width)
        setattr(self.project_args, 'display_height', display_height)
        setattr(self.project_args, 'img_width', display_width)
        setattr(self.project_args, 'img_height', display_height)

        beautify_calculated_params = calculate_beautify_params(
            current_image_height=display_height,
            current_image_width=display_width,
            base_font_size=self.project_args.beautify_settings['base_font_size'],
            base_line_width=self.project_args.beautify_settings['base_line_width'],
            base_label_padding_x=self.project_args.beautify_settings['base_label_padding_x'],
            base_label_padding_y=self.project_args.beautify_settings['base_label_padding_y'],
            base_radius=self.project_args.beautify_settings['base_radius'],
            ref_dim_for_scaling=720,
            font_path=self.project_args.beautify_settings['font_path'],
            text_color_bgr=self.project_args.beautify_settings['text_color_bgr'],
            use_chinese_mapping=self.project_args.beautify_settings['use_chinese_mapping'],
            label_mapping=self.project_args.beautify_settings['label_mapping'],
            color_mapping=self.project_args.beautify_settings['color_mapping']
        )
        setattr(self.project_args, 'beautify_calculated_params', beautify_calculated_params)

        logger.info(f"最终 YOLO 参数: {vars(self.yolo_args)}")
        logger.info(f"最终项目参数: {vars(self.project_args)}")
        logger.info(f"计算后的美化参数: {vars(getattr(self.project_args, 'beautify_calculated_params')) if isinstance(getattr(self.project_args, 'beautify_calculated_params'), argparse.Namespace) else getattr(self.project_args, 'beautify_calculated_params')}")

        self.ui.conf_num.setValue(getattr(self.yolo_args, 'conf', 0.25))
        self.ui.iou_number.setValue(getattr(self.yolo_args, 'iou', 0.45))
        self.ui.save_data.setChecked(getattr(self.project_args, 'save', False))
        self.ui.tts_enable_checkbox.setChecked(getattr(self.project_args, 'tts_enable', True))

        if getattr(self.project_args, 'model_path', None):
            self.model_path = getattr(self.project_args, 'model_path')
            self.ui.model_name.setText(Path(self.model_path).name)
        if getattr(self.project_args, 'source_path', None):
            self.source = getattr(self.project_args, 'source_path')
            if self.source == "0":
                self.is_camera = True
                self.ui.upload_image.setText("摄像头已选择，点击开始播放")
            elif Path(self.source).is_file():
                self.is_image = self.source.lower().endswith(('.jpg', '.jpeg', '.png'))
                self.show_preview(self.source, is_video=not self.is_image)
            elif Path(self.source).is_dir():
                self.is_directory = True
                self.ui.upload_image.setText(f"已选择目录: {Path(self.source).name}（无图片预览）")

        self.connect_signals()
        self.update_button_states()

    def project_args_for_thread(self):

        return self.project_args

    def setup_ui(self):

        self.ui.model_name.setReadOnly(True)
        self.ui.model_name.setPlaceholderText("请选择模型文件...")
        self.ui.model_name.setStyleSheet("QLineEdit { border: 1px solid gray; padding: 2px; text-overflow: ellipsis; }")
        self.ui.model_name.setMaximumWidth(200)

        icon_path = Path(__file__).parent.parent / "yolo_server" / "icons" / "folder.png"
        if icon_path.exists():
            self.ui.model_select.setIcon(QIcon(str(icon_path)))
            self.ui.model_select.setText("")
        else:
            self.ui.model_select.setText("选择模型")

        self.ui.upload_image.setScaledContents(True)
        self.ui.upload_image.setText("上传预览")
        self.ui.finall_result.setScaledContents(True)
        self.ui.finall_result.setText("检测结果")
        self.ui.video_progressBar.setValue(0)
        self.ui.video_progressBar.setTextVisible(True)
        self.ui.video_progressBar.setValue(0)
        self.ui.video_progressBar.setTextVisible(True)
        self.ui.conf_num.setRange(0.0, 1.0)
        self.ui.conf_num.setSingleStep(0.05)
        self.ui.conf_num.setValue(0.25)
        self.ui.conf_num.setEnabled(True)
        self.ui.conf_slider.setMinimum(0)
        self.ui.conf_slider.setMaximum(100)
        self.ui.conf_slider.setValue(25)
        self.ui.iou_number.setRange(0.0, 1.0)
        self.ui.iou_number.setSingleStep(0.05)
        self.ui.iou_number.setValue(0.45)
        self.ui.iou_number.setEnabled(True)
        self.ui.iou_slider.setMinimum(0)
        self.ui.iou_slider.setMaximum(100)
        self.ui.iou_slider.setValue(45)
        self.ui.save_data.setChecked(False)
        self.ui.tts_enable_checkbox.setChecked(True)
        self.ui.detection_quantity.setText("-")
        self.ui.detection_time.setText("-")
        self.ui.detection_result.setText("无检测结果")
        self.statusBar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("FPS: 0")
        self.statusBar.addWidget(self.status_label)
        self.statusBar.addWidget(self.fps_label)

        try:
            from PySide6 import QtWidgets
            if hasattr(self.ui, 'verticalLayout') and isinstance(self.ui.verticalLayout, QtWidgets.QVBoxLayout):
                self.ui.log_display = QTextEdit()
                self.ui.log_display.setReadOnly(True)
                self.ui.log_display.setMaximumHeight(100)
                class QTextEditHandler(logging.Handler):
                    def __init__(self, text_edit):
                        super().__init__()
                        self.text_edit = text_edit
                    def emit(self, record):
                        msg = self.format(record)
                        self.text_edit.append(msg)
                text_handler = QTextEditHandler(self.ui.log_display)
                text_handler.setLevel(logging.DEBUG)
                text_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(text_handler)
                self.ui.verticalLayout.addWidget(self.ui.log_display)
            else:
                logger.warning("无法添加日志显示，请检查 UI 布局，self.ui.verticalLayout 不存在或不是 QVBoxLayout")
        except AttributeError:
            logger.warning("无法添加日志显示，请检查 UI 布局")

    def connect_signals(self):

        self.ui.model_select.clicked.connect(self.select_model)
        self.ui.video.clicked.connect(self.select_video)
        self.ui.image.clicked.connect(self.select_image)
        self.ui.dirs.clicked.connect(self.select_dirs)
        self.ui.camera.clicked.connect(self.select_camera)
        self.ui.yolo_start.clicked.connect(self.start_inference)
        self.ui.video_start.clicked.connect(self.start_video)
        self.ui.video_stop.clicked.connect(self.stop_video)
        self.ui.video_termination.clicked.connect(self.terminate_video)
        self.ui.conf_num.valueChanged.connect(self.sync_conf_slider)
        self.ui.conf_slider.valueChanged.connect(self.sync_conf_num)
        self.ui.iou_number.valueChanged.connect(self.sync_iou_slider)
        self.ui.iou_slider.valueChanged.connect(self.sync_iou_num)
        self.ui.save_data.stateChanged.connect(self._update_save_param_from_ui)
        self.ui.tts_enable_checkbox.stateChanged.connect(self._update_tts_enable_from_ui)

    def sync_conf_slider(self, value):

        self.ui.conf_slider.setValue(int(value * 100))
        logger.debug(f"更新 conf_slider 值: {value}")

    def sync_conf_num(self):

        value = self.ui.conf_slider.value() / 100.0
        self.ui.conf_num.setValue(value)
        logger.debug(f"更新 conf_num 值: {value}")

    def sync_iou_slider(self):

        self.ui.iou_slider.setValue(int(self.ui.iou_number.value() * 100))
        logger.debug(f"更新 iou_slider 值: {self.ui.iou_number.value()}")

    def sync_iou_num(self):

        value = self.ui.iou_slider.value() / 100.0
        self.ui.iou_number.setValue(value)
        logger.debug(f"更新 iou_number 值: {value}")

    def _update_save_param_from_ui(self):

        is_checked = self.ui.save_data.isChecked()
        setattr(self.project_args, 'save', is_checked)
        setattr(self.project_args, 'save_txt', is_checked)
        setattr(self.project_args, 'save_conf', is_checked)
        setattr(self.project_args, 'save_crop', is_checked)
        logger.info(f"UI 更新：保存数据功能设置为: {is_checked}")
        self.update_button_states()

    def _update_tts_enable_from_ui(self):
        is_checked = self.ui.tts_enable_checkbox.isChecked()
        setattr(self.project_args, 'tts_enable', is_checked)
        logger.info(f"UI 更新：语音播报功能设置为: {is_checked}")

    def select_model(self):

        try:
            default_dir = Path(__file__).parent.parent / "yolo_server" / "models" / "checkpoints"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择 YOLO 模型文件", str(default_dir), "YOLO 模型文件 (*.pt);;所有文件 (*.*)"
            )
            if file_path:
                self.model_path = file_path
                self.ui.model_name.setText(Path(file_path).name)
                setattr(self.project_args, 'model_path', file_path)
                logger.info(f"选择的模型: {self.model_path}")
            else:
                self.model_path = None
                self.ui.model_name.setText("")
                setattr(self.project_args, 'model_path', None)
                logger.info("未选择模型")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择模型失败")
            logger.error(f"选择模型失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择模型失败: {str(e)}")

    def select_video(self):

        try:
            default_dir = Path(__file__).parent.parent / "yolo_server" / "test"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", str(default_dir), "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)"
            )
            if file_path:
                self.terminate_video()
                self.source = file_path
                setattr(self.project_args, 'source_path', file_path)
                self.is_camera = False
                self.is_image = False
                self.is_directory = False
                self.show_preview(file_path, is_video=True)
                logger.info(f"选择的视频: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                setattr(self.project_args, 'source_path', None)
                logger.info("未选择视频")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择视频失败")
            logger.error(f"选择视频失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择视频失败: {str(e)}")

    def select_image(self):

        try:
            default_dir = Path(__file__).parent.parent / "yolo_server" / "test"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片文件", str(default_dir), "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*.*)"
            )
            if file_path:
                self.terminate_video()
                self.source = file_path
                setattr(self.project_args, 'source_path', file_path)
                self.is_camera = False
                self.is_image = True
                self.is_directory = False
                self.show_preview(file_path, is_video=False)
                logger.info(f"选择的图片: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                setattr(self.project_args, 'source_path', None)
                logger.info("未选择图片")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择图片失败")
            logger.error(f"选择图片失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择图片失败: {str(e)}")

    def select_dirs(self):

        try:
            default_dir = Path(__file__).parent.parent / "yolo_server" / "test"
            default_dir = default_dir.resolve()
            default_dir.mkdir(parents=True, exist_ok=True)
            dir_path = QFileDialog.getExistingDirectory(self, "选择图片或视频目录", str(default_dir))
            if dir_path:
                self.terminate_video()
                self.source = dir_path
                setattr(self.project_args, 'source_path', dir_path)
                self.is_camera = False
                self.is_image = False
                self.is_directory = True
                for img_path in Path(dir_path).glob("*.[jp][pn][gf]"):
                    self.show_preview(str(img_path), is_video=False)
                    break
                else:
                    self.ui.upload_image.setText(f"已选择目录: {Path(dir_path).name}（无图片预览）")
                logger.info(f"选择的目录: {self.source}")
            else:
                self.source = None
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText("上传预览")
                setattr(self.project_args, 'source_path', None)
                logger.info("未选择目录")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择目录失败")
            logger.error(f"选择目录失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择目录失败: {str(e)}")

    def select_camera(self):
        class CameraDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
                self.setStyleSheet("""
                                QDialog {
                                    background: #f8fafd;
                                    border: 3px solid #4e8cff;
                                    border-radius: 5px;
                                }
                            """)
                layout = QVBoxLayout(self)
                label = QLabel("请输入摄像头编号 (本机为0，外接为1/2...)")
                layout.addWidget(label)
                self.spin = QSpinBox()
                self.spin.setRange(0, 10)
                self.spin.setValue(0)
                self.spin.setStyleSheet("""
                                    font-size:18px; 
                                    padding:4px 8px; 
                                    border:1px solid #b0b0b0; 
                                    border-radius:6px; 
                                    background:#fff;
                                """)
                layout.addWidget(self.spin)
                btn_layout = QHBoxLayout()
                self.ok_btn = QPushButton("确认")
                self.ok_btn.setStyleSheet("font-size:15px; min-width:60px; min-height:28px; border-radius:6px; background:#4e8cff; color:#fff;")
                self.cancel_btn = QPushButton("取消")
                self.cancel_btn.setStyleSheet("font-size:15px; min-width:60px; min-height:28px; border-radius:6px; background:#b0b0b0; color:#fff;")
                btn_layout.addWidget(self.ok_btn)
                btn_layout.addWidget(self.cancel_btn)
                layout.addLayout(btn_layout)
                self.ok_btn.clicked.connect(self.accept)
                self.cancel_btn.clicked.connect(self.reject)
            def get_value(self):
                return self.spin.value()
        try:
            self.terminate_video()
            dialog = CameraDialog(self)
            if dialog.exec() == QDialog.Accepted:
                cam_id = dialog.get_value()
                self.source = str(cam_id)
                setattr(self.project_args, 'source_path', str(cam_id))
                self.is_camera = True
                self.is_image = False
                self.is_directory = False
                self.ui.upload_image.setText(f"摄像头{cam_id}已选择，点击开始播放")
                logger.info(f"选择输入: 摄像头{cam_id}")
            else:
                self.source = None
                self.is_camera = False
                self.ui.upload_image.setText("上传预览")
                setattr(self.project_args, 'source_path', None)
                logger.info("未选择摄像头")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 选择摄像头失败")
            logger.error(f"选择摄像头失败: {str(e)}")
            self.ui.log_display.append(f"错误: 选择摄像头失败: {str(e)}")

    def show_preview(self, file_path, is_video=False):

        try:
            if is_video:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        self.ui.upload_image.setText("无法读取视频")
                        return
                else:
                    self.ui.upload_image.setText("无法打开视频")
                    return
            else:
                frame = cv2.imread(file_path)
                if frame is None:
                    self.ui.upload_image.setText("无法读取图片")
                    return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.upload_image.setPixmap(pixmap.scaled(self.ui.upload_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logger.debug(f"显示预览: {file_path}, shape: {h}x{w}")
        except Exception as e:
            self.status_label.setText("预览失败")
            logger.error(f"显示预览失败: {str(e)}")
            self.ui.log_display.append(f"错误: 显示预览失败: {str(e)}")

    def start_inference(self):

        try:
            if not self.model_path:
                self.status_label.setText("请先选择模型文件")
                return
            if not self.source:
                self.status_label.setText("请先选择输入源")
                return
            self.start_video()
        except Exception as e:
            self.status_label.setText(f"错误: 开始推理失败")
            logger.error(f"开始推理失败: {str(e)}")
            self.ui.log_display.append(f"错误: 开始推理失败: {str(e)}")

    def start_video(self):

        try:
            if not self.source:
                self.status_label.setText("请先选择输入源")
                self.ui.upload_image.setText("请先选择视频、摄像头、图片或目录")
                return
            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.paused = False
                self.status_label.setText("正在推理")
                logger.info("推理已恢复")
                self.update_button_states()
                return
            self.inference_thread = InferenceThread(
                self.source,
                self.model_path,
                self.project_args_for_thread(),
                self,
                self.yolo_args,
                tts_engine=self.tts_engine,
                chinese_voice=self.chinese_voice,
                english_voice=self.english_voice,
                tts_state=self.tts_state
            )
            self.inference_thread.frame_ready.connect(self.update_frames)
            self.inference_thread.progress_updated.connect(self.update_progress)
            self.inference_thread.error_occurred.connect(self.show_error)
            self.inference_thread.finished.connect(self.video_finished)
            self.inference_thread.start()
            self.status_label.setText("正在推理")
            logger.info("推理已开始")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 开始推理失败")
            logger.error(f"开始推理失败: {str(e)}")
            self.ui.log_display.append(f"错误: 开始推理失败: {str(e)}")

    def stop_video(self):

        try:
            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.paused = True
                self.status_label.setText("已暂停")
                logger.info("推理已暂停")
            self.update_button_states()
        except Exception as e:
            self.status_label.setText(f"错误: 暂停失败")
            logger.error(f"暂停失败: {str(e)}")
            self.ui.log_display.append(f"错误: 暂停失败: {str(e)}")

    def terminate_video(self):

        try:
            logger.info("开始终止线程")
            if self.inference_thread and self.inference_thread.isRunning():
                self.inference_thread.running = False
                self.inference_thread.quit()
                self.inference_thread.wait(500)
                if self.inference_thread.isRunning():
                    self.inference_thread.terminate()
                self.inference_thread = None
                logger.info("推理已终止")
            if not self.is_image:
                self.ui.upload_image.setText("上传预览")
            self.ui.finall_result.setText("检测结果")
            self.ui.video_progressBar.setValue(0)
            self.ui.detection_quantity.setText("-")
            self.ui.detection_time.setText("-")
            self.ui.detection_result.setText("无检测结果")
            self.status_label.setText("就绪")
            self.update_button_states()
            logger.info("UI 已重置")
        except Exception as e:
            self.status_label.setText(f"错误: 停止失败")
            logger.error(f"停止失败: {str(e)}")
            self.ui.log_display.append(f"错误: 停止失败: {str(e)}")

    def closeEvent(self, event):

        try:
            logger.info("开始关闭窗口")
            self.terminate_video()
            event.accept()
            logger.info("窗口已关闭")
        except Exception as e:
            logger.error(f"关闭窗口失败: {str(e)}")
            self.ui.log_display.append(f"错误: 关闭窗口失败: {str(e)}")
            event.ignore()

    def update_frames(self, raw_frame, annotated_frame, result):

        try:
            start_time = cv2.getTickCount()
            display_width = getattr(self.project_args, 'display_width', self.resolution_map["720"][0])
            display_height = getattr(self.project_args, 'display_height', self.resolution_map["720"][1])

            raw_frame_resized = cv2.resize(raw_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(raw_frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.upload_image.setPixmap(pixmap.scaled(self.ui.upload_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            annotated_frame_resized = cv2.resize(annotated_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(annotated_frame_resized, cv2.COLOR_BGR2RGB)
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.ui.finall_result.setPixmap(pixmap.scaled(self.ui.finall_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            total_face_count = 0
            masked_face_count = 0
            unmasked_face_count = 0
            total_time = 0.0

            if result is not None and hasattr(result, 'boxes'):
                boxes = result.boxes if result.boxes is not None else []
                total_face_count = len(boxes)
                unmasked_face_count = sum(1 for box in boxes if hasattr(box, 'cls') and int(box.cls) == 0)
                masked_face_count = total_face_count - unmasked_face_count
                total_time = sum(result.speed.values()) if hasattr(result, 'speed') and result.speed is not None else 0.0

            self.ui.detection_quantity.setText(f"{unmasked_face_count} 人")
            self.ui.detection_time.setText(f"{total_time:.2f} ms")
            self.ui.detection_result.setText(f"""共检测到人脸数量: {total_face_count} 个\n
其中佩戴口罩人员数量: {masked_face_count} 个\n
其中未佩戴口罩人员数量: {unmasked_face_count} 个\n
当前帧检测耗时: {total_time:.2f} ms
""")

            end_time = cv2.getTickCount()
            frame_time = ((end_time - start_time) / cv2.getTickFrequency()) * 1000
            fps = 1000 / frame_time if frame_time > 0 else 0
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.status_label.setText("正在推理")
            logger.debug(f"帧更新耗时: {frame_time:.2f}ms")
        except Exception as e:
            self.status_label.setText("更新帧失败")
            logger.error(f"更新帧失败: {str(e)}")
            self.ui.log_display.append(f"错误: 更新帧失败: {str(e)}")

    def update_progress(self, progress):

        self.ui.video_progressBar.setValue(progress)

    def video_finished(self):

        self.status_label.setText("推理完成")
        logger.info("视频处理完成")
        if self.is_image:
             pass
        else:
            self.terminate_video()

    def show_error(self, error_msg):

        self.status_label.setText(f"错误: {error_msg}")
        self.ui.upload_image.setText(error_msg)
        self.ui.finall_result.setText(error_msg)
        self.ui.detection_quantity.setText("-")
        self.ui.detection_time.setText("-")
        self.ui.detection_result.setText("无检测结果")
        self.terminate_video()
        logger.error(f"错误: {error_msg}")
        self.ui.log_display.append(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)

    def update_button_states(self):

        has_source = getattr(self, 'source', None) is not None
        has_model = getattr(self, 'model_path', None) is not None
        is_running = bool(getattr(self, 'inference_thread', None) and self.inference_thread.isRunning())
        is_paused = bool(is_running and getattr(self.inference_thread, 'paused', False))
        is_camera_selected = getattr(self, 'is_camera', False)

        self.ui.yolo_start.setEnabled(has_source and has_model and not is_running)
        self.ui.video_start.setEnabled(has_source and has_model and (not is_running or is_paused))
        self.ui.video_stop.setEnabled(is_running and not is_paused)
        self.ui.video_termination.setEnabled(is_running or is_paused)
        self.ui.model_select.setEnabled(not is_running and not is_paused)
        self.ui.video.setEnabled(not is_running and not is_paused)
        self.ui.image.setEnabled(not is_running and not is_paused)
        self.ui.dirs.setEnabled(not is_running and not is_paused)
        self.ui.camera.setEnabled(not is_running and not is_paused)
        self.ui.video_progressBar.setEnabled(has_source and not is_camera_selected)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    from PySide6 import QtWidgets
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
