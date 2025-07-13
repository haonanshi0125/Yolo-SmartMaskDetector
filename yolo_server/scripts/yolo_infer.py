import argparse
import cv2
from ultralytics import YOLO
from pathlib import Path
import sys

yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0, str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

from utils.logging_utils import setup_logging
from utils.config_utils import load_yaml_config, merger_configs, log_parameters
from utils.beautify import calculate_beautify_params
from utils.infer_frame import process_frame
from utils.tts_utils import process_tts_detection, init_tts  # 确保tts_utils已更新
from utils.paths import LOGS_DIR, CHECKPOINTS_DIR, YOLO_SERVER_ROOT

def parser_args():
    parser = argparse.ArgumentParser(description="口罩识别系统推理脚本")
    parser.add_argument('--model', default="train_20250709_235047_yolo11n_best.pt", help="模型权重文件")
    parser.add_argument('--source', type=str, default="0", help="输入源（图像/文件夹/视频/摄像头ID）")
    parser.add_argument('--conf', type=float, default=0.25, help="置信度阈值")
    parser.add_argument('--iou', type=float, default=0.45, help="IOU阈值")
    parser.add_argument('--show', type=bool, default=True, help="是否显示推理结果")
    parser.add_argument('--save', type=bool, default=True, help="是否保存推理结果")
    parser.add_argument('--save_txt', type=bool, default=True, help="是否保存推理结果txt文件")
    parser.add_argument('--save_conf', type=bool, default=True, help="是否保存推理结果置信度")
    parser.add_argument('--save_crop', type=bool, default=True, help="是否保存推理结果裁剪图片")
    parser.add_argument('--save_frames', type=bool, default=True, help="是否保存推理结果帧")
    parser.add_argument('--display_size', type=str, default="720", choices=["360", "480", "720", "1080", "1440"], help="显示图片大小")
    parser.add_argument('--beautify', type=bool, default=True, help="是否美化推理结果")
    parser.add_argument('--use_chinese_mapping', type=bool, default=True, help="是否使用中文映射")
    parser.add_argument('--font_size', type=int, default=22, help="字体大小")
    parser.add_argument('--line_width', type=int, default=4, help="边框宽度")
    parser.add_argument('--label_padding_x', type=int, default=5, help="标签内边距X")
    parser.add_argument('--label_padding_y', type=int, default=5, help="标签内边距Y")
    parser.add_argument('--radius', type=int, default=4, help="边框圆角半径")
    parser.add_argument('--use_yaml', type=bool, default=True, help="是否使用yaml配置文件")
    parser.add_argument('--tts_enable', type=bool, default=True, help="是否启用语音合成")
    parser.add_argument('--tts_duration', type=int, default=10, help="语音合成时长")
    parser.add_argument('--tts_interval', type=int, default=10, help="语音间隔")
    parser.add_argument('--tts_text_cn', default="请规范佩戴口罩！！！", help="中文语音合成内容")
    parser.add_argument('--tts_text_en', default="Please wear your mask properly!!!", help="英文语音合成内容")
    return parser.parse_args()

def main():

    args = parser_args()

    logger = setup_logging(
        base_path=LOGS_DIR,
        log_type="infer",
        model_name=args.model,
        temp_log=False
    )

    yaml_config = {}
    if args.use_yaml:
        yaml_config = load_yaml_config(config_type='infer')
    yolo_args, project_args = merger_configs(args, yaml_config, mode="infer")

    resolution_map = {
        "360": (640, 360),
        "480": (640, 480),
        "720": (1280, 720),
        "1080": (1920, 1080),
        "1440": (2560, 1440),
    }
    display_width, display_height = resolution_map[args.display_size]

    beautiful_params = calculate_beautify_params(
        current_image_height=display_height,
        current_image_width=display_width,
        base_font_size=args.font_size,
        base_line_width=args.line_width,
        base_label_padding_x=args.label_padding_x,
        base_label_padding_y=args.label_padding_y,
        base_radius=args.radius,
        ref_dim_for_scaling=720,
        font_path="D:/PycharmProjects/SafeYolo/yolo_server/utils/LXGWWenKai-Bold.ttf",
        text_color_bgr=(0, 0, 0),
        use_chinese_mapping=args.use_chinese_mapping,
        label_mapping=yaml_config["beautify_settings"]["label_mapping"],
        color_mapping=yaml_config["beautify_settings"]["color_mapping"]
    )

    # 初始化TTS引擎，获取中英文语音
    tts_engine, chinese_voice, english_voice = init_tts() if args.tts_enable else (None, None, None)
    if args.tts_enable and not tts_engine:
        logger.error("TTS语音合成引擎初始化失败")
        args.tts_enable = False
    tts_state = {
        'no_mask_start_time': None,
        'last_tts_time': None,
    }

    log_parameters(project_args)
    model = YOLO(CHECKPOINTS_DIR / args.model)
    logger.info(f"模型加载成功: {CHECKPOINTS_DIR / args.model}")

    source = args.source
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            raise RuntimeError(f"无法打开视频源: {source}")
        window_name = "Yolo11 Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_writer = None
        frames_dir = None
        yolo_args.stream = True
        yolo_args.show = False
        yolo_args.save = False
        logger.info(yolo_args)
        for idx, result in enumerate(model.predict(**vars(yolo_args))):
            if idx == 0:
                save_dir = YOLO_SERVER_ROOT / Path(result.save_dir)
                logger.info(f"此次推理结果保存路径: {save_dir}")
                if args.save_frames:
                    frames_dir = save_dir / "0_frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"保存帧图像路径: {frames_dir}")
                if args.save:
                    video_path = save_dir / "output.mp4"
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        fps,
                        (display_width, display_height),
                    )
                    logger.info(f"保存视频路径: {video_path}")
                    if video_writer:
                        logger.info(f"视频写入器创建成功")

            frame = result.orig_img
            # 调用更新后的TTS处理函数，传入中英文语音参数
            process_tts_detection(
                result,
                args.tts_enable,
                args.tts_duration,
                args.tts_interval,
                tts_engine,
                tts_state,
                args.tts_text_cn,
                args.tts_text_en,
                chinese_voice,
                english_voice
            )
            annotated_frame = process_frame(frame, result, project_args, beautiful_params)

            if video_writer:
                annotated_frame = cv2.resize(annotated_frame, dsize=(display_width, display_height))
                video_writer.write(annotated_frame)

            if frames_dir:
                cv2.imwrite(str(frames_dir / f"{idx}.png"), annotated_frame)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    else:
        yolo_args.stream = False
        yolo_args.show = False
        results = model.predict(**vars(yolo_args))
        save_dir = Path(results[0].save_dir)
        base_save_dir = save_dir / "beautify"
        base_save_dir.mkdir(parents=True, exist_ok=True)
        for idx, result in enumerate(results):
            annotated_frame = process_frame(result.orig_img, result, project_args, beautiful_params)
            if args.save:
                save_path = base_save_dir / f"{idx}.png"
                cv2.imwrite(str(save_path), annotated_frame)
    logger.info(f"推理结束".center(50, "_"))

if __name__ == "__main__":
    main()
