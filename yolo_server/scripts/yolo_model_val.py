from ultralytics import YOLO
import argparse
import sys
from pathlib import Path

yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"

if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0, str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

from utils.paths import CHECKPOINTS_DIR, LOGS_DIR, CONFIGS_DIR
from logging_utils import setup_logging, rename_log_file
from performance_utils import time_it
from result_utils import log_results
from config_utils import load_yaml_config, log_parameters, merger_configs
from system_utils import log_device_info
from datainfo_utils import log_dataset_info

def parser_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Validation")
    parser.add_argument("--data", type=str, default="data.yaml", help="数据集配置文件")
    parser.add_argument("--weights", type=str, default="train_20250709_235047_yolo11n_best.pt", help="模型权重文件")
    parser.add_argument("--batch", type=int, default=16, help="训练批次大小")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备")
    parser.add_argument("--workers", type=int, default=8, help="训练数据加载线程数")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="数据集划分")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用yaml配置文件")
    return parser.parse_args()

def validate_model(model, yolo_args):
    results = model.val(**vars(yolo_args))
    return results

def main():
    args = parser_args()
    model_name = Path(args.weights).stem
    logger = setup_logging(base_path=LOGS_DIR, log_type="val", model_name=model_name, temp_log=True)
    logger.info("YOLO 口罩识别模型验证程序启动")
    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config(config_type='val')

        yolo_args, project_args = merger_configs(args, yaml_config, mode='val')

        log_device_info()

        log_parameters(project_args)

        log_dataset_info(args.data, mode=args.split)

        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / args.data
        if not data_path.exists():
            logger.error(f"数据集配置文件不存在: {data_path}")
            raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")

        model_path = Path(args.weights)
        if not model_path.is_absolute():
            model_path = CHECKPOINTS_DIR / args.weights
        if not model_path.exists():
            logger.error(f"模型权重文件不存在: {model_path}")
            raise FileNotFoundError(f"模型权重文件不存在: {model_path}")
        logger.info(f"加载模型权重文件: {model_path}")
        model = YOLO(model_path)

        def add_save_dir_train(trainer):
            trainer.validator.metrics.save_dir = trainer.validator.save_dir

        def add_save_dir_val(validator):
            validator.metrics.save_dir = validator.save_dir
        model.add_callback("on_train_end", add_save_dir_train)
        model.add_callback("on_val_end", add_save_dir_val)

        decorated_run_validation = time_it(iterations=1, name="模型验证", logger_instance=logger)(validate_model)
        results = decorated_run_validation(model, yolo_args)

        log_results(results)

        model_name_for_log = project_args.weights.replace(".pt", "")
        rename_log_file(logger, results.save_dir, model_name_for_log)

    except Exception as e:
        logger.error(f"模型验证程序异常: {e}")
    finally:
        logger.info("YOLO 口罩识别模型验证程序结束")

if __name__ == "__main__":
    main()
