import argparse
from pathlib import Path
import sys
from ultralytics import YOLO

yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0, str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

from logging_utils import setup_logging, rename_log_file
from performance_utils import time_it
from config_utils import load_yaml_config, merger_configs, log_parameters
from paths import LOGS_DIR, PRETRAINED_MODELS_DIR, CHECKPOINTS_DIR
from system_utils import log_device_info
from datainfo_utils import log_dataset_info
from result_utils import log_results
from model_utils import copy_checkpoint_models

def parser_args():
    parser = argparse.ArgumentParser(description="YOLO Training")
    parser.add_argument("--data", type = str, default = "data.yaml", help = "数据集配置文件")
    parser.add_argument("--weights", type = str, default = "yolo11n.pt", help = "模型权重文件")
    parser.add_argument("--batch", type = int, default = 16, help = "训练批次大小")
    parser.add_argument("--epochs", type = int, default = 1, help = "训练轮数")
    parser.add_argument("--device", type = str, default = "cpu", help = "训练设备")
    parser.add_argument("--workers", type = int, default = 8, help = "训练数据加载线程数")
    parser.add_argument("--use_yaml", type = bool, default = True, help = "是否使用yaml配置文件")
    return parser.parse_args()

def run_training(model, yolo_args):
    results = model.train(**vars(yolo_args))
    return results

def main(logger):
    logger.info(f"YOLO 口罩识别模型训练脚本启动".center(50,"="))
    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config()

        yolo_args, project_args = merger_configs(args, yaml_config)

        log_device_info()

        log_dataset_info(data_config_name="data.yaml", mode="train")

        log_parameters(project_args)

        logger.info(f"初始化YOLO模型,加载模型{project_args.weights}")
        model_path = PRETRAINED_MODELS_DIR / project_args.weights
        if not model_path.exists():
            logger.warning(f"模型文件{model_path}不存在, 请将模型{project_args.weights}放入到{model_path}!")
            raise FileNotFoundError(f"模型文件{model_path}不存在")

        model = YOLO(model_path)
        decorated_run_training = time_it(iterations=1, name="模型训练", logger_instance=logger)(run_training)
        results = decorated_run_training(model, yolo_args)

        if results and hasattr(model.trainer, 'save_dir'):
            logger.info(f"模型的训练结果保存在: {model.trainer.save_dir}")
            log_results(results, model.trainer)
        else:
            logger.warning("此次训练过程中未产生有效的目录")

        copy_checkpoint_models(train_dir_path=model.trainer.save_dir,
                               model_filename=project_args.weights,
                               checkpoint_dir=CHECKPOINTS_DIR)

        model_name_for_log = project_args.weights.replace(".pt", "")
        rename_log_file(logger, str(model.trainer.save_dir), model_name_for_log)
        logger.info(f"YOLO 口罩识别模型训练结束".center(50, "="))
    except Exception as e:
        logger.error(f"发生未知错误: {e}")
    return

if __name__ == "__main__":
    args = parser_args()
    logger = setup_logging(base_path=LOGS_DIR,
                           log_type="train",
                           model_name=args.weights.replace(".pt", ""),
                           temp_log=True)
    main(logger)
