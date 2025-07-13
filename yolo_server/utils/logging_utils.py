import logging
import sys
from datetime import datetime
from pathlib import Path
from colorlog import ColoredFormatter
import re

def setup_logging(base_path: Path,
            log_type: str = "general",
            model_name: str = None,
            log_level: str = logging.INFO,
            temp_log: bool = False,
            encoding: str = "utf-8",
            ):

    log_dir = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "temp" if temp_log else log_type.replace(" ", "_")
    log_filename_parts = [prefix, timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ", "_"))
    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = log_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s"))
    logger.addHandler(file_handler)

    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s : %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    logger.info(f"日志记录器开始初始化".center(50, "="))
    logger.info(f"日志记录器最低记录级别: {logging.getLevelName(log_level)}")
    logger.info(f"日志记录器初始化完成".center(50, "="))

    return logger

def rename_log_file(logger, save_dir, model_name):

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            old_log_file = Path(handler.baseFilename)
            timestamp_parts = re.findall(r"(\d{8}_\d{6})", old_log_file.stem, re.S)[0]
            train_prefix = Path(save_dir).name
            new_log_file = old_log_file.parent / f"{train_prefix}_{timestamp_parts}_{model_name}.log"
            handler.close()
            logger.removeHandler(handler)

            if old_log_file.exists():
                try:
                    old_log_file.rename(new_log_file)
                    logger.info(f"日志文件已经成功重命名: {new_log_file}")
                except OSError as e:
                    logger.error(f"重命名日志文件失败: {e}")
                    re_added_handler = logging.FileHandler(old_log_file, encoding='utf-8')
                    re_added_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s"))
                    logger.addHandler(re_added_handler)
                continue
            else:
                logger.warning(f"尝试重命名旧的日志文件 '{old_log_file}' 不存在")
                continue

            new_handler = logging.FileHandler(new_log_file, encoding='utf-8')
            new_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s"))
            logger.addHandler(new_handler)
            break
