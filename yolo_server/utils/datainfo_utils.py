import yaml
import logging
from pathlib import Path
from utils.paths import CONFIGS_DIR

COMMON_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp', '*.webp']

logger = logging.getLogger(__name__)

def get_dataset_info(data_config_name: str, mode: str = "train") -> tuple[int, list[str], int, str]:

    nc: int = 0
    classes_names: list[str] = []
    samples: int = 0

    if mode == 'infer':
        return 0, [], 0, "推理模式，不提供数据集来源信息"

    data_path: Path = CONFIGS_DIR / data_config_name

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"数据集配置文件 '{data_path}' 不存在。请检查 CONFIGS_DIR 或文件名称是否正确。")
        return nc, classes_names, samples, f"配置文件不存在: {data_path}"
    except yaml.YAMLError as e:
        logger.error(f"读取或解析数据集配置文件 '{data_path}' 失败: {e}")
        return nc, classes_names, samples, f"配置文件解析失败: {data_path}"
    except Exception as e:
        logger.error(f"打开或读取数据集配置文件 '{data_path}' 时发生未知错误: {e}")
        return nc, classes_names, samples, f"配置文件读取错误: {data_path}"

    nc = config.get("nc", 0)
    classes_names = config.get("names", [])
    split_key: str = mode
    dataset_root_from_config = config.get("path")
    split_relative_path_str: str = config.get(split_key)

    if not split_relative_path_str:
        logger.warning(
            f"配置文件 '{data_config_name}' 中未定义 '{split_key}' 模式的图片路径。尝试使用默认约定 '{mode}/images'。")
        split_relative_path_str = f"{mode}/images"

    if dataset_root_from_config:
        dataset_base_path = Path(dataset_root_from_config)
        if not dataset_base_path.is_absolute():
            dataset_base_path = data_path.parent / dataset_root_from_config
    else:
        dataset_base_path = data_path.parent

    split_path: Path = dataset_base_path / split_relative_path_str

    if split_path.is_dir():
        for ext in COMMON_IMAGE_EXTENSIONS:
            samples += len(list(split_path.glob(ext)))
        source = f"{mode.capitalize()} images from: {split_path}"
    else:
        logger.error(f"数据集图片路径不存在或不是一个目录: '{split_path}'。请检查配置文件中的路径或数据集完整性。")
        source = f"{mode.capitalize()} images not found at: {split_path}"

    return nc, classes_names, samples, source

def log_dataset_info(data_config_name: str, mode: str = 'train') -> dict:

    nc, classes_names, samples, source = get_dataset_info(data_config_name, mode)

    logger.info("=".center(40, '='))
    logger.info(f"数据集信息 ({mode.capitalize()} 模式)")
    logger.info('-' * 40)
    logger.info(f"{'Config File':<20}: {data_config_name}")
    logger.info(f"{'Class Count':<20}: {nc}")
    logger.info(f"{'Class Names':<20}: {', '.join(classes_names) if classes_names else '未知'}")
    logger.info(f"{'Sample Count':<20}: {samples}")
    logger.info(f"{'Data Source':<20}: {source}")
    logger.info('-' * 40)

    return {
        "config_file": data_config_name,
        "mode": mode,
        "class_count": nc,
        "class_names": classes_names,
        "sample_count": samples,
        "data_source": source
    }
