import logging
import yaml
from utils.paths import CONFIGS_DIR, RUNS_DIR
from utils.configs import COMMENTED_TRAIN_CONFIG, DEFAULT_TRAIN_CONFIG
from utils.configs import COMMENTED_VAL_CONFIG, DEFAULT_VAL_CONFIG
from utils.configs import COMMENTED_INFER_CONFIG, DEFAULT_INFER_CONFIG
from pathlib import Path
import argparse

VALID_YOLO_TRAIN_ARGS = set(DEFAULT_TRAIN_CONFIG)
VALID_YOLO_VAL_ARGS = set(DEFAULT_VAL_CONFIG)
VALID_YOLO_INFER_ARGS = set(DEFAULT_INFER_CONFIG)

BOOLEAN_PARAMS = {
    key for config in (DEFAULT_TRAIN_CONFIG, DEFAULT_VAL_CONFIG, DEFAULT_INFER_CONFIG)
        for key, value in config.items() if isinstance(value, bool)
}

logger = logging.getLogger(__name__)

def load_yaml_config(config_type: str='train'):

    config_path = CONFIGS_DIR / f"{config_type}.yaml"
    if not config_path.exists():
        logger.info(f"配置文件不存在，尝试生成默认配置文件: {config_path}")
        if config_type in ['train', 'val', 'infer']:
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                generate_default_config(config_type=config_type)
            except Exception as e:
                logger.error(f"生成配置文件失败: {e}")
                raise FileNotFoundError(f"无法生成配置文件: {config_path}")
        else:
            logger.error(f"不支持的配置文件类型: {config_type}")
            raise ValueError(f"不支持的配置文件类型: {config_type}")
    try:
        logger.info(f"开始加载配置文件: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"配置文件解析失败: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件发生错误: {e}")
        raise

def generate_default_config(config_type: str = "train"):

    config_path = CONFIGS_DIR / f"{config_type}.yaml"
    if config_type == 'train':
        config = COMMENTED_TRAIN_CONFIG
    elif config_type == 'val':
        config = COMMENTED_VAL_CONFIG
    elif config_type == 'infer':
        config = COMMENTED_INFER_CONFIG
    else:
        logger.error(f"未知的配置文件类型: {config_type}")
        raise ValueError(f"不支持的配置文件类型: {config_type}，当前仅支持 【train, val, infer】三种模式")
    try:
        with open(config_path, 'w', encoding="utf-8") as f:
            f.write(config)
        logger.info(f"生成默认 {config_type} 配置文件成功: {config_path}")
    except IOError as e:
        logger.error(f"写入默认 {config_type} 配置文件失败: {e}")
        raise e
    except Exception as e:
        logger.error(f"写入默认 {config_type} 配置文件发生未知错误: {e}")
        raise e

def _process_params_value(key, value):

    if key in BOOLEAN_PARAMS and isinstance(value, str):
        return value.lower() == "true"
    elif isinstance(value, str) and value.lower() == "none":
        return None
    elif key == "classes" and isinstance(value, str):
        if not value:
            return None
        try:
            return [int(i.strip()) for i in value.split(",")]
        except ValueError:
            logger.warning(f"警告: {key}参数的值{value}无法转换为列表,保留原值")
            return value
    else:
        return value

def merger_configs(args, yaml_config, mode="train"):

    if mode == "train":
        valid_args = VALID_YOLO_TRAIN_ARGS
        default_config = DEFAULT_TRAIN_CONFIG
    elif mode == "val":
        valid_args = VALID_YOLO_VAL_ARGS
        default_config = DEFAULT_VAL_CONFIG
    elif mode == "infer":
        valid_args = VALID_YOLO_INFER_ARGS
        default_config = DEFAULT_INFER_CONFIG
    else:
        logger.error(f"无效模式: {mode}, 支持的模式有: train, val, infer")
        raise ValueError(f"无效模式: {mode}, 支持的模式有: train, val, infer")

    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()
    merged_params = default_config.copy()

    if hasattr(args, "use_yaml") and args.use_yaml and yaml_config:
        for key, value in yaml_config.items():
            merged_params[key] = _process_params_value(key, value)
        logger.debug(f"已合并YAML参数: {merged_params}")

    cmd_args = {k: v for k, v in vars(args).items() if k != 'extra_args' and v is not None}
    for key, value in cmd_args.items():
        merged_params[key] = _process_params_value(key, value)
        setattr(project_args, f"{key}_specified", True)

    if hasattr(args, "extra_args"):
        if len(args.extra_args) % 2 != 0:
            logger.error(f"额外参数格式错误, 必须成对出现: 如 (--key value) ")
            raise ValueError("额外参数格式错误")

        for i in range(0, len(args.extra_args), 2):
            key = args.extra_args[i].lstrip("--")
            value = args.extra_args[i + 1]
            processed_value = _process_params_value(key, value)
            if processed_value == value:
                try:
                    if value.replace(".", "", 1).isdigit():
                        value = float(value) if '.' in value else int(value)
                    elif value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif value.lower() == "none":
                        value = None
                except ValueError:
                    logger.warning(f"无法将字符串 '{value}' 转换为有效的数据类型。")
                merged_params[key] = value
            else:
                merged_params[key] = processed_value
                setattr(project_args, f"{key}_specified", True)

    if 'data' in merged_params and merged_params['data']:
        data_path = Path(merged_params['data'])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path
        merged_params['data'] = str(data_path)

    if 'project' in merged_params and merged_params['project']:
        project_path = Path(merged_params['project'])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path
        merged_params['project'] = str(project_path)

    for key, value in merged_params.items():
        setattr(project_args, key, value)
        if key in valid_args:
            setattr(yolo_args, key, value)
        if key in yaml_config and not hasattr(project_args, f"{key}_specified"):
            setattr(project_args, f"{key}_specified", False)

    return yolo_args, project_args

def log_parameters(args, exclude_params=None):

    if exclude_params is None:
        exclude_params = ['use_yaml', 'log_level', 'extra_args']
    logger.info(f"开始记录模型参数信息".center(50, "="))
    logger.info("-" * 40)
    params_dict = {}
    for key, value in vars(args).items():
        if key not in exclude_params and not key.endswith('_specified'):
            source = '命令行' if getattr(args, f"{key}_specified", False) else 'Yaml'
            logger.info(f"{key:<20}: {value} (来源: {source})")
            params_dict[key] = {"value": value, "source": source}

    return params_dict
