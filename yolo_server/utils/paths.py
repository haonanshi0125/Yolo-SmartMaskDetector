from pathlib import Path

# YOLO服务的项目根目录
YOLO_SERVER_ROOT = Path(__file__).resolve().parent.parent

# 配置文件目录
CONFIGS_DIR = YOLO_SERVER_ROOT / 'configs'

# 数据存放目录
DATA_DIR = YOLO_SERVER_ROOT / 'data'

# 结果存放目录
RUNS_DIR = YOLO_SERVER_ROOT / 'runs'

# 模型存放目录
MODELS_DIR = YOLO_SERVER_ROOT / 'models'

# 预训练模型存放目录
PRETRAINED_MODELS_DIR = MODELS_DIR / 'pretrained'

# 训练好的模型存放目录
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'

# 顶层脚本存放目录
SCRIPTS_DIR = YOLO_SERVER_ROOT / 'scripts'

# 日志文件存放目录
LOGS_DIR = YOLO_SERVER_ROOT / 'logs'

# 具体数据存放路径
RAW_DATA_DIR = DATA_DIR / 'raw'

# 原始图像存放路径
RAW_IMAGES_DIR = RAW_DATA_DIR / 'images'

# 原始非YOLO标注标签存放路径
ORIGINAL_ANNOTATIONS_DIR = RAW_DATA_DIR / 'original_annotations'

# YOLO txt 数据暂存目录
YOLO_STAGED_LABELS_DIR = RAW_DATA_DIR / 'yolo_staged_labels'
