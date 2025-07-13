import platform
import sys
import os
import psutil
import subprocess
import time
import importlib.metadata
import torch
import cpuinfo
import ultralytics
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

try:
    import onnx
except ImportError:
    onnx = None

_PYNVML_AVAILABLE = False
try:
    from pynvml import *
    _PYNVML_AVAILABLE = True
except ImportError:
    pass

def format_merge(bytes_size: int | float | None):

    if bytes_size is None or not isinstance(bytes_size, (int, float)):
        return "N/A"
    bytes_size_float = float(bytes_size)
    if bytes_size_float >= 1024 ** 3:
        return f"{bytes_size_float / (1024 ** 3):.2f} GB"
    return f"{bytes_size_float / (1024 ** 2):.2f} MB"

def _get_package_version(package_name: str):

    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"
    except Exception as e:
        logging.getLogger("YOLO_Training").warning(f"获取包 '{package_name}' 版本失败: {e}")
        return "获取失败"

def _get_nvidia_driver_version(logger: logging.Logger):

    try:
        results = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        return results.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"执行 'nvidia-smi' 命令失败：{e}. (请确保NVIDIA驱动已正确安装并添加到PATH)")
        return "获取失败或未安装NVIDIA驱动"
    except Exception as e:
        logger.error(f"获取NVIDIA驱动程序版本时发生未知错误: {e}")
        return "获取失败或未安装NVIDIA驱动"

def _get_realtime_gpu_metrics(gpu_index: int, logger: logging.Logger):

    metrics = {}
    if not _PYNVML_AVAILABLE:
        metrics[f"GPU_{gpu_index}_实时使用信息"] = "pynvml库未安装或不可用"
        return metrics
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        metrics[f"GPU_{gpu_index}_利用率"] = f"GPU:{utilization.gpu}% / Mem:{utilization.memory}%"
        metrics[f"GPU_{gpu_index}_实时使用显存"] = format_merge(memory_info.used)
    except NVMLError as error:
        logger.warning(f"获取GPU {gpu_index}实时信息失败(pynvml): {error}")
        metrics[f"GPU_{gpu_index}_实时使用信息"] = "获取失败"
    except Exception as e:
        logger.warning(f"获取GPU {gpu_index}实时信息发生未知错误: {e}")
        metrics[f"GPU_{gpu_index}_实时使用信息"] = "获取失败"
    return metrics

@lru_cache(maxsize=1)
def get_device_info():

    logger = logging.getLogger("YOLO_Training")
    results = {
        "基本设备信息": {
            "操作系统": f"{platform.system()} {platform.release()}",
            "Python版本": platform.python_version(),
            "Python解释器路径": sys.executable,
            "Python虚拟环境": os.environ.get("CONDA_DEFAULT_ENV", "未知"),
            "当前检测时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "主机名": platform.node(),
            "当前用户": os.getenv('USER') or os.getenv('USERNAME', '未知用户'),
        },
        "CPU信息": {
            "CPU型号": cpuinfo.get_cpu_info().get('brand_raw', '未知CPU型号'),
            "CPU物理核心数": psutil.cpu_count(logical=False) or 0,
            "CPU逻辑核心数": psutil.cpu_count(logical=True) or 0,
            "CPU使用率": f"{psutil.cpu_percent()}%",
        },
        "GPU信息": {},
        "内存信息": {},
        "环境信息": {},
        "磁盘信息": {}
    }
    cuda_available = torch.cuda.is_available()
    _pynvml_local_initialized = False
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        results["GPU信息"] = {
            "CUDA是否可用": True,
            "CUDA版本": torch.version.cuda,
            "NVIDIA驱动程序版本": _get_nvidia_driver_version(logger),
            "可用的GPU数量": gpu_count,
        }
        all_gpus_detail = []
        if _PYNVML_AVAILABLE:
            try:
                nvmlInit()
                _pynvml_local_initialized = True
            except NVMLError as error:
                logger.warning(f"初始化pynvml失败: {error}. 部分GPU实时信息可能无法获取。")
                _pynvml_local_initialized = False
        for i in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_detail = {
                    f"GPU_{i}_型号": torch.cuda.get_device_name(i),
                    f"GPU_{i}_总显存": format_merge(props.total_memory),
                    f"GPU_{i}_算力": f"{props.major}.{props.minor}",
                    f"GPU_{i}_多处理器数量": props.multi_processor_count,
                    f"GPU_{i}_PyTorch_已分配显存": format_merge(torch.cuda.memory_allocated(i)),
                    f"GPU_{i}_PyTorch_已缓存显存": format_merge(torch.cuda.memory_reserved(i)),
                }
                if _pynvml_local_initialized:
                    gpu_detail.update(_get_realtime_gpu_metrics(i, logger))
                else:
                    gpu_detail[f"GPU_{i}_实时使用信息"] = "pynvml未加载或初始化失败"
                all_gpus_detail.append(gpu_detail)
            except Exception as e:
                logger.error(f"获取GPU {i}详细信息失败: {e}")
                all_gpus_detail.append({f"GPU_{i}_信息": "获取失败或异常"})
        results["GPU详细列表"] = all_gpus_detail
        if _pynvml_local_initialized:
            try:
                nvmlShutdown()
            except NVMLError as error:
                logger.warning(f"关闭pynvml失败: {error}")
    else:
        results["GPU信息"] = {
            "CUDA是否可用": False,
            "CUDA版本": "N/A",
            "NVIDIA驱动程序版本": "N/A",
            "可用的GPU数量": 0,
        }
        results["GPU详细列表"] = {"信息": "未检测到CUDA可用GPU (当前使用CPU)"}
    virtual_mem = psutil.virtual_memory()
    results["内存信息"] = {
        "总内存": format_merge(virtual_mem.total),
        "已使用内存": format_merge(virtual_mem.used),
        "剩余内存": format_merge(virtual_mem.available),
        "内存使用率": f"{virtual_mem.percent}%",
    }
    results["环境信息"] = {
        "PyTorch版本": torch.__version__,
        "cuDNN版本": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A",
        "Ultralytics_Version": ultralytics.__version__,
        "ONNX版本": _get_package_version("onnx"),
        "Numpy版本": _get_package_version("numpy"),
        "OpenCV版本": _get_package_version("opencv-python"),
        "Pillow版本": _get_package_version("Pillow"),
        "Torchvision版本": _get_package_version("torchvision"),
    }
    disk_info = psutil.disk_usage('/')
    results["磁盘信息"] = {
        "总空间": format_merge(disk_info.total),
        "已用空间": format_merge(disk_info.used),
        "剩余空间": format_merge(disk_info.free),
        "使用率": f"{disk_info.percent}%"
    }

    return results

def format_log_line(key: str, value: str, width: int = 20):

    display_width = sum(2 if 0x4e00 <= ord(char) <= 0x9fff else 1 for char in key)
    padding = width - display_width + len(key)
    return f"    {key:<{padding}}: {value}"

def log_device_info():

    device_info = get_device_info()
    logger.info("=".center(40, '='))
    logger.info("设备信息概览")
    logger.info("=".center(40, '='))
    for category, info in device_info.items():
        if category == "GPU详细列表":
            logger.info(f"{category}:")
            if type(info) != type([]):
                info = [info]
            for gpu_idx, gpu_detail in enumerate(info):
                if "未检测到CUDA可用GPU" in gpu_detail.get('数量', ""):
                    logger.info(f'  {gpu_detail.get("数量", "")}')
                    break
                logger.info(f"  --- GPU {gpu_idx} 详情 ---")
                for key, value in gpu_detail.items():
                    logger.info(format_log_line(key, value, width=25))
        else:
            logger.info(f"{category}:")
            for key, value in info.items():
                logger.info(format_log_line(key, value, width=20))
    logger.info("=".center(40, '='))

    return device_info
