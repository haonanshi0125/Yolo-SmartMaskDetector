import yaml
from pathlib import Path
import logging
import random
from collections import defaultdict
from typing import List, Dict, Union, Tuple
import math

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"]
YOLO_SPLITS = ["train", "val", "test"]
DEFAULT_ENCODING = "utf-8"
DATA_YAML_ERROR_PREFIX = "FATAL ERROR: data.yaml"
VALIDATION_ERROR_PREFIX = "ERROR: "
WARN_PREFIX = "WARN: "
DEBUG_PREFIX = "DEBUG: "

from performance_utils import time_it

def _load_yaml_file(yaml_path: Path) -> Dict:

    if not yaml_path.exists():
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 文件不存在: {yaml_path}，请检查配置文件路径是否正确。")
        raise FileNotFoundError(f"data.yaml 文件不存在: {yaml_path}")
    try:
        with open(yaml_path, "r", encoding=DEFAULT_ENCODING) as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise yaml.YAMLError("YAML文件内容不是一个有效的字典结构。")
        return config
    except yaml.YAMLError as e:
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 读取或解析失败: {e}。请检查YAML文件格式。")
        raise yaml.YAMLError(f"读取data.yaml文件失败: {e}")
    except Exception as e:
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 读取时发生未知错误: {e}")
        raise e

def _validate_yaml_config_content(config: Dict) -> Tuple[List[str], int]:

    classes_names = config.get("names", [])
    nc = config.get("nc", 0)

    if not classes_names or not isinstance(classes_names, list) or not all(isinstance(name, str) for name in classes_names):
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 缺少 'names' 字段或其格式不正确（应为字符串列表）。")
        raise ValueError("data.yaml 缺少 'names' 字段或其格式不正确。")

    if not isinstance(nc, int) or nc <= 0:
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 缺少 'nc' 字段或其值无效（应为大于0的整数）。")
        raise ValueError("data.yaml 缺少 'nc' 字段或其值无效。")

    if len(classes_names) != nc:
        logger.critical(f"{DATA_YAML_ERROR_PREFIX} 中类别信息不一致。names长度: {len(classes_names)}, nc: {nc}。请检查配置文件。")
        raise ValueError("数据集类别数量与配置文件不一致。")

    logger.info(f"数据集类别数量与配置文件一致，类别数量为：{nc}，类别为：{classes_names}")
    return classes_names, nc

def _get_image_paths_in_directory(directory_path: Path) -> List[Path]:

    if not directory_path.exists():
        logger.error(f"{VALIDATION_ERROR_PREFIX} 图像路径不存在: {directory_path}。")
        return []

    all_imgs = []
    for ext in IMG_EXTENSIONS:
        all_imgs.extend(list(directory_path.glob(ext)))

    if not all_imgs:
        logger.error(f"{VALIDATION_ERROR_PREFIX} 图像目录 {directory_path} 不包含任何图像文件。")
    else:
        logger.info(f"图像目录 {directory_path} 包含 {len(all_imgs)} 张图像。")
    return all_imgs

def _read_label_file_lines(label_path: Path) -> Tuple[List[str], str]:

    try:
        with open(label_path, "r", encoding=DEFAULT_ENCODING) as f:
            lines = f.read().splitlines()
        return lines, ""
    except Exception as e:
        return [], f"读取标签文件失败: {label_path}，错误: {e}"

def _validate_single_label_content(lines: List[str], label_path: Path, nc: int, task_type: str) -> Tuple[bool, str]:

    if not lines:
        return True, ""

    for line_idx, line in enumerate(lines):
        parts = line.split(" ")

        is_format_correct = True
        error_detail = ""

        if task_type == "detection":
            if len(parts) != 5:
                error_detail = "不符合 YOLO 检测格式（期望5个浮点数：类别ID、中心点X、中心点Y、宽度、高度）"
                is_format_correct = False
        elif task_type == "segmentation":
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                error_detail = "不符合 YOLO 分割格式（期望至少7个值：类别ID、多边形X1、Y1、X2、Y2...；且类别ID后坐标对数量为偶数）"
                is_format_correct = False
        else:
            error_detail = f"未知任务类型 '{task_type}'，无法验证标签格式。"
            is_format_correct = False

        if not is_format_correct:
            return False, f"标签文件格式错误: {label_path}，第 {line_idx + 1} 行: '{line}' - {error_detail}"

        try:
            class_id = int(parts[0])
            if not (0 <= class_id < nc):
                return False, f"标签文件 {label_path} 内容错误: 类别ID {class_id} 超出 [0, {nc - 1}] 范围。第 {line_idx + 1} 行: '{line}'"

            coords = [float(x) for x in parts[1:]]
            if not all(0 <= x <= 1 for x in coords):
                return False, f"标签文件 {label_path} 内容错误，坐标 {coords} 超出 [0,1] 范围。第 {line_idx + 1} 行: '{line}'"
        except ValueError:
            return False, f"标签文件 {label_path} 包含无效值（非数字）: '{line}'"

    return True, ""

def _calculate_std_dev(data_list: List[float]) -> float:

    if len(data_list) < 2:
        return 0.0
    mean = sum(data_list) / len(data_list)
    variance = sum([(x - mean) ** 2 for x in data_list]) / (len(data_list) - 1)
    return math.sqrt(variance)

@time_it(name="数据集配置验证与分析", logger_instance=logger)
def verify_dataset_config(
    yaml_path: Path,
    mode: str = "SAMPLE",
    task_type: str = "detection",
    sample_ratio: float = 0.1,
    min_samples: int = 10
) -> Tuple[bool, List[Dict], Dict[str, List[Path]]]:

    invalid_samples_raw = []
    unique_invalid_samples = []
    all_image_paths_per_split = defaultdict(list)
    overall_validation_status = True

    try:
        config = _load_yaml_file(yaml_path)
        classes_names, nc = _validate_yaml_config_content(config)
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        logger.critical(f"致命错误：数据集配置加载或校验失败：{e}。无法继续验证。")
        return False, [], {}

    splits_to_check = [s for s in YOLO_SPLITS if s in config and config[s] is not None]
    if 'test' not in splits_to_check:
        logger.info(f"{WARN_PREFIX} data.yaml中未定义 'test' 路径或其路径值为None，跳过test划分的图像/标签验证和分析。")

    split_analysis_data: Dict[str, Dict[str, Union[int, defaultdict]]] = {
        split: {
            "total_images_analyzed": 0,
            "total_instances": 0,
            "class_counts": defaultdict(int),
            "images_per_class": defaultdict(int),
            "bbox_areas": defaultdict(list),
            "bbox_aspect_ratios": defaultdict(list),
        }
        for split in splits_to_check
    }

    for split_name in splits_to_check:
        split_path = Path(config[split_name]).resolve()
        logger.info(f"开始验证和分析 {split_name} 划分路径: {split_path}")

        current_split_all_images = _get_image_paths_in_directory(split_path)
        if not current_split_all_images:
            overall_validation_status = False
            continue

        all_image_paths_per_split[split_name] = current_split_all_images
        images_to_verify = current_split_all_images
        if mode == "SAMPLE":
            sample_size_actual = max(min_samples, int(len(current_split_all_images) * sample_ratio))
            images_to_verify = random.sample(
                current_split_all_images, min(sample_size_actual, len(current_split_all_images))
            )
            logger.info(f"{split_name} 划分验证模式为 SAMPLE，随机抽样 {len(images_to_verify)} 张图像进行详细验证和分析。")
        else:
            logger.info(f"{split_name} 划分验证模式为 FULL，正在验证和分析所有 {len(current_split_all_images)} 张图像。")

        split_analysis_data[split_name]["total_images_analyzed"] = len(images_to_verify)
        data_collector = split_analysis_data[split_name]

        for img_path in images_to_verify:
            label_dir = img_path.parent.parent / "labels"
            label_path = label_dir / (img_path.stem + ".txt")

            if not label_path.exists():
                error_msg = f"标签文件不存在: {label_path}，无法找到图像 {img_path.name} 的标签。"
                logger.error(f"{VALIDATION_ERROR_PREFIX}{error_msg}")
                overall_validation_status = False
                invalid_samples_raw.append({
                    "image_path": img_path.resolve(),
                    "label_path": label_path,
                    "error_message": error_msg
                })
                continue

            label_lines, read_error = _read_label_file_lines(label_path)
            if read_error:
                logger.error(f"{VALIDATION_ERROR_PREFIX}{read_error}")
                overall_validation_status = False
                invalid_samples_raw.append({
                    "image_path": img_path.resolve(),
                    "label_path": label_path,
                    "error_message": read_error
                })
                continue

            is_label_valid, validation_error = _validate_single_label_content(label_lines, label_path, nc, task_type)
            if not is_label_valid:
                logger.error(f"{VALIDATION_ERROR_PREFIX}标签内容验证失败：{validation_error}")
                overall_validation_status = False
                invalid_samples_raw.append({
                    "image_path": img_path.resolve(),
                    "label_path": label_path,
                    "error_message": validation_error
                })
                continue

            if not label_lines:
                continue

            classes_in_this_image = set()
            for line in label_lines:
                parts = line.split(" ")
                try:
                    class_id = int(parts[0])
                    data_collector["class_counts"][class_id] += 1
                    data_collector["total_instances"] += 1
                    classes_in_this_image.add(class_id)

                    if task_type == "detection" and len(parts) == 5:
                        bbox_w_norm = float(parts[3])
                        bbox_h_norm = float(parts[4])
                        if bbox_w_norm > 0 and bbox_h_norm > 0:
                            area_norm = bbox_w_norm * bbox_h_norm
                            data_collector["bbox_areas"][class_id].append(area_norm)
                            data_collector["bbox_aspect_ratios"][class_id].append(bbox_w_norm / bbox_h_norm)
                    elif task_type == "segmentation" and len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                        points = [float(p) for p in parts[1:]]
                        xs = [points[i] for i in range(0, len(points), 2)]
                        ys = [points[i] for i in range(1, len(points), 2)]
                        if xs and ys:
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)
                            bbox_w_norm = max_x - min_x
                            bbox_h_norm = max_y - min_y

                            if bbox_w_norm > 0 and bbox_h_norm > 0:
                                area_norm = bbox_w_norm * bbox_h_norm
                                data_collector["bbox_areas"][class_id].append(area_norm)
                                data_collector["bbox_aspect_ratios"][class_id].append(bbox_w_norm / bbox_h_norm)
                except (ValueError, IndexError):
                    logger.debug(f"{DEBUG_PREFIX} 标签文件 {label_path} 格式或数值错误，跳过此行统计。行内容: '{line}'")
                    continue

            for class_id in classes_in_this_image:
                data_collector["images_per_class"][class_id] += 1

    seen_invalid_data = set()
    for item in invalid_samples_raw:
        item_tuple = (str(item['image_path']), str(item['label_path']), item['error_message'])
        if item_tuple not in seen_invalid_data:
            unique_invalid_samples.append(item)
            seen_invalid_data.add(item_tuple)

    if unique_invalid_samples:
        logger.error(f"SUMMARY: 数据集基本结构或标签内容验证失败。发现 {len(unique_invalid_samples)} 个无效的图像-标签对。")
        logger.info("请检查上方标记为 'ERROR:' 的详细无效样本信息。")
        overall_validation_status = False
    else:
        logger.info("SUMMARY: 数据集基本结构和标签内容验证通过！")

    logger.info("=" * 100)
    logger.info("Dataset Class Distribution Analysis Results".center(100))
    logger.info("=" * 100)

    for split_name in splits_to_check:
        data = split_analysis_data[split_name]

        logger.info(f"\n-------------------- {split_name.upper()} 划分分析 --------------------")
        if data["total_images_analyzed"] == 0:
            logger.info(f"此 {split_name.upper()} 划分中没有可分析的图像文件。")
            continue

        logger.info(f"分析图像总数: {data['total_images_analyzed']}")
        logger.info(f"总实例数: {data['total_instances']}")

        if data["total_instances"] == 0:
            logger.info(f"此 {split_name.upper()} 划分中所有分析的图像均无标注实例。")
            logger.info("-" * 130)
            logger.info(
                f"{'Class ID':<8} {'Class Name':<20} {'Total Instances':>15} {'Instance %':>12} "
                f"{'Image Count':>12} {'Image %':>12} {'Avg Inst/Img':>12} {'Avg Bbox Area':>15} "
                f"{'StdDev Area':>12} {'Avg Aspect Ratio':>15} {'StdDev Aspect':>12}"
            )
            logger.info("-" * 130)
            continue

        max_class_name_len = max(len(name) for name in classes_names) if classes_names else len('Class Name')
        class_name_col_width = max(max_class_name_len, len('Class Name'))

        header = (
            f"{'Class ID':<8} "
            f"{'Class Name':<{class_name_col_width}} "
            f"{'Total Instances':>15} "
            f"{'Instance %':>12} "
            f"{'Image Count':>12} "
            f"{'Image %':>12} "
            f"{'Avg Inst/Img':>12} "
            f"{'Avg Bbox Area':>15} "
            f"{'StdDev Area':>12} "
            f"{'Avg Aspect Ratio':>15} "
            f"{'StdDev Aspect':>12}"
        )

        separator = "-" * len(header)
        logger.info(separator)
        logger.info(header)
        logger.info(separator)

        sorted_class_ids = sorted(data["class_counts"].keys())
        for class_id in sorted_class_ids:
            class_name = classes_names[class_id] if 0 <= class_id < nc else "未知类别"
            instance_count = data["class_counts"][class_id]
            instance_percentage = (instance_count / data["total_instances"]) * 100 if data["total_instances"] > 0 else 0.0

            image_coverage_count = data["images_per_class"][class_id]
            image_coverage_percentage = (image_coverage_count / data["total_images_analyzed"]) * 100 if data["total_images_analyzed"] > 0 else 0.0
            avg_instances_per_image = instance_count / image_coverage_count if image_coverage_count > 0 else 0.0

            bbox_areas_list: List[float] = data["bbox_areas"][class_id]
            avg_bbox_area = sum(bbox_areas_list) / len(bbox_areas_list) if bbox_areas_list else 0.0
            std_dev_area = _calculate_std_dev(bbox_areas_list)

            bbox_aspect_ratios_list = data["bbox_aspect_ratios"][class_id]
            avg_bbox_aspect_ratio = sum(bbox_aspect_ratios_list) / len(bbox_aspect_ratios_list) if bbox_aspect_ratios_list else 0.0
            std_dev_aspect = _calculate_std_dev(bbox_aspect_ratios_list)

            row = (
                f"{str(class_id):<8} "
                f"{class_name:<{class_name_col_width}} "
                f"{instance_count:>15} "
                f"{instance_percentage:>12.2f} "
                f"{image_coverage_count:>12} "
                f"{image_coverage_percentage:>12.2f} "
                f"{avg_instances_per_image:>12.2f} "
                f"{avg_bbox_area:>15.4f} "
                f"{std_dev_area:>12.4f} "
                f"{avg_bbox_aspect_ratio:>15.2f} "
                f"{std_dev_aspect:>12.2f}"
            )
            logger.info(row)
        logger.info(separator)

    logger.info("=" * 100)
    logger.info("数据集分析完成！".center(100))
    logger.info("=" * 100)
    logger.info("SUMMARY: 数据集验证与分析已完成！")

    return overall_validation_status, unique_invalid_samples, all_image_paths_per_split

@time_it(name="数据集分割验证", logger_instance=logger)
def verify_split_uniqueness(yaml_path: Path) -> bool:

    logger.info("开始数据集划分唯一性验证（检查train, val, test之间是否存在重复图像）。")
    try:
        config = _load_yaml_file(yaml_path)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.critical(f"致命错误：无法加载 data.yaml，无法执行划分唯一性验证：{e}。")
        return False

    split_image_stems: Dict[str, set] = {}
    overall_uniqueness_status = True

    splits_to_check = [s for s in YOLO_SPLITS if s in config and config[s] is not None]
    if 'test' not in splits_to_check and 'test' in YOLO_SPLITS:
        logger.info(f"{WARN_PREFIX} data.yaml 中未定义 'test' 路径或其路径值为None，跳过test划分唯一性检查。")

    for split_name in splits_to_check:
        split_path = Path(config[split_name]).resolve()

        if not split_path.exists():
            logger.error(f"{VALIDATION_ERROR_PREFIX}'{split_name}' 图像路径不存在: {split_path}，无法执行唯一性验证。")
            overall_uniqueness_status = False
            continue

        img_stems = set()
        for ext in IMG_EXTENSIONS:
            for img_file in split_path.glob(ext):
                img_stems.add(img_file.stem)

        split_image_stems[split_name] = img_stems
        logger.info(f"'{split_name}' 划分包含 {len(img_stems)} 张唯一图像。")

    combinations = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in combinations:
        if s1 in split_image_stems and s2 in split_image_stems:
            common_stems = split_image_stems[s1].intersection(split_image_stems[s2])
            if common_stems:
                logger.error(
                    f"{VALIDATION_ERROR_PREFIX}在 {s1} 和 {s2} 之间发现重复图像！数量: {len(common_stems)}。示例: {list(common_stems)[:5]}"
                )
                overall_uniqueness_status = False
            else:
                logger.info(f"{s1} 和 {s2} 之间未发现重复图像。")

    if overall_uniqueness_status:
        logger.info("SUMMARY: 数据集划分唯一性验证通过：各子集之间未发现重复图像。")
    else:
        logger.error("SUMMARY: 数据集划分唯一性验证失败，存在重复图像。请检查上方日志中标记为 'ERROR:' 的消息。")

    return overall_uniqueness_status

def delete_invalid_files(invalid_data_list: List[Dict]):

    logger.info("开始删除无效的图像和标签文件...")
    deleted_image_count = 0
    deleted_label_count = 0

    if not invalid_data_list:
        logger.info("没有需要删除的无效文件。")
        return

    for item in invalid_data_list:
        img_path = item['image_path']
        label_path = item['label_path']
        error_msg = item.get('error_message', '未知错误')
        logger.debug(f"{DEBUG_PREFIX}尝试删除图像: {img_path} 及其标签: {label_path}，原因：'{error_msg}'。")

        try:
            if img_path.exists():
                img_path.unlink() # 删除文件
                logger.info(f"SUCCESS: 成功删除图像文件: {img_path}")
                deleted_image_count += 1
            else:
                logger.warning(f"{WARN_PREFIX}图像文件不存在，跳过删除: {img_path}")

            if label_path.exists():
                label_path.unlink() # 删除文件
                logger.info(f"SUCCESS: 成功删除标签文件: {label_path}")
                deleted_label_count += 1
            else:
                logger.warning(f"{WARN_PREFIX}标签文件不存在，跳过删除: {label_path}")

        except OSError as e:
            logger.error(f"{VALIDATION_ERROR_PREFIX}删除文件失败: {e} - 无法删除图像: {img_path} 或标签: {label_path}")

    logger.info(f"SUMMARY: 删除操作完成。共删除了 {deleted_image_count} 个图像文件和 {deleted_label_count} 个标签文件。")

if __name__ == "__main__":
    from logging_utils import setup_logging
    logger = setup_logging(base_path=Path('./logs'), log_type='test')
    verify_dataset_config(yaml_path=Path('D:/PycharmProjects/SafeYolo/yolo_server/configs/data.yaml'), mode='FULL')
