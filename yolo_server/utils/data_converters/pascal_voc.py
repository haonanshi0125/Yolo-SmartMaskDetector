import logging
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Union, Set

logger = logging.getLogger(__name__)

def _parse_xml_annotation(xml_path: Path, classes: List[str]) -> List[str]:

    yolo_labels = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_elem = root.find("size")

        if size_elem is None:
            logger.error(f"XML文件 '{xml_path.name}'格式错误: 缺少size元素,无法提取图片尺寸信息，跳过")
            return []

        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)

        if width <= 0 or height <= 0:
            logger.error(f"XML文件 '{xml_path.name}'格式错误: 图片尺寸信息错误 (W：{width}, H: {height})，跳过")
            return []

        for obj in root.iter("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                logger.warning(f"XML文件 '{xml_path.name}'格式错误: 缺少name元素,跳过")
                continue
            name = name_elem.text.strip()

            if name not in classes:
                continue
            class_id = classes.index(name)
            xml_box = obj.find("bndbox")
            if xml_box is None:
                logger.error(f"XML文件 '{xml_path.name}'格式错误:对象 '{name}' 缺少bndbox元素,跳过")
                continue
            try:
                xmin = int(xml_box.find("xmin").text)
                ymin = int(xml_box.find("ymin").text)
                xmax = int(xml_box.find("xmax").text)
                ymax = int(xml_box.find("ymax").text)
            except (AttributeError, ValueError) as e:
                logger.error(f"XML文件 '{xml_path.name}'格式错误:对象 '{name}'的边界框解析失败,跳过: {e}")
                continue

            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                logger.warning(f"XML文件 '{xml_path.name}'格式错误:对象 '{name}'的边界框超出图片尺寸范围，跳过")
                continue

            center_x = (xmin + xmax) / 2 / width
            center_y = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            center_x = max(0.0, min(1.0, center_x))
            center_y = max(0.0, min(1.0, center_y))
            box_width = max(0.0, min(1.0, box_width))
            box_height = max(0.0, min(1.0, box_height))

            yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        return yolo_labels
    except FileNotFoundError:
        logger.error(f"XML文件 '{xml_path.name}'不存在，跳过")
    except ET.ParseError as e:
        logger.error(f"XML文件 '{xml_path.name}'解析失败: {e}，跳过")
    except Exception as e:
        logger.error(f"XML文件 '{xml_path.name}'处理失败: {e}，跳过")
    return []

def convert_pascal_voc_to_yolo(
        xml_input_dir: Path,
        output_yolo_txt_dir: Path,
        target_classes_for_yolo: Union[List[str], None] = None) -> List[str]:

    logger.info(f"开始将Pascal VOC XML文件从 '{xml_input_dir}' "
                f"转换为Yolo格式文件 '{output_yolo_txt_dir}'".center(50, "="))

    if not xml_input_dir.exists():
        logger.error(f"输入目录 '{xml_input_dir}' 不存在")
        raise FileNotFoundError(f"输入目录 '{xml_input_dir}' 不存在")

    xml_files_found = list(xml_input_dir.glob("*.xml"))
    if not xml_files_found:
        logger.error(f"输入目录 '{xml_input_dir}' 中不存在XML文件")
        raise []

    if target_classes_for_yolo is not None:
        classes = target_classes_for_yolo
        logger.info(f"Pascal VOC转换模式为：手动模式，已指定目标类别为:"
                    f" {classes}".center(50, "="))
    else:
        unique_classes: Set[str] = set()
        logger.info(f"Pascal VOC转换模式为：自动模式，"
                    f"开始扫描XML文件以获取所有类别信息".center(50, "="))
        for xml_file in xml_files_found:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.iter("object"):
                    name_elem = obj.find("name")
                    if name_elem is not None and name_elem.text:
                        unique_classes.add(name_elem.text.strip())
            except ET.ParseError as e:
                logger.warning(f"扫描XML文件 '{xml_file.name}'解析失败: {e}")
            except Exception as e:
                logger.warning(f"扫描XML文件 '{xml_file.name}'处理失败: {e}")
        classes = sorted(list(unique_classes))
        if not classes:
            logger.error(f"从XML文件：{xml_input_dir}中未找到任何类别信息，请检查XML文件")
            raise []
        logger.info(f"Pascal Voc模式转换，自动模式，"
                    f"已获取所有类别信息: {classes}".center(50, "="))

    output_yolo_txt_dir.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    for xml_file in xml_files_found:
        yolo_labels = _parse_xml_annotation(xml_file, classes)
        if yolo_labels:
            txt_file_path = output_yolo_txt_dir / (xml_file.stem + ".txt")
            try:
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    for label in yolo_labels:
                        f.write(label + "\n")
                converted_count += 1
            except Exception as e:
                logger.error(f"写入Yolo格式文件 '{txt_file_path.name}' 失败: {e}")
                continue
        else:
            logger.warning(f"XML文件 '{xml_file.name}' 未生成有效的Yolo标签，可能为无类别目标或解析失败")
    logger.info(f"从'{xml_input_dir}'转换完成，共转换 {converted_count} "
                f"个XML文件为Yolo格式文件，保存在 '{output_yolo_txt_dir}'")
    return classes
