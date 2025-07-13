import argparse
import sys
from pathlib import Path
import yaml
import shutil
import logging
from typing import List, Union, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0,str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1,str(utils_path))

from performance_utils import time_it
from logging_utils import setup_logging
from paths import (YOLO_SERVER_ROOT,
                RAW_IMAGES_DIR,
                ORIGINAL_ANNOTATIONS_DIR,
                YOLO_STAGED_LABELS_DIR,
                CONFIGS_DIR,
                LOGS_DIR,
                DATA_DIR,
                )
from data_converters_utils import convert_data_to_yolo

logger = logging.getLogger("YOLO DataConversion")

class YOLODatasetProcessor:

    def __init__(self, train_rate: float = 0.8, val_rate: float = 0.1,
                annotation_format: str = "pascal_voc",
                final_classes_order: Union[List[str], None] = None,
                coco_task: str = "detection",
                coco_cls91to80: bool = False):

        self.project_root_path = YOLO_SERVER_ROOT
        self.raw_images_path = RAW_IMAGES_DIR
        self.yolo_staged_labels_path = YOLO_STAGED_LABELS_DIR
        self.output_data_path = DATA_DIR
        self.config_path = CONFIGS_DIR

        self.annotation_format = annotation_format
        self.coco_task = coco_task
        self.coco_cls91to80 = coco_cls91to80
        self.classes: List[str] = final_classes_order
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.test_rate = 1 - train_rate - val_rate

        if not (0.0 <= self.train_rate <= 1.0 and
                0.0 <= self.val_rate <= 1.0 and
                0.0 <= self.test_rate <= 1.0 and
                self.train_rate + self.val_rate + self.test_rate - 1.0 <= 1e-6):

            logger.error("训练集、验证集和测试集的比例之和必须等于1.0或者配置比例无效，请检查配置")
            raise ValueError("训练集、验证集和测试集的比例之和必须等于1.0或者配置比例无效，请检查配置")

        self.config_path.mkdir(parents=True, exist_ok=True)
        self.output_dirs: Dict[str, Dict[str, Path]] = {
            "train": {"images": self.output_data_path / "train" / "images",
                    "labels": self.output_data_path / "train" / "labels"
                    },
            "val": {"images": self.output_data_path / "val" / "images",
                    "labels": self.output_data_path / "val" / "labels"
                    },
            "test": {"images": self.output_data_path / "test" / "images",
                    "labels": self.output_data_path / "test" / "labels"
                    }
        }

    def _check_staged_data_existence(self):

        if not self.yolo_staged_labels_path.exists() or not any(self.yolo_staged_labels_path.glob("*.txt")):
            logger.error(f"YOLO标签暂存目录：'{self.yolo_staged_labels_path}'不存在或者标签文件为空，请检查配置")
            raise FileNotFoundError("暂存区中不存在标签文件，请检查配置")

        if not self.raw_images_path.exists() or not any(self.raw_images_path.glob("*.jpg")):
            logger.error(f"原始图像暂存目录：'{self.raw_images_path}'不存在或者图像文件为空，请检查配置")
            raise FileNotFoundError("暂存区中不存在图像文件，请检查配置")

        logger.info(f"原始数据暂存区通过检查，图像位于：{self.raw_images_path.relative_to(YOLO_SERVER_ROOT)},"
                    f"YOLO标签位于：{self.yolo_staged_labels_path.relative_to(YOLO_SERVER_ROOT)}")

    def _ensure_output_dirs_exist(self):
        for split_info in self.output_dirs.values():
            for dir_path in split_info.values():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"已经创建或确认目录存在：{dir_path.relative_to(YOLO_SERVER_ROOT)}")
        logger.info("数据集划分目录结构已确认")

    def _find_matching_files(self) -> List[Tuple[Path, Path]]:

        txt_files = list(self.yolo_staged_labels_path.glob("*.txt"))
        if not txt_files:
            logger.warning(f"在暂存区 '{self.yolo_staged_labels_path.relative_to(YOLO_SERVER_ROOT)}' "
                        f"中未找到任何YOLO标签文件")
            return []

        matching_pairs: List[Tuple[Path, Path]] = []
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        for txt_file in txt_files:
            found_image = False
            for ext in img_extensions:
                img_name_stem = txt_file.stem
                image_path = self.raw_images_path / (img_name_stem + ext)
                if image_path.exists():
                    matching_pairs.append((image_path, txt_file))
                    found_image = True
                    break
            if not found_image:
                logger.warning(f"未在'{self.raw_images_path.relative_to(YOLO_SERVER_ROOT)}'"
                            f"中找到匹配的图像文件'{txt_file.name}',跳过此文件")
        if not matching_pairs:
            logger.warning(f"在暂存区 '{self.yolo_staged_labels_path.relative_to(YOLO_SERVER_ROOT)}' "
                        f"中未找到任何匹配的图像和标签文件")
        else:
            logger.info(f"找到 {len(matching_pairs)} 个匹配的图像和标签文件")
        return matching_pairs

    def _split_and_process_data(self, matching_pairs: List[Tuple[Path, Path]]):

        if not matching_pairs:
            logger.warning("没有数据集可供划分，请检查配置")
            return

        label_files = [pair[1] for pair in matching_pairs]
        image_paths = [pair[0] for pair in matching_pairs]

        if len(matching_pairs) < 3:
            logger.warning(f"数据集数量过少 {len(matching_pairs)}，无法进行有效分割，将所有数据分配给训练集")
            self._process_single_split(label_files, image_paths, "train")
            return

        train_labels, temp_labels,train_images, temp_images = train_test_split(
            label_files, image_paths, train_size=self.train_rate,
            random_state=42, shuffle=True)
        val_labels, test_labels, val_images, test_images = [], [], [], []

        if temp_labels:
            remaining_rate = self.val_rate + self.test_rate
            if remaining_rate == 0 or len(temp_labels) < 2:
                val_labels,val_images = temp_labels, temp_images
                logger.warning(f"第一次划分之后，剩余数据集数量过少 {len(temp_labels)}或者剩余比例为0，"
                            f"无法进行有效分割，将所有数据分配给验证集")
            else:
                val_ration_in_temp = self.val_rate / remaining_rate
                if abs(val_ration_in_temp) < 1e-6:
                    test_labels, test_images = temp_labels, temp_images
                    logger.info("验证集比例为0，所有剩余数据分配给测试集")
                elif abs(val_ration_in_temp - 1) < 1e-6:
                    val_labels, val_images = temp_labels, temp_images
                    logger.info("测试集比例为0，所有剩余数据分配给验证集")
                else:
                    val_labels, test_labels, val_images, test_images = train_test_split(
                        temp_labels, temp_images, train_size=val_ration_in_temp,
                        random_state=42, shuffle=True)

        logger.info(f"数据集划分完成，具体结果如下")
        logger.info(f"训练集：{len(train_labels)} 个标签文件，{len(train_images)} 个图像文件")
        logger.info(f"验证集：{len(val_labels)} 个标签文件，{len(val_images)} 个图像文件")
        logger.info(f"测试集：{len(test_labels)} 个标签文件，{len(test_images)} 个图像文件")

        self._process_single_split(train_labels, train_images, "train")
        self._process_single_split(val_labels, val_images, "val")
        self._process_single_split(test_labels, test_images, "test")

    def _process_single_split(self, label_files: List[Path], image_paths: List[Path], split_name: str):

        logger.info(f"正在处理 {split_name} 数据集,共{len(label_files)} 个标签文件，{len(image_paths)} 个图像文件")
        target_img_dir = self.output_dirs[split_name]["images"]
        target_label_dir = self.output_dirs[split_name]["labels"]

        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        copied_images_count = 0
        failed_images_count = 0

        for image_path in image_paths:
            new_image_path = target_img_dir / image_path.name
            try:
                shutil.copy(image_path, new_image_path)
                copied_images_count += 1
                logger.debug(f"复制图像文件 {image_path.name} 到 '{new_image_path.relative_to(YOLO_SERVER_ROOT)}' 成功")
            except Exception as e:
                failed_images_count += 1
                logger.error(f"复制图像文件 {image_path.name} 失败，错误信息为 {e}")
        logger.info(f"复制图像文件完成，共成功复制 {copied_images_count} 个文件，失败 {failed_images_count} 个文件")

        copied_labels_count = 0
        failed_labels_count = 0
        for label_file_path in label_files:
            new_label_file = target_label_dir / label_file_path.name
            try:
                shutil.copy(label_file_path, new_label_file)
                copied_labels_count += 1
                logger.debug(f"复制标签文件 {label_file_path.name} 到 '{new_label_file.relative_to(YOLO_SERVER_ROOT)}' 成功")
            except Exception as e:
                failed_labels_count += 1
                logger.error(f"复制标签文件 {label_file_path.name} 失败，错误信息为 {e}")
        logger.info(f"复制标签文件完成，共成功复制 {copied_labels_count} 个文件，失败 {failed_labels_count} 个文件")

    def _generate_data_yaml(self):

        abs_data_path = self.output_data_path.absolute()
        train_images_abs_path = (self.output_dirs["train"]["images"]).resolve()
        val_images_abs_path = (self.output_dirs["val"]["images"]).resolve()
        test_images_abs_path = (self.output_dirs["test"]["images"]).resolve()

        data_yaml_content: Dict[str, Any] = {
            "path": str(abs_data_path),
            "train": str(train_images_abs_path),
            "val": str(val_images_abs_path),
            "test": str(test_images_abs_path),
            "nc": len(self.classes),
            "names": self.classes
        }
        yaml_path = self.config_path / "data.yaml"
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data_yaml_content, f, default_flow_style=None,sort_keys=False,allow_unicode=True)
            logger.info(f"已生成数据配置文件 {yaml_path.relative_to(YOLO_SERVER_ROOT)}")
            logger.info(f"数据配置文件内容如下：\n {yaml.dump(data_yaml_content, default_flow_style=None, sort_keys=False,allow_unicode=True)}")
        except Exception as e:
            logger.error(f"生成数据配置文件 {yaml_path.relative_to(YOLO_SERVER_ROOT)} 失败，错误信息为 {e}")

    @time_it(iterations=1, name='数据准备与划分', logger_instance=logger)
    def process_data(self,source_data_root_dir: Path = ORIGINAL_ANNOTATIONS_DIR,):

        logger.info(f"开始进行数据处理流程".center(50, "="))
        try:
            logger.info(f"数据处理流程开始，处理原始标注数据 {self.annotation_format.upper()} 格式，"
                        f"数据位于：{source_data_root_dir}")

            if self.annotation_format != 'yolo':
                if self.yolo_staged_labels_path.exists():
                    shutil.rmtree(self.yolo_staged_labels_path)
                    logger.info(f"已清理YOLO格式的临时标签文件，"
                                f"路径为：{self.yolo_staged_labels_path.relative_to(YOLO_SERVER_ROOT)}")
                self.yolo_staged_labels_path.mkdir(parents=True, exist_ok=True)
            if self.annotation_format == "yolo":
                if not self.classes:
                    logger.error(f"当 annotation_format 为 yolo格式，必须要配置 final_classes_order 参数，")
                    return

                self.yolo_staged_labels_path = ORIGINAL_ANNOTATIONS_DIR
                logger.info("检测到原生YOLO格式数据，将直接使用原始数据，YOLO标签暂存目录已经指向原始标签目录")
                if not any(self.yolo_staged_labels_path.glob("*.txt")):
                    logger.critical(f"未找到YOLO格式的标签文件，请检查数据集，"
                                    f"路径为：{self.yolo_staged_labels_path.relative_to(YOLO_SERVER_ROOT)}")
                    return
            elif self.annotation_format in ["coco", "pascal_voc"]:

                if not RAW_IMAGES_DIR.exists() or not any(RAW_IMAGES_DIR.iterdir()):
                    logger.critical(f"未找到原始图像数据，请检查数据集，"
                                    f"路径为：{RAW_IMAGES_DIR.relative_to(YOLO_SERVER_ROOT)}")
                    return
                if not ORIGINAL_ANNOTATIONS_DIR.exists() or not any(ORIGINAL_ANNOTATIONS_DIR.iterdir()):
                    logger.critical(f"未找到原始标注数据，请检查数据集，"
                                    f"路径为：{ORIGINAL_ANNOTATIONS_DIR.relative_to(YOLO_SERVER_ROOT)}")
                    return

                conversion_input_dir = source_data_root_dir
                self.classes = convert_data_to_yolo(conversion_input_dir,
                        self.annotation_format,
                        final_classes_order=self.classes if self.annotation_format == "pascal_voc" else None,
                        coco_task=self.coco_task,
                        coco_cls91to80=self.coco_cls91to80)

                if not self.classes:
                    logger.critical(f"未找到任何类别，请检查数据集，"
                                    f"路径为：{conversion_input_dir.relative_to(YOLO_SERVER_ROOT)}")
                    return
                logger.info(f"{self.annotation_format.upper()}格式数据转换成YOLO格式成功"
                            f"已生成 {len(self.classes)} 个类别,具体内容为：{self.classes}")
            else:
                logger.critical(f"暂不支持 {self.annotation_format.upper()} 格式数据转换，请选择正确的格式")
                return

            self._check_staged_data_existence()

            matching_paris = self._find_matching_files()
            if not matching_paris:
                logger.critical(f"未找到匹配的文件，请检查数据集，"
                                f"路径为：{self.yolo_staged_labels_path.relative_to(YOLO_SERVER_ROOT)}")
                return

            self._split_and_process_data(matching_paris)
            self._generate_data_yaml()

        except Exception as e:
            logger.critical(f"数据处理流程失败，错误信息为 {e}",exc_info=True)
        finally:
            logger.info(f"数据处理流程结束".center(50, "="))

def _clean_and_initialize_dirs(processor_instance:YOLODatasetProcessor):
    logger.info(f"开始清理旧数据集目录".center(50, "="))
    for split_name, split_info in processor_instance.output_dirs.items():
        for dir_type,dir_path in split_info.items():
            if dir_path.exists():
                shutil.rmtree(dir_path,ignore_errors=True)
                logger.info(f"已清理旧数据集'{split_name}','{dir_type} "
                            f"目录: {dir_path.relative_to(YOLO_SERVER_ROOT)}'")
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"已初始化新数据集 '{split_name} ','{dir_type}'"
                        f"目录: {dir_path.relative_to(YOLO_SERVER_ROOT)}'")
    data_yaml_file = CONFIGS_DIR / "data.yaml"
    if data_yaml_file.exists():
        data_yaml_file.unlink()
        logger.info(f"已清理旧数据集配置文件: {data_yaml_file.relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"数据集目录清理完毕".center(50, "="))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 数据集处理工具",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--format", type=str,
                        default="pascal_voc",
                        choices=["coco", "pascal_voc", "yolo"],
                        help="支持的数据集标注格式，coco, pascal_voc, yolo")
    parser.add_argument("--train_rate", type=float, default=0.8, help="训练集占比,默认0.8")
    parser.add_argument("--valid_rate", type=float, default=0.1, help="验证集占比,默认0.1")
    parser.add_argument("--classes",type=str,
                        nargs="+",
                        default=None,
                        help="类别名称列表，以空格分开，例如：--classes class1 class2 class3 \n"
                            "当 --format 为 yolo 时, 必须提供该参数"
                            "当 --format 为 coco 时， 此参数会被忽略"
                            "当 --format 为 pascal_voc 时，可选提供，不指定则使用自动模式")
    parser.add_argument("--coco_task", type=str,
                        default="detection",
                        choices=["detection", "segmentation"],
                        help="COCO任务类型，可选：detection, segmentation")
    parser.add_argument("--coco_cls91to80",default=False,
                        action="store_true", help="将COCO 91类映射 80类")
    args = parser.parse_args()

    logger = setup_logging(base_path=LOGS_DIR, log_type="yolo_trans", model_name=None)

    processor = YOLODatasetProcessor(train_rate=args.train_rate,
                                    val_rate=args.valid_rate,
                                    annotation_format=args.format,
                                    final_classes_order=args.classes,
                                    coco_task=args.coco_task,
                                    coco_cls91to80=args.coco_cls91to80)

    _clean_and_initialize_dirs(processor)
    processor.process_data()

    logger.info("所有数据处理流程完成，请检查以下路径文件")
    logger.info(f"训练集图像目录：{processor.output_dirs['train']['images'].relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"训练集标注文件：{processor.output_dirs['train']['labels'].relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"验证集图像目录：{processor.output_dirs['val']['images'].relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"验证集标注文件：{processor.output_dirs['val']['labels'].relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"测试集图像目录：{processor.output_dirs['test']['images'].relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"测试集标注文件：{processor.output_dirs['test']['labels'].relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"数据集配置文件：{processor.config_path.relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"详细的日志文件位于 {LOGS_DIR.relative_to(YOLO_SERVER_ROOT)}")
    logger.info(f"接下来请执行数据验证脚本 yolo_validate.py 以验证数据转换是否正确")
