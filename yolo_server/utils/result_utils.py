import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

def log_results(results, model_trainer=None) -> dict:

    def safe_float_conversion(value, default_val=np.nan):
        if value is None:
            return default_val
        try:
            return float(value)
        except (TypeError, ValueError):
            return default_val

    task = getattr(results, 'task', 'unknown_task')

    save_dir_from_results = getattr(results, 'save_dir', None)
    if save_dir_from_results:
        save_dir = str(save_dir_from_results)
    elif model_trainer and hasattr(model_trainer, 'save_dir'):
        save_dir = str(model_trainer.save_dir)
        logger.info(f"从 model.trainer.save_dir 获取到模型保存路径: {save_dir}")
    else:
        save_dir = 'N/A'
        logger.warning("未能从 results 或 model.trainer 获取到有效的模型保存路径。")

    fitness = safe_float_conversion(getattr(results, 'fitness', np.nan))
    names = getattr(results, 'names', {})
    maps = getattr(results, 'maps', np.array([]))
    speed = getattr(results, 'speed', {})
    preprocess_ms = safe_float_conversion(speed.get('preprocess'))
    inference_ms = safe_float_conversion(speed.get('inference'))
    loss_ms = safe_float_conversion(speed.get('loss'))
    postprocess_ms = safe_float_conversion(speed.get('postprocess'))

    if all(not np.isnan(v) for v in [preprocess_ms, inference_ms, loss_ms, postprocess_ms]):
        total_time_ms = preprocess_ms + inference_ms + loss_ms + postprocess_ms
    else:
        total_time_ms = np.nan

    metrics_dict = getattr(results, 'results_dict', {})

    result_data = {
        "task": task,
        "save_dir": save_dir,
        "timestamp": datetime.now().isoformat(),
        "speed_ms_per_image": {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "loss": loss_ms,
            "postprocess": postprocess_ms,
            "total_processing": total_time_ms
        },
        "overall_metrics": {
            "fitness": fitness
        },
        "class_mAP50-95": {}
    }

    for key, value in metrics_dict.items():
        result_data["overall_metrics"][key] = safe_float_conversion(value)

    if names and maps.size > 0:
        for idx, class_name in names.items():
            if idx < maps.size:
                result_data["class_mAP50-95"][class_name] = safe_float_conversion(maps[idx])
            else:
                logger.warning(f"类别 '{class_name}' ({idx}) 没有对应的mAP值。可能 maps 数组长度不足。")
    else:
        logger.info("未获取到类别名称或类别 mAP 数据。")

    logger.info('=' * 60)
    logger.info(f"YOLO Results Summary ({task.capitalize()} Task)")
    logger.info('=' * 60)
    logger.info(f"{'Task':<20}: {task}")
    logger.info(f"{'Save Directory':<20}: {save_dir}")
    logger.info(f"{'Timestamp':<20}: {result_data['timestamp']}")
    logger.info('-' * 40)
    logger.info("Processing Speed (ms/image)")
    logger.info('-' * 40)
    logger.info(f"{'Preprocess':<20}: {result_data['speed_ms_per_image'].get('preprocess', np.nan):.3f} ms")
    logger.info(f"{'Inference':<20}: {result_data['speed_ms_per_image'].get('inference', np.nan):.3f} ms")
    logger.info(f"{'Loss Calc':<20}: {result_data['speed_ms_per_image'].get('loss', np.nan):.3f} ms")
    logger.info(f"{'Postprocess':<20}: {result_data['speed_ms_per_image'].get('postprocess', np.nan):.3f} ms")
    logger.info(f"{'Total Per Image':<20}: {result_data['speed_ms_per_image'].get('total_processing', np.nan):.3f} ms")
    logger.info('-' * 40)
    logger.info('Overall Evaluation Metrics')
    logger.info('-' * 40)
    logger.info(f"{'Fitness Score':<20}: {result_data['overall_metrics'].get('fitness', np.nan):.4f}")

    if task == 'detect' or task == 'segment':
        logger.info(f"{'Precision(B)':<20}: {result_data['overall_metrics'].get('metrics/precision(B)', np.nan):.4f}")
        logger.info(f"{'Recall(B)':<20}: {result_data['overall_metrics'].get('metrics/recall(B)', np.nan):.4f}")
        logger.info(f"{'mAP50(B)':<20}: {result_data['overall_metrics'].get('metrics/mAP50(B)', np.nan):.4f}")
        logger.info(f"{'mAP50-95(B)':<20}: {result_data['overall_metrics'].get('metrics/mAP50-95(B)', np.nan):.4f}")

        if task == 'segment':
            logger.info("--- Mask Metrics ---")
            logger.info(
                f"{'Precision(M)':<20}: {result_data['overall_metrics'].get('metrics/precision(M)', np.nan):.4f}")
            logger.info(f"{'Recall(M)':<20}: {result_data['overall_metrics'].get('metrics/recall(M)', np.nan):.4f}")
            logger.info(f"{'mAP50(M)':<20}: {result_data['overall_metrics'].get('metrics/mAP50(M)', np.nan):.4f}")
            logger.info(f"{'mAP50-95(M)':<20}: {result_data['overall_metrics'].get('metrics/mAP50-95(M)', np.nan):.4f}")

    else:
        logger.info(f"当前任务类型 '{task}' 的详细评估指标未完全支持。")
        for key, value in result_data["overall_metrics"].items():
            if key not in ['fitness', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
                        'metrics/mAP50-95(B)',
                        'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']:
                logger.info(f"{key:<20}: {value:.4f}")

    logger.info('-' * 40)

    if result_data['class_mAP50-95']:
        logger.info("Class-wise mAP@0.5:0.95 (Box Metrics)")
        logger.info('-' * 40)
        valid_class_maps = {k: v for k, v in result_data['class_mAP50-95'].items() if not np.isnan(v)}
        if valid_class_maps:
            sorted_class_maps = sorted(valid_class_maps.items(), key=lambda item: item[1], reverse=True)
            for class_name, mAP_value in sorted_class_maps:
                logger.info(f"{class_name:<20}: {mAP_value:.4f}")
        else:
            logger.warning("所有类别 mAP 值均为 NaN，无法进行排序和打印。")
    else:
        logger.warning("未获取到类别级别的 mAP 数据。")
    logger.info('=' * 60)

    return result_data
