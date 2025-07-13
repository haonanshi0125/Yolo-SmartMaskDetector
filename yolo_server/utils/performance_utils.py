import logging
import time
from functools import wraps

_default_logger = logging.getLogger(__name__)

def time_it(iterations: int = 1, name: str = None, logger_instance: logging.Logger = None):
    """函数计时装饰器，可重复执行多次并计算平均耗时。"""
    logger_to_use = logger_instance if logger_instance else _default_logger

    def _format_time_auto_unit(total_seconds: float) -> str:
        """根据耗时选择合适的单位格式化输出。"""
        if total_seconds < 0.000001:
            return f'{total_seconds * 1_000_000:.2f} 纳秒'
        elif total_seconds < 0.001:
            return f'{total_seconds * 1_000:.2f} 微秒'
        elif total_seconds < 1.0:
            return f'{total_seconds * 1000:.2f} 毫秒'
        elif total_seconds < 60.0:
            return f'{total_seconds:.2f} 秒'
        elif total_seconds < 3600.0:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f'{minutes:.0f} 分钟 {seconds:.2f} 秒'
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = (total_seconds % 3600) % 60
            return f'{hours:.0f} 小时 {minutes:.0f} 分钟 {seconds:.2f} 秒'

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_display_name = name or func.__name__
            total_elapsed_time = 0.0
            result = None

            for _ in range(iterations):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                total_elapsed_time += end_time - start_time

            avg_elapsed_time = total_elapsed_time / iterations
            formatted_avg_time = _format_time_auto_unit(avg_elapsed_time)

            if iterations == 1:
                logger_to_use.info(f"性能报告: '{func_display_name}' 平均耗时: {formatted_avg_time}")
            else:
                logger_to_use.info(f"性能报告: '{func_display_name}' 执行了 {iterations} 次，平均耗时: {formatted_avg_time}")

            return result
        return wrapper
    return decorator
