from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

text_size_cache = OrderedDict()
MAX_CACHE_SIZE = 500

def preload_cache(font_path, font_size, label_mapping):

    global text_size_cache
    text_size_cache.clear()

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"警告：无法加载字体文件 '{font_path}'。跳过字体大小 {font_size} 的预缓存。")
        return text_size_cache

    for label_val in list(label_mapping.values()) + list(label_mapping.keys()):
        text_to_measure = f"{label_val} 80.0%"
        cache_key = label_val
        temp_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_image)

        if 'font' in locals() and font is not None:
            bbox = draw.textbbox((0, 0), text_to_measure, font=font)
            text_size_cache[cache_key] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        else:
            print(f"警告：字体未成功加载，无法预缓存 '{label_val}'。")
            break

    return text_size_cache

def get_text_size(text, font_obj, max_cache_size=500):

    parts = text.split(" ")
    if len(parts) > 1 and parts[-1].endswith('%'):
        label_part = " ".join(parts[:-1])
    else:
        label_part = text

    cache_key = label_part
    if cache_key in text_size_cache:
        return text_size_cache[cache_key]

    temp_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_image)
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    text_size_cache[cache_key] = (width, height)
    if len(text_size_cache) > max_cache_size:
        text_size_cache.popitem(last=False)

    return (width, height)

def calculate_beautify_params(
        current_image_height,
        current_image_width,
        base_font_size=26,
        base_line_width=4,
        base_label_padding_x=10,
        base_label_padding_y=10,
        base_radius=8,
        ref_dim_for_scaling=720,
        font_path="D:/PycharmProjects/SafeYolo/yolo_server/utils/LXGWWenKai-Bold.ttf",
        text_color_bgr=(0, 0, 0),
        use_chinese_mapping=True,
        label_mapping=None,
        color_mapping=None
):

    if label_mapping is None:
        label_mapping = {}
    if color_mapping is None:
        color_mapping = {}

    current_short_dim = min(current_image_height, current_image_width)
    if ref_dim_for_scaling == 0:
        scale_factor = 1.0
        logger.warning("ref_dim_for_scaling 为0，缩放因子将设置为1.0。")
    else:
        scale_factor = current_short_dim / ref_dim_for_scaling

    font_size_adjusted = max(10, int(base_font_size * scale_factor))
    line_width_adjusted = max(1, int(base_line_width * scale_factor))
    label_padding_x_adjusted = max(5, int(base_label_padding_x * scale_factor))
    label_padding_y_adjusted = max(5, int(base_label_padding_y * scale_factor))
    radius_adjusted = max(3, int(base_radius * scale_factor))

    cache_dict = preload_cache(font_path, font_size=font_size_adjusted, label_mapping=label_mapping)
    logger.info(cache_dict)

    return {
        "font_path": font_path,
        "font_size": font_size_adjusted,
        "line_width": line_width_adjusted,
        "label_padding_x": label_padding_x_adjusted,
        "label_padding_y": label_padding_y_adjusted,
        "radius": radius_adjusted,
        "text_color_bgr": text_color_bgr,
        "use_chinese_mapping": use_chinese_mapping,
        "label_mapping": label_mapping,
        "color_mapping": color_mapping,
    }

def draw_filled_rounded_rect(img, pt1, pt2, color, radius,
                             top_left_round=True, top_right_round=True,
                             bottom_left_round=True, bottom_right_round=True):

    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    if top_left_round:
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x1 + radius, y1 + radius), color, -1)

    if top_right_round:
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x2 - radius, y1), (x2, y1 + radius), color, -1)

    if bottom_left_round:
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1, y2 - radius), (x1 + radius, y2), color, -1)

    if bottom_right_round:
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x2 - radius, y2 - radius), (x2, y2), color, -1)

def draw_bordered_rounded_rect(img, pt1, pt2, color, line_width, radius,
                               top_left_round=True, top_right_round=True,
                               bottom_left_round=True, bottom_right_round=True):

    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, line_width)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, line_width)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, line_width)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, line_width)

    if top_left_round:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, line_width)
    else:
        cv2.line(img, (x1, y1 + radius), (x1, y1), color, line_width)
        cv2.line(img, (x1, y1), (x1 + radius, y1), color, line_width)

    if top_right_round:
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, line_width)
    else:
        cv2.line(img, (x2, y1 + radius), (x2, y1), color, line_width)
        cv2.line(img, (x2 - radius, y1), (x2, y1), color, line_width)

    if bottom_left_round:
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, line_width)
    else:
        cv2.line(img, (x1, y2 - radius), (x1, y2), color, line_width)
        cv2.line(img, (x1, y2), (x1 + radius, y2), color, line_width)

    if bottom_right_round:
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, line_width)
    else:
        cv2.line(img, (x2, y2 - radius), (x2, y2), color, line_width)
        cv2.line(img, (x2 - radius, y2), (x2, y2), color, line_width)

def custom_plot(
        image,
        boxes,
        confs,
        labels,
        use_chinese_mapping=True,
        font_path="D:/PycharmProjects/SafeYolo/yolo_server/utils/LXGWWenKai-Bold.ttf",
        font_size=26,
        line_width=4,
        label_padding_x=10,
        label_padding_y=10,
        radius=8,
        text_color_bgr=(0, 0, 0),
        label_mapping=None,
        color_mapping=None
):

    if label_mapping is None:
        label_mapping = {}
    if color_mapping is None:
        color_mapping = {}

    result_image_cv = image.copy()
    img_height, img_width = image.shape[:2]
    try:
        font_pil = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"错误：无法加载字体文件 '{font_path}'。将使用Pillow默认字体。")
        font_pil = ImageFont.load_default()

    texts_to_draw = []
    for box, conf, label_key in zip(boxes, confs, labels):
        x1, y1, x2, y2 = map(int, box)
        color_bgr = color_mapping.get(label_key, (0, 255, 0))

        if use_chinese_mapping:
            display_label = label_mapping.get(label_key, label_key)
        else:
            display_label = label_key

        label_text_full = f"{display_label} {conf * 100:.1f}%"
        text_width, text_height = get_text_size(label_text_full, font_pil, font_size)
        label_box_actual_width = text_width + 2 * label_padding_x
        label_box_actual_height = text_height + 2 * label_padding_y
        label_box_actual_width = max(label_box_actual_width, 2 * radius)
        label_box_x_min = int(x1 - line_width // 2)
        label_box_y_min_potential_above = y1 - label_box_actual_height

        if label_box_y_min_potential_above < 0:
            if (y2 - y1) >= (label_box_actual_height + line_width * 2):
                label_box_y_min = int(y1 - line_width / 2)
                label_box_y_max = y1 + label_box_actual_height
                draw_label_inside = True
            else:
                label_box_y_min = y2 + line_width
                label_box_y_max = y2 + label_box_actual_height + line_width
                if label_box_y_max > img_height:
                    label_box_y_max = img_height
                    label_box_y_min = img_height - label_box_actual_height
                draw_label_inside = False
        else:
            label_box_y_min = label_box_y_min_potential_above
            label_box_y_max = y1
            draw_label_inside = False

        label_box_x_max = label_box_x_min + label_box_actual_width
        align_right = False
        if label_box_x_max > img_width:
            align_right = True
            label_box_x_min = int(x2 + line_width // 2) - label_box_actual_width
            if label_box_x_min < 0:
                label_box_x_min = 0

        is_label_wider_than_det_box = label_box_actual_width > (x2 - x1)
        label_top_left_round = True
        label_top_right_round = True
        label_bottom_left_round = True
        label_bottom_right_round = True

        if not draw_label_inside:
            if label_box_y_min == y1 - label_box_actual_height:
                if align_right:
                    label_bottom_left_round = is_label_wider_than_det_box
                    label_bottom_right_round = False
                else:
                    label_bottom_left_round = False
                    label_bottom_right_round = is_label_wider_than_det_box
            elif label_box_y_min == y2 + line_width:
                if align_right:
                    label_top_left_round = is_label_wider_than_det_box
                    label_top_right_round = False
                else:
                    label_top_left_round = False
                    label_top_right_round = is_label_wider_than_det_box
        else:
            label_top_left_round = True
            label_top_right_round = True
            if align_right:
                label_bottom_left_round = is_label_wider_than_det_box
                label_bottom_right_round = False
            else:
                label_bottom_left_round = False
                label_bottom_right_round = is_label_wider_than_det_box or not is_label_wider_than_det_box

        det_top_left_round = True
        det_top_right_round = True
        det_bottom_left_round = True
        det_bottom_right_round = True

        if not draw_label_inside:
            if label_box_y_min == y1 - label_box_actual_height:
                if align_right:
                    det_top_left_round = is_label_wider_than_det_box
                    det_top_right_round = False
                else:
                    det_top_left_round = False
                    det_top_right_round = not is_label_wider_than_det_box
            elif label_box_y_min == y2 + line_width:
                if align_right:
                    det_bottom_left_round = is_label_wider_than_det_box
                    det_bottom_right_round = False
                else:
                    det_bottom_left_round = False
                    det_bottom_right_round = is_label_wider_than_det_box
        else:
            det_top_left_round = False
            det_top_right_round = False

        draw_bordered_rounded_rect(result_image_cv, (x1, y1), (x2, y2),
                                color_bgr, line_width, radius,
                                det_top_left_round, det_top_right_round,
                                det_bottom_left_round, det_bottom_right_round)

        draw_filled_rounded_rect(result_image_cv, (label_box_x_min, label_box_y_min),
                                (label_box_x_min + label_box_actual_width, label_box_y_max),
                                color_bgr, radius,
                                label_top_left_round, label_top_right_round,
                                label_bottom_left_round, label_bottom_right_round)

        text_x = label_box_x_min + (label_box_actual_width - text_width) // 2
        text_y = label_box_y_min + (label_box_actual_height - text_height) // 2

        texts_to_draw.append({
            'text': label_text_full,
            'position': (text_x, text_y),
            'font': font_pil,
            'fill_bgr': text_color_bgr
        })

    if texts_to_draw:

        image_pil = Image.fromarray(cv2.cvtColor(result_image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        for text_info in texts_to_draw:
            fill_rgb = (text_info['fill_bgr'][2], text_info['fill_bgr'][1], text_info['fill_bgr'][0])
            draw.text(text_info['position'], text_info['text'], font=text_info['font'], fill=fill_rgb)

        result_image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

    return result_image_cv
