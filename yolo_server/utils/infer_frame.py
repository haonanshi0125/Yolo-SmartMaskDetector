import cv2
from utils.beautify import custom_plot

def process_frame(frame, result, project_args, beautiful_params, current_fps = None):

    annotated_frame = frame.copy()
    original_height, original_width = frame.shape[:2]

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    labels_idx = result.boxes.cls.cpu().numpy().astype(int)
    labels = [result.names[int(cls_idx)] for cls_idx in labels_idx]

    if project_args.beautify:
        annotated_frame = custom_plot(
            annotated_frame,
            boxes,
            confs,
            labels,
            **beautiful_params
        )
    else:
        annotated_frame = result.plot()

    if current_fps is not None and current_fps > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (0, 255, 0)
        text_background_color = (0, 0, 0)

        (text_width, text_height), _ = cv2.getTextSize(f"FPS: {current_fps:.2f}", font, font_scale, font_thickness)

        padding = 10
        box_x1 = original_width - text_width - padding * 2
        box_y1 = original_height - text_height - padding * 2
        box_x2 = original_width
        box_y2 = original_height

        cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2), text_background_color, -1)
        text_x = original_width - text_width - padding
        text_y = original_height - padding
        cv2.putText(annotated_frame, f"FPS: {current_fps:.2f}", (text_x, text_y), font, font_scale, text_color, font_thickness)

    return annotated_frame
