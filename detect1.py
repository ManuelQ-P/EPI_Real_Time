import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from collections import Counter
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

def get_most_prominent_color(image, grid_size=10):
    h, w, _ = image.shape
    prominent_colors = []

    for i in range(0, h, grid_size):
        for j in range(0, w, grid_size):
            grid = image[i:i + grid_size, j:j + grid_size]
            pixels = grid.reshape(-1, 3)
            pixel_counts = Counter(map(tuple, pixels))
            most_common_color = pixel_counts.most_common(1)[0][0]
            prominent_colors.append(most_common_color)

    final_color_counts = Counter(prominent_colors)
    dominant_color = final_color_counts.most_common(1)[0][0]

    return dominant_color

def match_color(rgb_color):
    r, g, b = rgb_color
    if r > 150 and g < 80 and b < 80:
        return 'Red'
    elif r < 80 and g < 80 and b > 150:
        return 'Blue'
    elif r > 200 and g > 200 and b < 100:
        return 'Yellow'
    elif r < 100 and g > 150 and b < 100:
        return 'Green'
    elif r > 100 and g < 100 and b > 150:
        return 'Purple'
    elif r > 200 and g > 100 and b < 80:
        return 'Orange'
    elif r > 200 and g < 150 and b > 150:
        return 'Pink'
    elif r < 60 and g < 60 and b < 60:
        return 'Black'
    elif r > 200 and g > 200 and b > 200:
        return 'White'
    elif 100 < r < 200 and 100 < g < 200 and 100 < b < 200:
        return 'Gray'
    else:
        return 'Other'

def get_position_label(center_x, center_y, img_width, img_height):
    horizontal_pos = "Center"
    vertical_pos = "Center"
    if center_x < img_width // 3:
        horizontal_pos = "Left"
    elif center_x > 2 * img_width // 3:
        horizontal_pos = "Right"
    if center_y < img_height // 3:
        vertical_pos = "Top"
    elif center_y > 2 * img_height // 3:
        vertical_pos = "Bottom"
    if horizontal_pos == "Center" and vertical_pos == "Center":
        return "Center"
    return f"{vertical_pos}-{horizontal_pos}"

def run(weights='yolov5s.pt', conf_thres=0.25, device='', dnn=False, output_txt="summary.txt"):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names = model.stride, model.names
    imgsz = check_img_size((640, 640), s=stride)

    cap = cv2.VideoCapture(0)

    with open(output_txt, 'a') as f:  # Append mode
        f.write("Real-Time Object Detection Summary\n\n")
        f.flush()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_height, img_width, _ = frame.shape

            img = torch.from_numpy(frame).to(device)
            img = img.permute(2, 0, 1)
            img = img.half() if model.fp16 else img.float()
            img /= 255.0
            if len(img.shape) == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)

            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        x1, y1, x2, y2 = map(int, xyxy)
                        obj_width = x2 - x1
                        obj_height = y2 - y1
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        position_label = get_position_label(center_x, center_y, img_width, img_height)
                        obj_img = frame[y1:y2, x1:x2]
                        dominant_color = get_most_prominent_color(obj_img)
                        color_name = match_color(dominant_color)
                        summary = f"Object: {names[int(cls)]}, Confidence: {conf:.2f}, Color: {color_name}, " \
                                  f"Position: {position_label}, Width: {obj_width}, Height: {obj_height}\n"
                        f.write(summary)
                        f.flush()  # Ensure immediate write to file
                        display_text = f'{label}, Color: {color_name}, Pos: {position_label}, W: {obj_width}, H: {obj_height}'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, display_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Detection', frame)
            time.sleep(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--output-txt', type=str, default="summary.txt", help='path to output summary txt file')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

