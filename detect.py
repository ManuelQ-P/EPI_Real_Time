import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

def run(weights='yolov5s.pt', conf_thres=0.25, device='', dnn=False):
    # Initialize
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names = model.stride, model.names
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        img = torch.from_numpy(frame).to(device)
        img = img.permute(2, 0, 1)  # Change from [H, W, C] to [C, H, W]
        img = img.half() if model.fp16 else img.float()  # Convert to float or half precision
        img /= 255.0  # Scale from [0, 255] to [0, 1]
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # Add batch dimension

        # Model Prediction
        pred = model(img, augment=False, visualize=False)

        # Non-Maximum Suppression
        pred = non_max_suppression(pred, conf_thres, 0.45, None, False, max_det=1000)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                # Display results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show image
        cv2.imshow('YOLOv5 Detection', frame)

        # Wait for 1 seconds
        time.sleep(1)

        # Exit on 'q' key
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
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

