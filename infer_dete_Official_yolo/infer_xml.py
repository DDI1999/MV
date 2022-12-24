import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import random
import openvino.runtime as ov

from utils.mydatasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.plots import plot_one_box

'''
pytorch推理yolo模型
'''


class detect():
    def __init__(self):

        self.device = "cpu"
        model_xml_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/IR/32/two_cla_best_grad.xml"
        model_bin_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\\IR\\32\\two_cla_best_grad.bin"
        imgsz = 640
        # Initialize
        set_logging()
        self.conf_thres = 0.6
        self.iou_thres = 0.1

        assert os.path.exists(model_xml_path), ".xml file does not exist..."
        assert os.path.exists(model_bin_path), ".bin file does not exist..."
        # Load model

        core = ov.Core()

        self.model = core.read_model(model=model_xml_path)
        self.compiled_model = core.compile_model(model=self.model, device_name="CPU")
        self.input_layer = self.model.input(0)
        self.output_layer = self.model.outputs

        self.stride = 32  # max downsample stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        # Get names and colors
        self.names = ["feamle", "male"]
        self.num_classes = len(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detection(self, image):
        raw_image = image
        dataset = LoadImages(image, img_size=self.imgsz, stride=self.stride)
        image = dataset.next()
        image = torch.from_numpy(image).to(self.device).float()
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        # Inference
        request = self.compiled_model.create_infer_request()
        request.infer(inputs={self.input_layer.any_name: image})
        result = []
        for i in range(len(self.output_layer)):
            result.append(request.get_output_tensor(i).data)

        # Apply NMS
        pred = non_max_suppression(torch.Tensor(result[0]), self.conf_thres, self.iou_thres,
                                   agnostic=False)
        if pred[0] is None:
            return raw_image
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''

            if len(det):
                # Rescale boxes from img_size to im0 size 将resize之后的box坐标还原到原图box坐标
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], raw_image.shape).round()

                # Print results 添加结果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, raw_image, label=label, color=self.colors[int(cls)], line_thickness=1)

            return raw_image  # BGR


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='./weights/two_cla_best.pt', help='model.pt path(s)')
    # parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # opt = parser.parse_args()
    # print(opt)
    path = 'E:/yolov7/yolov7-tiny-pytorch_/img/795.jpg'
    img = cv2.imread(path)
    # img = Image.open(path)
    with torch.no_grad():
        detection = detect()
        BGR = detection.detection(image=img)
        image = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
        Image.fromarray(image).show()
