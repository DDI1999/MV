import time

import cv2
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image
from numpy import random

from infer_detection.utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                                         resize_image, show_config)
from infer_detection.utils.utils_bbox import DecodeBox

'''
opencvDNN部署推理v7的onnx模型
'''


class Infer:
    def __init__(self):
        onnx_path = "E:/work/mv/mv_demo/IR/onnx/best97.onnx"
        classes_path = 'E:/work/mv/mv_demo/infer_detection/model_data/classes.txt'
        anchors_path = 'E:/work/mv/mv_demo/infer_detection/model_data/yolo_anchors.txt'
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = [640, 640]
        self.confidence = 0.6
        self.nms_iou = 0.1
        self.letterbox_image = True

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(classes_path)
        anchors, num_anchors = get_anchors(anchors_path)
        self.bbox_util = DecodeBox(anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   anchors_mask)

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
        self.colors = [tuple(self.colors[i]) for i in range(len(self.colors))]

        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = self.net.getLayerNames()
        self.layerNames = self.net.getUnconnectedOutLayers()
        self.outputsNames = [self.names[i - 1] for i in self.layerNames]

    def detection(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        # blob = cv2.dnn.blobFromImage(image, 1 / 255, (640, 640), swapRB=True, crop=False)
        self.net.setInput(image_data)
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        outputs = list(self.net.forward(self.outputsNames))
        outputs.reverse()
        outputs = self.bbox_util.decode_box(outputs)
        onnx_result = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
        if onnx_result[0] is None:
            return image
        # print('onnx_result[0]: {}'.format(onnx_result[0]))
        top_label = np.array(onnx_result[0][:, 6], dtype='int32')
        top_conf = onnx_result[0][:, 4] * onnx_result[0][:, 5]
        top_boxes = onnx_result[0][:, :4]
        # print(top_label, '\n', top_conf, '\n', top_boxes)

        #   字体边框
        font = ImageFont.truetype(font='E:/work/mv/mv_demo/infer_detection/model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #   图像绘制

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle((left + i, top + i, right - i, bottom - i), outline=self.colors[c])
            draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)), fill=self.colors[c])
            draw.text(tuple(text_origin), str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image


if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    # infer = Infer()
    # while True:
    #     flog, frame = cap.read()
    #     if flog is not None:
    #         frame = Infer.detection(frame)
    #         cv2.imshow('frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # image = Image.open("E:/yolov7/yolov7-tiny-pytorch_/img/795.jpg")
    # infer = Infer()
    # r_image = infer.detection(image)
    # r_image.show()

    cap = cv2.VideoCapture(0)
    infer = Infer()
    fps = 0
    while (True):
        # 一帧一帧捕捉
        ret, frame = cap.read()
        # 我们对帧的操作在这里
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        ti = time.time()
        r_image = infer.detection(img)
        fps = np.round((fps + (1. / (time.time() - ti))) / 2, 0)
        # fps = 1 / (time.time() - ti)
        img = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
        image = cv2.putText(img, f'FPS: {fps}', (5, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255), 2)

        # 显示返回的每帧
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # r_image.show()
    # 当所有事完成，释放 VideoCapture 对象
    cap.release()
    cv2.destroyAllWindows()
