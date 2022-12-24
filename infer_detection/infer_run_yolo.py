import sys
import os
import logging as log
import time

import cv2
from openvino.runtime import Core, AsyncInferQueue
import openvino.runtime as ov
from numpy import random
import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from infer_detection.utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                                         resize_image)
from infer_detection.utils.utils_bbox import DecodeBox

'''
openvino infer v7.xml not official 
'''


class Infer:
    def __init__(self):
        device = "CPU"
        # model_xml_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/IR/int8/yolov7_97_int8.xml"
        # model_bin_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\\IR\\int8\\yolov7_97_int8.bin"
        model_xml_path = "E:/work/openvino_quant/openvino_/IR/int8/yolov7_tiny_int8_demo_mv.xml"
        model_bin_path = "E:/work/openvino_quant/openvino_/IR/int8/yolov7_tiny_int8_demo_mv.bin"
        # classes_path = os.path.abspath(os.path.dirname(__file__)) + "\\model_data\\classes.txt"
        anchors_path = os.path.abspath(os.path.dirname(__file__)) + "\\model_data\\yolo_anchors.txt"
        self.class_names = ["feamle", "male"]
        # set log format
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

        assert os.path.exists(model_xml_path), ".xml file does not exist..."
        assert os.path.exists(model_bin_path), ".bin file does not exist..."

        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = [640, 640]
        self.confidence = 0.7
        self.nms_iou = 0.1
        self.letterbox_image = True

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#

        # self.class_names,self.num_classes = get_classes(classes_path)
        self.num_classes = len(self.class_names)
        anchors, num_anchors = get_anchors(anchors_path)
        self.bbox_util = DecodeBox(anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   anchors_mask)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        # hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
        self.colors = [tuple(self.colors[i]) for i in range(len(self.colors))]

        core = ov.Core()

        self.model = core.read_model(model=model_xml_path)
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.input_layer = self.model.input(0)
        self.output_layer = self.model.outputs

        #  async
        # self.infer_queue = ov.AsyncInferQueue(self.compiled_model)
        # self.infer_queue.set_callback(self.callback)

    def detection(self, image):
        # image = Image.fromarray(image)
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # sync

        request = self.compiled_model.create_infer_request()
        request.infer(inputs={self.input_layer.any_name: image_data})

        # async

        # self.infer_queue.start_async(inputs={self.input_layer.any_name: image_data})
        # self.infer_queue.start_async({self.input_layer.any_name: image_data}, 0)
        # self.infer_queue.wait_all()
        # self.infer_queue.results.data
        # get_output

        result = []
        for i in range(len(self.output_layer)):
            result.append(request.get_output_tensor(i).data)

        outputs = self.bbox_util.decode_box(result)
        # ---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        # ---------------------------------------------------------#
        results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                     image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                     nms_thres=self.nms_iou)
        if results[0] is None:
            return image
        top_label = np.array(results[0][:, 6], dtype='int32')
        top_conf = results[0][:, 4] * results[0][:, 5]
        top_boxes = results[0][:, :4]

        #   字体边框
        font = ImageFont.truetype(font=os.path.abspath(os.path.dirname(__file__)) + "\\model_data\\simhei.ttf",
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


if __name__ == '__main__':
    img_path = 'E:/yolov7/yolov7-tiny-pytorch_/img/795.jpg'
    # img = cv2.imread(img_path)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.open(img_path)
    infer = Infer()
    infer.detection(image).show()
    # cap = cv2.VideoCapture(0)
    # infer = Infer()
    # fps = 0
    # while (True):
    #     # 一帧一帧捕捉
    #     ret, frame = cap.read()
    #     # 我们对帧的操作在这里
    #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(img)
    #     ti = time.time()
    #     r_image = infer.detection(img)
    #     fps = np.round((fps + (1. / (time.time() - ti))) / 2, 0)
    #     # fps = 1 / (time.time() - ti)
    #     img = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
    #     image = cv2.putText(img, f'FPS: {fps}', (5, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.75, (0, 0, 255), 2)
    #
    #     # 显示返回的每帧
    #     cv2.imshow('frame', image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     # r_image.show()
    # # 当所有事完成，释放 VideoCapture 对象
    # cap.release()
    # cv2.destroyAllWindows()
