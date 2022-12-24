import time
import cv2

from onnxruntime import InferenceSession
import colorsys

import numpy as np
import torch
from PIL import ImageDraw, ImageFont, Image

from infer_detection.utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                                         resize_image, show_config)
from infer_detection.utils.utils_bbox import DecodeBox

'''
onnxruntime inference v7.onnx not official  
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

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        self.onnx_model = InferenceSession(onnx_path)
        self.input_name = self.onnx_model.get_inputs()[0].name

    # 图像前处理
    def detection(self, image):
        # image = Image.fromarray(image)
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        # onnx_Inference

        onnx_outputs = self.onnx_model.run(None, {self.input_name: np.array(image_data)})

        onnx_outputs = self.bbox_util.decode_box(onnx_outputs)
        # ---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        # ---------------------------------------------------------#
        onnx_result = self.bbox_util.non_max_suppression(torch.cat(onnx_outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
        if onnx_result[0] is None:
            return image
        print('onnx_result[0]: {}'.format(onnx_result[0]))
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


if __name__ == '__main__':
    image = Image.open("E:/yolov7/yolov7-tiny-pytorch_/img/795.jpg")
    infer = Infer()
    r_image = infer.detection(image)
    r_image.show()
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
