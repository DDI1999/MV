import os
import json
import logging as log
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import argparse
from infer.myTransforms import ResizeTo224
from onnxruntime import InferenceSession

'''
onnxruntime infer
'''


class detection():
    def __init__(self):
        model_path = 'C:/Users/dell/Desktop/ConvNeXt/onnx/convnext_plume_224.onnx'
        # set log format
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

        assert os.path.exists(model_path), ".xml file does not exist..."

        json_path = 'C:/Users/dell/Desktop/mv/mv_demo/infer/class_index.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r", encoding='utf-8') as f:
            self.class_indict = json.load(f)

        self.onnx_model = InferenceSession(model_path)
        self.input_name = self.onnx_model.get_inputs()[0].name

    def infer(self, img):
        img = Image.fromarray(img)
        data_transform = transforms.Compose([
            ResizeTo224(),
            transforms.ToTensor(),
        ])
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # onnx_Inference

        result = self.onnx_model.run([], {self.input_name: np.array(img)})
        output_onnx = np.squeeze(result)
        predict_onnx = torch.softmax(torch.tensor(output_onnx), dim=0)
        index = np.array(predict_onnx).argmax()

        return self.class_indict[str(index)], predict_onnx[index].numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../pre_img/nostretching_1.jpg")
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()
