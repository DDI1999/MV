import sys
import os
import json
import logging as log

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import openvino.runtime as ov

from infer.myTransforms import ResizeTo224


def detection(img):
    device = "CPU"
    model_xml_path = "../IR/int8/yolov7_97_int8.xml"
    model_bin_path = "../IR/int8/yolov7_97_int8.bin"


    # set log format
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    assert os.path.exists(model_xml_path), ".xml file does not exist..."
    assert os.path.exists(model_bin_path), ".bin file does not exist..."
    core = ov.Core()

    model = core.read_model(model=model_xml_path)
    compiled_model = core.compile_model(model=model, device_name=device)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    # img.show()
    data_transform = transforms.Compose([
        ResizeTo224(),
        transforms.ToTensor(),
    ])
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = './class_index.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)

    # model.input(0).any_name
    input_layer = model.input(0)
    # print('Model Input Info')

    output_layer = model.output(0)
    # print('Model Output Info')

    request = compiled_model.create_infer_request()
    request.infer(inputs={input_layer.any_name: img})
    result = request.get_output_tensor(output_layer.index).data
    # [[]] -> []
    prediction = np.squeeze(result)
    predict = torch.softmax(torch.tensor(prediction), dim=0)

    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))

    index = np.array(predict).argmax()
    print("class: {} pro: {:.4}".format(class_indict[str(index)], predict[index].numpy()))


