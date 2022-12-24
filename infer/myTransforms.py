from random import random

import numpy as np
from PIL import Image
import os


class ResizeTo224(object):
    def __init__(self, size=224):  # ...是要传入的多个参数

        # 对多参数进行传入
        # 如 self.p = p 传入概率
        # ...
        self.desired_size = size

    def __call__(self, img):  # __call__函数还是只有一个参数传入
        # 该自定义transforms方法的具体实现过程

        old_size = img.size

        ratio = float(self.desired_size) / max(old_size)
        new_size = [int(x * ratio) for x in old_size]
        im = img.resize((new_size[0], new_size[1]), Image.BILINEAR)
        new_im = Image.new('L', (self.desired_size, self.desired_size))
        new_im.paste(im, ((self.desired_size - new_size[0]) // 2,
                          (self.desired_size - new_size[1]) // 2))
        return new_im

# class AddPepperNoise(object):
#     """增加椒盐噪声
#     Args:
#         snr （float）: Signal Noise Rate
#         p (float): 概率值，依概率执行该操作
#     """
#
#     def __init__(self, snr, p=0.9):
#         assert isinstance(snr, float) and (isinstance(p, float))
#         self.snr = snr
#         self.p = p
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): PIL Image
#         Returns:
#             PIL Image: PIL image.
#         """
#         if random.uniform(0, 1) < self.p:
#             img_ = np.array(img).copy()
#             h, w, c = img_.shape
#             signal_pct = self.snr
#             noise_pct = (1 - self.snr)
#             mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
#             mask = np.repeat(mask, c, axis=2)
#             img_[mask == 1] = 255  # 盐噪声
#             img_[mask == 2] = 0  # 椒噪声
#             return Image.fromarray(img_.astype('uint8')).convert('RGB')
#         else:
#             return img
