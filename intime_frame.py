# Author: ypp
# CreateTime: 2022/10/30
# FileName: cv
# Description: simple introduction of the code
import os.path

import cv2
import logging


# save img

class gen_frame():
    def __init__(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.count = 0  # count the number of pictures
        self.frame_interval = 1  # video frame count interval frequency
        self.frame_interval_count = 0
        self.save_path = 'E:/work/mv/gen_frame/data/'
        c = 0
        for i in os.listdir(self.save_path):  # HK_01
            if int(i.split('_')[-1]) >= c:
                c = int(i.split('_')[-1]) + 1
        self.save_path = os.path.join(self.save_path, 'HK_{}'.format(c))
        os.mkdir(self.save_path)

    def save_image(self, num, image):
        image_path = os.path.join(self.save_path, '{}.jpg'.format(str(num)))
        cv2.imwrite(image_path, image)

    def gen_frame_30(self, frame_data):
        # store operation every time f frame
        if self.frame_interval_count % self.frame_interval == 0:
            self.save_image(self.count, frame_data)
            logging.info("num：" + str(self.count) + ", frame: " +
                         str(self.frame_interval_count))
            self.count += 1
        self.frame_interval_count += 1

# if __name__ == '__main__':
# gf = gen_frame()
# gf.gen_frame_30(cv2.imread('./ddd.jpg'))
