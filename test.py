# import time
#
# import serial.tools.list_ports
#
# predict_index = 0
# NO1_open = '01 05 00 00 FF 00 8C 3A'  # "上灯"
# NO1_down = "01 05 00 00 00 00 CD CA"
#
# NO2_open = "01 05 00 01 FF 00 DD FA"
# NO2_down = "01 05 00 01 00 00 9C 0A"
#
# NO3_open = "01 05 00 02 FF 00 2D FA"
# NO3_down = "01 05 00 02 00 00 6C 0A"
#
# NO4_open = "01 05 00 03 FF 00 7C 3A"
# NO4_down = "01 05 00 03 00 00 3D CA"
# light_num = 4
#
# predict_light = [NO1_open, NO2_open, NO3_open,
#                  NO4_open, NO1_down, NO2_down, NO3_down, NO4_down]
#
# port_list = list(serial.tools.list_ports.comports())
# print(port_list[1].device)
# ser = serial.Serial(port_list[1].device, 9600, timeout=1)
#
# k = bytes.fromhex(predict_light[predict_index])
# d = bytes.fromhex(predict_light[predict_index + light_num])
# success_bytes = ser.write(k)
# end_code = success_bytes // 2 + 4
# data = []
# for _ in range(end_code):
#     tmp = ser.read(1)
#     data.append(tmp.hex())
# time.sleep(2)
# success_bytes = ser.write(d)
# data = []
# end_code = success_bytes // 2 + 4
# for _ in range(end_code):
#     tmp = ser.read(1)
#     data.append(tmp.hex())
import sys

from PyQt5.Qt import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("hhhhh")





    window.show()
    sys.exit(app.exec_())
