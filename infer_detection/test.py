import torch

a = torch.arange(36).reshape(3, 2, 2, 3).unsqueeze(dim=0)  # 初始化张量a
b = torch.arange(18).reshape(3, 1, 2, 3).unsqueeze(dim=0)  # 初始化张量a
print('size   of a:', a.size())  # 查看a的shape
print('size   of a:', b.size())  # 查看a的shape
c = a.reshape(1, -1, 3)
# print(c.size())
# print(c)
print('----------------')
d = b.view(1, -1, 3)
# print(d.size())
# print(d)
e = [c, d]
f = torch.cat(e, 1)
print(f)
# a = [False, False, True]
print(f.size())
print(f.shape)
xc = f[..., 2] > 34
for ind, i in enumerate(f):
    i = i[xc[ind]]
import os

# print(os.path.abspath(os.path.dirname(__file__)))
# print(os.path.abspath(os.path.dirname(os.getcwd())))
# print(os.path.abspath(os.getcwd()))
# print(os.getcwd())
import serial
import serial.tools.list_ports

# if __name__ == '__main__':
# com = serial.Serial('COM5', 9600)
# success_bytes = com.write('This is data for test'.encode())
# print(success_bytes)
# print(com)

# plist = list(serial.tools.list_ports.comports())
#
# if len(plist) <= 0:
#     print("The Serial port can't find!")
# else:
#     plist_0 = list(plist[1])
#     serialName = plist_0[1]
#     serialFd = serial.Serial(serialName, 9600, timeout=60)
#     print(plist_0)
#     print(plist)
#     print("check which port was really used >", serialFd.name)
print(torch.FloatTensor([(4.4, 3.4), (5.5, 7.5), (8.8, 10)]))
print(torch.FloatTensor([(4.4, 3.4), (5.5, 7.5), (8.8, 10)]).index_select(1, torch.LongTensor([0])))
print(torch.FloatTensor([1, 3, 2]))
