# -*- coding:utf-8 -*-
import cv2
import sys
import threading
from PyQt5.Qt import *
import os
from PyQt5 import QtWidgets
import numpy as np
from window import Ui_MainWindow
# 分类接口
# from infer.infer_onnx_class import detection
# from infer.infer_torch_class import detection
# from infer.class_infer_run import infer
# 检测接口
# from infer_detection.onnx_infer_yolo import Infer
from infer_detection.infer_run_yolo import Infer
# 样本帧
from intime_frame import gen_frame  # 导入save_image类

sys.path.append("../MvImport")
from MvImport.MvCameraControl_class import *


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    sendAddDeviceName = pyqtSignal()  # 定义一个添加设备列表的信号。
    deviceList = MV_CC_DEVICE_INFO_LIST()  # 设备列表
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE  # gige/usb

    g_bExit = False  # 相机开启标志
    camera_information = False  # 获取相机标志
    opencamera_flay = False  # 打开相机标志
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    # ch:创建推理实例
    infer = Infer()
    # infer = detection()
    # ch:创建save实例
    gen_frame = gen_frame()

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        # self.connect_and_emit_sendAddDeviceName()
        self.init()

        self.label.setScaledContents(True)  # 图片自适应
        # self.label_2.setScaledContents(True)  # 图片自适应

    def init(self):
        # 打开摄像头
        self.pushButton.clicked.connect(self.openCamera)
        # 关闭摄像头

        self.pushButton_2.clicked.connect(self.closeCamera)
        # Connect the sendAddDeviceName signal to a slot.
        # self.sendAddDeviceName.connect(self.camera_information)
        # Emit the signal.
        # self.sendAddDeviceName.emit()

    # 获得所有相机的列表存入cmbSelectDevice中
    def get_camera_information(self):
        '''选择所有能用的相机到列表中，
             gige相机需要配合 sdk 得到。
        '''
        # 得到相机列表
        # tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        # ch:枚举设备 | en:Enum device
        ret = MvCamera.MV_CC_EnumDevices(self.tlayerType, self.deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            # QMessageBox.critical(self, '错误', '读取设备驱动失败！')
            # sys.exit()
        if self.deviceList.nDeviceNum == 0:
            QMessageBox.critical(self, "错误", "没有发现设备 ！ ")
            # print("find no device!")
            # sys.exit()
        else:
            QMessageBox.information(self, "提示", "发现了 %d 个设备 !" % self.deviceList.nDeviceNum)
        # print("Find %d devices!" % self.deviceList.nDeviceNum)
        if self.deviceList.nDeviceNum == 0:
            return None

        for i in range(0, self.deviceList.nDeviceNum):
            mvcc_dev_info = cast(self.deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)

        self.camera_information = True

    # 设置相机基本参数
    def set_parameter(self):
        AcquisitionFrameRate = 90  # 帧率
        Width = 1440
        Height = 1080
        PixelFormat = PixelType_Gvsp_BayerRG8
        # PixelFormat = PixelType_Gvsp_Mono8
        ExposureTime = 4000

        ret_width = self.cam.MV_CC_SetIntValue("Width", Width)
        if ret_width != 0:
            QMessageBox.critical(self, "错误", "Width ! ret[0x%x]" % ret_width)
        ret_height = self.cam.MV_CC_SetIntValue("Height", Height)
        if ret_height != 0:
            QMessageBox.critical(self, "错误", "Height ! ret[0x%x]" % ret_height)
        ret_frameRate = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(AcquisitionFrameRate))
        if ret_frameRate != 0:
            QMessageBox.critical(self, "错误", "AcquisitionFrameRate ! ret[0x%x]" % ret_frameRate)
        ret_pixelFormat = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelFormat)
        if ret_pixelFormat != 0:
            QMessageBox.critical(self, "错误", "PixelFormat ! ret[0x%x]" % ret_pixelFormat)
        ret_exposureTime = self.cam.MV_CC_SetFloatValue("ExposureTime", float(ExposureTime))
        if ret_exposureTime != 0:
            QMessageBox.critical(self, "错误", "ExposureTime ! ret[0x%x]" % ret_exposureTime)

    def get_parameter(self):
        ResultingFrameRate = MVCC_FLOATVALUE()
        memset(byref(ResultingFrameRate), 0, sizeof(MVCC_FLOATVALUE))

        self.cam.MV_CC_GetFloatValue("AcquisitionFrameRate", ResultingFrameRate)  # 实际帧率
        # ResultingFrameRate
        return ResultingFrameRate.fCurValue

    def get_Value(self, cam, param_type="int_value", node_name="PayloadSize"):
        """
        :param cam:            相机实例
        :param_type:           获取节点值得类型
        :param node_name:      节点名 可选 int 、float 、enum 、bool 、string 型节点
        :return:               节点值
        """
        if param_type == "int_value":
            stParam = MVCC_INTVALUE_EX()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
            ret = cam.MV_CC_GetIntValueEx(node_name, stParam)
            if ret != 0:
                print("获取 int 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
                sys.exit()
            int_value = stParam.nCurValue
            return int_value

        elif param_type == "float_value":
            stFloatValue = MVCC_FLOATVALUE()
            memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
            ret = cam.MV_CC_GetFloatValue(node_name, stFloatValue)
            if ret != 0:
                print("获取 float 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
                sys.exit()
            float_value = stFloatValue.fCurValue
            return float_value

        elif param_type == "enum_value":
            stEnumValue = MVCC_ENUMVALUE()
            memset(byref(stEnumValue), 0, sizeof(MVCC_ENUMVALUE))
            ret = cam.MV_CC_GetEnumValue(node_name, stEnumValue)
            if ret != 0:
                print("获取 enum 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
                sys.exit()
            enum_value = stEnumValue.nCurValue
            return enum_value

        elif param_type == "bool_value":
            stBool = c_bool(False)
            ret = cam.MV_CC_GetBoolValue(node_name, stBool)
            if ret != 0:
                print("获取 bool 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
                sys.exit()
            return stBool.value

        elif param_type == "string_value":
            stStringValue = MVCC_STRINGVALUE()
            memset(byref(stStringValue), 0, sizeof(MVCC_STRINGVALUE))
            ret = cam.MV_CC_GetStringValue(node_name, stStringValue)
            if ret != 0:
                print("获取 string 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name, ret))
                sys.exit()
            string_value = stStringValue.chCurValue
            return string_value

    # 打开摄像头
    def openCamera(self):
        self.get_camera_information()
        if self.camera_information:
            self.g_bExit = False
            # ch:选择设备并创建句柄 | en:Select device and create handle
            stDeviceList = cast(self.deviceList.pDeviceInfo[int(0)], POINTER(MV_CC_DEVICE_INFO)).contents
            ret = self.cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                # print("create handle fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "创建句柄失败 ! ret[0x%x]" % ret)
                # sys.exit()
            # ch:打开设备 | en:Open device
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                # print("open device fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "打开设备失败 ! ret[0x%x]" % ret)
                # sys.exit()
            #  设置默认参数
            else:
                self.set_parameter()

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        # print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
                        QMessageBox.warning(self, "警告", "报文大小设置失败 ! ret[0x%x]" % ret)
                else:
                    # print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
                    QMessageBox.warning(self, "警告", "报文大小获取失败 ! ret[0x%x]" % nPacketSize)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                # print("set trigger mode fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "设置触发模式失败 ! ret[0x%x]" % ret)
                # sys.exit()

            # ch:获取数据包大小 | en:Get payload size
            stParam = MVCC_INTVALUE()
            memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
            ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                # print("get payload size fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "获取有效负载大小失败 ! ret[0x%x]" % ret)
                sys.exit()
            nDataSize = stParam.nCurValue
            pdata = (c_ubyte * nDataSize)()  # 8位

            # ch:开始取流 | en:Start grab image
            ret = self.cam.MV_CC_StartGrabbing()
            if ret != 0:
                # print("start grabbing fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "开始抓取图像失败 ! ret[0x%x]" % ret)
                # sys.exit()

            self.opencamera_flay = True
            try:

                hThreadHandle = threading.Thread(target=self.work_thread,
                                                 args=(self.cam, pdata, nDataSize))
                hThreadHandle.start()
            except:
                # print("error: unable to start thread")
                QMessageBox.critical(self, "错误", "无法启动线程 ! ")

        else:
            QMessageBox.critical(self, '错误', '获取相机信息失败！')
            return None

    # 关闭相机
    def closeCamera(self):
        if self.opencamera_flay:
            self.g_bExit = True
            # ch:停止取流 | en:Stop grab image
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                # print("stop grabbing fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "停止抓取图像失败 ! ret[0x%x]" % ret)
                # sys.exit()

            # ch:关闭设备 | Close device
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                # print("close deivce fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "停止设备失败 ! ret[0x%x]" % ret)
            # ch:销毁句柄 | Destroy handle
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                # print("destroy handle fail! ret[0x%x]" % ret)
                QMessageBox.critical(self, "错误", "销毁处理失败 ! ret[0x%x]" % ret)

            self.label.clear()  # 清除label组件上的图片
            # self.label_2.clear()  # 清除label组件上的图片
            self.label.setText("摄像头")
            # self.label_2.setText("显示图片")
            self.camera_information = False
            self.opencamera_flay = False
        else:
            QMessageBox.critical(self, '错误', '未打开摄像机！')
            return None

    # 逐帧处理数据流
    def work_thread(self, cam=0, pData=0, nDataSize=0):
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        while True:
            ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
            if ret == 0:
                image = np.frombuffer(pData, dtype=np.uint8)  # 将c_ubyte_Array转化成ndarray得到(1555200,)
                # image = np.asarray(pData)  # 将c_ubyte_Array转化成ndarray得到（1555200，）
                image = self.image_control(image, stFrameInfo)
                # 调用save_image保存帧(BGR格式保存)
                # self.gen_frame.gen_frame_30(image)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 分类接口
                # classes, pro = self.infer.detection(image)  # return class and pro
                # classes, pro = self.infer.infer(image)
                # self.lineEdit.setText('class: {:10} pro: {:.4}'.format(classes, pro))
                # self.label_2.setText('class: {:10} pro: {:.4}'.format(classes, pro))

                # 检测接口
                image = self.infer.detection(image)
                image = np.asarray(image)
                #  显示图片

                image_height, image_width, image_depth = image.shape  # 读取图像高宽深度
                self.image_show = QImage(image.data, image_width, image_height, image_width * image_depth,
                                         QImage.Format_RGB888)

                # image_height, image_width = image.shape  # 读取图像高宽
                # self.image_show = QImage(image.data, image_width, image_height,
                #                          QImage.Format_Grayscale8)
                self.label.setPixmap(QPixmap.fromImage(self.image_show))

                # 获取相机参数
                # print(self.get_parameter())
                # print(self.get_Value(cam, 'float_value', 'AcquisitionFrameRate'))

            if self.g_bExit:
                del pData
                break

    # 将获取的数据帧转换成RGB返回
    def image_control(self, data, stFrameInfo):
        if stFrameInfo.enPixelType == 17301505:
            return data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            # self.image_show_(image=image, name=stFrameInfo.nHeight)
        elif stFrameInfo.enPixelType == 17301513:
            data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            return cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)
            # self.image_show_(image=image, name=stFrameInfo.nHeight)
        elif stFrameInfo.enPixelType == 35127316:
            data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            # self.image_show_(image=image, name=stFrameInfo.nHeight)
        elif stFrameInfo.enPixelType == 34603039:
            data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
            return cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
            # self.image_show_(image=image, name=stFrameInfo.nHeight)
        elif stFrameInfo.enPixelType == 17301505:
            data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
            return data

    # 重写关闭函数
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '提示', "确认退出吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            # 用过sys.exit(0)和sys.exit(app.exec_())，但没起效果
            os._exit(0)
        else:
            event.ignore()


if __name__ == '__main__':
    from PyQt5 import QtCore

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应分辨率

    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
