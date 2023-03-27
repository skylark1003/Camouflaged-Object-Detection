from ui_cod import Ui_cod
from ui_camera import Ui_camera
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PIL import Image
import cv2
import sys

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from scipy import misc
from model.ResNet_models import Generator
from model.data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class MainWindow(QMainWindow, Ui_cod):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.ori_img.setAlignment(Qt.AlignCenter)  # 设置居中显示
        self.cod_img.setAlignment(Qt.AlignCenter)  # 设置居中显示
       
    def select_image(self):
        # 创建一个文件选择对话框对象
        file_dialog = QFileDialog()
        # 设置文件选择对话框的文件类型过滤器
        file_dialog.setNameFilter('Images (*.png *.xpm *.jpg *.bmp *.gif)')
        # 设置文件选择对话框的视图模式为详细列表
        file_dialog.setViewMode(QFileDialog.Detail)

        # 如果文件选择对话框执行成功
        if file_dialog.exec_():
            # 获取用户选择的文件路径
            file_path = file_dialog.selectedFiles()[0]

            # 使用文件路径创建一个 QPixmap 对象
            pixmap = QPixmap(file_path)
            # 将 QPixmap 对象设置为 ori_img 的图片
            self.ori_img.setPixmap(pixmap.scaled(self.ori_img.width(), self.ori_img.height(), Qt.KeepAspectRatio))
   
    # 进入拍照界面
    def open_camera(self):
        self.camera_window = CameraWindow(parent=self)
        self.camera_window.show()

    # 进行图片分割
    def cod_image(self):
        if not self.ori_img.pixmap():
            QMessageBox.warning(self, "提示", "当前无原始图片！")
            return
        # 获取界面上的原始图像
        # 从 QLabel 组件中获取原始图像
        ori_image = self.ori_img.pixmap()
        temp_path = "./Img/temp.jpg"
        ori_image.save(temp_path)

        # 基于 PyTorch 深度学习框架的图像分割模型测试脚本。
        # 该脚本使用了预先训练好的 ResNet 模型（Generator），
        # 对指定的图像进行测试，并将分割结果保存在指定的路径下。

        # 使用 argparse 模块定义了一些测试所需的参数，
        # 例如测试集大小和特征通道数量等：
        parser = argparse.ArgumentParser()
        parser.add_argument('--testsize', type=int, default=480, help='testing size')
        parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
        opt = parser.parse_args()

        # 设置测试图像数据集路径和预训练模型的路径，并加载预训练模型。
        dataset_path = './'

        generator = Generator(channel=opt.feat_channel)
        generator.load_state_dict(torch.load('./model/Model_50_gen.pth'))

        generator.cuda()
        generator.eval()

        # 定义测试数据集的名称
        test_datasets = ['Img/']

        # 然后循环遍历测试数据集，并在指定路径下创建一个文件夹来存储分割结果：
        for dataset in test_datasets:
            save_path = './result/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 对于每个测试集中的图像，使用 test_dataset 类加载图像数据，
            # 并将其传递给预训练的 ResNet 模型进行分割。
            # 然后使用 bilinear 插值将输出的特征图缩放到与原始图像相同的大小。
            # 最后将分割结果保存为图像文件：
            image_root = dataset_path + dataset
            test_loader = test_dataset(image_root, opt.testsize)
            for i in range(test_loader.size):
                print(i)
                image, HH, WW, name = test_loader.load_data()
                image = image.cuda()
                _,_,generator_pred = generator.forward(image)
                res = generator_pred
                res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
                cv2.imwrite(save_path+name, res)

        # 本地路径
        path = "./result/temp.png"
        # 将图片转化为QPixmap
        pixmap = QPixmap(path)
        # 设置QLabel的背景
        self.cod_img.setPixmap(pixmap.scaled(self.cod_img.width(), self.cod_img.height(), Qt.KeepAspectRatio))

    def save_image(self):
        # 判断是否有图片
        if not self.cod_img.pixmap():
            QMessageBox.warning(self, "提示", "当前没有检测结果！")
        
        else:
            # 弹出文件保存对话框，让用户选择保存路径、文件名和格式
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "JPEG (*.jpg);;PNG (*.png)", options=options)
            if not file_name:
                return
            
            # 从QPixmap中获取QImage
            image = self.cod_img.pixmap().toImage()

            # 保存图片
            if file_name.endswith('.jpg'):
                image.save(file_name, 'JPG')
            elif file_name.endswith('.png'):
                image.save(file_name, 'PNG')
            else:
                QMessageBox.warning(self, '提示', '不支持的图片格式')

    def ori_image_show(self):
        if not self.cod_img.pixmap():
            QMessageBox.warning(self, "提示", "当前没有检测结果！")
            return
        
        image1 = Image.open('Img\\temp.jpg')
        image2 = Image.open('result\\temp.png')

        # 将第二张图片的透明度调为50
        image2.putalpha(128)

        # 将第二张图片覆盖在第一张图片上
        image1.paste(image2, (0, 0), image2)

        # 保存合成后的图片
        image1.save('./result/mix.png')

        # 将合成后的图片显示在QLabel中
        pixmap = QPixmap('./result/mix.png')
        self.cod_img.setPixmap(pixmap.scaled(self.cod_img.width(), self.cod_img.height(), Qt.KeepAspectRatio))

    def ori_image_hide(self):
        if not self.cod_img.pixmap():
            QMessageBox.warning(self, "提示", "当前没有检测结果！")
            return
        # 本地路径
        path = "./result/temp.png"
        # 将图片转化为QPixmap
        pixmap = QPixmap(path)
        # 设置QLabel的背景
        self.cod_img.setPixmap(pixmap.scaled(self.cod_img.width(), self.cod_img.height(), Qt.KeepAspectRatio))

    # 退出程序
    def exit_app(self):
        if os.path.isfile("Img\\temp.jpg"):
            os.remove("Img\\temp.jpg")  # 删除文件
        if os.path.isfile("result\mix.png"):
            os.remove("result\mix.png")  # 删除文件
        if os.path.isfile("result\\temp.png"):
            os.remove("result\\temp.png")  # 删除文件
        app.quit()

class CameraWindow(QMainWindow, Ui_camera):
    def __init__(self, parent=None):
        super(CameraWindow, self).__init__(parent)
        self.setupUi(self)  # 初始化UI
        self.taken_image = None  # 初始化拍摄的照片为空
        self.parent = parent  # 保存父界面
        self.camera_video.setAlignment(Qt.AlignCenter)  # 设置居中显示

    # 重写show方法，在显示窗口时打开摄像头并捕获图像
    def show(self):
        self.capture = cv2.VideoCapture(0)  # 打开摄像头

        self.timer = QTimer()  # 定义定时器
        self.timer.timeout.connect(self.display_video_stream)  # 连接定时器信号槽
        self.timer.start(1000 // 30)  # 设置定时器触发时间间隔
        
        super().show()  # 调用父类show方法显示窗口

    # 定义显示视频流的槽函数
    def display_video_stream(self):
        ret, frame = self.capture.read()  # 读取摄像头图像

        if ret:  # 如果图像读取成功
            # 将BGR格式转换成RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 创建QImage对象并设置为QLabel的背景
            q_img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            q_img_scale = QPixmap.fromImage(q_img).scaled(self.camera_video.width(), self.camera_video.height(), 
                                                          Qt.KeepAspectRatio, Qt.FastTransformation)
            self.camera_video.setPixmap(q_img_scale)
            
    # 拍照
    def take_photo(self):
        # 停止定时器，停止捕捉图像
        self.timer.stop()
        ret, frame = self.capture.read()  # 读取图像

        if ret:  # 如果图像读取成功
            # 将BGR格式转换成RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 创建QImage对象并设置为QLabel的背景
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            q_img_scaled = q_img.scaled(self.camera_video.width(), self.camera_video.height(), Qt.KeepAspectRatio)
            self.camera_video.setPixmap(QPixmap.fromImage(q_img_scaled))
            self.taken_image = frame

    # 重拍
    def retake_photo(self):
        # 释放之前的图片资源
        self.taken_image = None

        # 显示摄像头内容
        self.timer.start(1000 // 30)  # 重新开启定时器
        self.display_video_stream()  # 显示摄像头内容

    # 确认使用该图片
    def confirm_photo(self):
        if self.taken_image is None:
            # 如果当前未拍摄照片，则弹出提示窗口
            QMessageBox.warning(self, "提示", "请先拍照！")
            return
        
        # 将拍摄的照片显示在主界面的ori_img中
        q_img = QImage(self.taken_image.data, self.taken_image.shape[1], self.taken_image.shape[0], QImage.Format_RGB888)
        q_img_scaled = q_img.scaled(self.parent.ori_img.width(), self.parent.ori_img.height(), Qt.KeepAspectRatio)
        self.parent.ori_img.setPixmap(QPixmap.fromImage(q_img_scaled))
        # 关闭摄像头窗口
        self.close()

    # 关闭界面
    def exit_camera(self):
        self.close()
   
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
