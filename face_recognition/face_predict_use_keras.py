#-*-coding:utf-8 -*-
__auther__ = 'Alex'
# -*- coding: utf-8 -*-

import cv2
import dlib
import time
from face_train_use_keras import Model

if __name__ == '__main__':
    # 加载模型
    model = Model()
    model.load_model(file_path='./model/jiaYouBG.model.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 捕获指定摄像头的实时视频流
    cap = cv2.VideoCapture("./video/2.mp4")

    inWidth = 300
    inHeight = 300
    inScaleFactor = 1.0
    net = cv2.dnn.readNetFromCaffe('openCVmodel/deploy.prototxt','openCVmodel/res10_300x300_ssd_iter_140000.caffemodel')

    predicterPath = "model/shape_predictor_68_face_landmarks.dat"
    sp = dlib.shape_predictor(predicterPath)

    # 人脸识别分类器本地存储路径
    cascade_path = "/home/alex/桌面/face_recgnition/openCVmodel/haarcascades/haarcascade_frontalface_alt2.xml"

    # 循环检测识别人脸
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧视频
        if not ok:
            break
        start = time.time()
        # 图像灰化，降低计算复杂度
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # =====================opencv+DNN识别人脸===================
        inputBlob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False)
        net.setInput(inputBlob, 'data')
        PointsArray = net.forward('detection_out')
        cols = frame.shape[1]
        rows = frame.shape[0]
        #PointArry是4维的,要进行维度压缩到后2维
        dets = PointsArray[0, 0,]

        # print(dets.shape)
        for det in dets:
            # 获取脸部左上角和右下角的坐标
            lx, ly, rx, ry = (int(det[3] * cols), int(det[4] * rows), int(det[5] * cols), int(det[6] * rows))
            # 进行人脸对齐
            face = dlib.rectangle(lx,ly,rx,ry)
            face_feature = sp(frame,face)
            face_alignment_image = dlib.get_face_chip(frame, face_feature, size=160)
            # 将对齐后的图片进行预测
            faceID = model.face_predict(face_alignment_image)
            if faceID == 0:
                cv2.rectangle(frame, (lx - 10, ly - 10), (rx + 10, ry + 10), color, thickness=2)
                # 文字提示是谁
                cv2.putText(frame, 'xiamama',(lx + 30, ly + 30),cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 255), 2)
            elif faceID == 1 :
                cv2.rectangle(frame, (lx - 10, ly - 10), (rx + 10, ry + 10), color, thickness=2)
                cv2.putText(frame, 'liuxing', (lx + 30, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            elif faceID == 2:
                cv2.rectangle(frame, (lx - 10, ly - 10), (rx + 10, ry + 10), color, thickness=2)
                cv2.putText(frame, 'xiaoyu', (lx + 30, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            elif faceID == 3:
                cv2.rectangle(frame, (lx - 10, ly - 10), (rx + 10, ry + 10), color, thickness=2)
                cv2.putText(frame, 'xiaoxue', (lx + 30, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                cv2.rectangle(frame, (lx - 10, ly - 10), (rx + 10, ry + 10), color, thickness=2)
                cv2.putText(frame, 'xiababa', (lx + 30, ly + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        print("cost time:",time.time()-start)
        cv2.imshow("faceDetect", frame)
        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()