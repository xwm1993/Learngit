# -*- coding:utf-8 -*-
__auther__ = 'alex'
import cv2
import dlib
import time

predicterPath="model/shape_predictor_68_face_landmarks.dat"

inWidth = 300
inHeight = 300
inScaleFactor = 1.0
net = cv2.dnn.readNetFromCaffe('openCVmodel/deploy.prototxt','openCVmodel/res10_300x300_ssd_iter_140000.caffemodel')
detector=dlib.get_frontal_face_detector()
sp=dlib.shape_predictor(predicterPath)

cap=cv2.VideoCapture("data/1.mp4")
imagNum=0
while cap.isOpened():
    # begin = time.time()
    ok,frame=cap.read()
    if not ok:
        break
    # faces=detector(frame,0)

    # =====================opencv+DNN识别人脸===================
    inputBlob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False)
    net.setInput(inputBlob, 'data')
    PointsArray = net.forward('detection_out')
    cols = frame.shape[1]
    rows = frame.shape[0]
    # dets = np.squeeze(PointsArray)
    dets=PointsArray[0,0,]
    faces = dlib.rectangles()
    # print(dets.shape)
    for det in dets:
        face = dlib.rectangle(int(det[3] * cols), int(det[4] * rows), int(det[5] * cols), int(det[6] * rows))
        faces.append(face)
    # print("face detect:",time.time()-begin)

    if len(faces)==0:
        print("no face found in the image")
        continue

    faces_feature=dlib.full_object_detections()

    # feature = time.time()
    for face in faces:
        faces_feature.append(sp(frame,face))
    # print("get feature:",time.time()-feature)

    # alignment=time.time()
    faces_alignment_image=dlib.get_face_chips(frame,faces_feature,size=160)
    # print("face alignment:",time.time()-alignment)

    for alignment_face in faces_alignment_image:
        imagNum +=1
        fileName='data/image/'+str(imagNum)+'.jpg'
        cv2.imwrite(fileName,alignment_face)

    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2, 8)
    cv2.imshow('faceDetect', frame)
    # print("total:",time.time()-begin)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()