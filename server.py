import socket
import os

import cv2
import numpy as np
import warnings
import time
import threading
import argparse
from copy import deepcopy

from sklearn import svm
import face_recognition

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name, display_fps

warnings.filterwarnings('ignore')

known_face_encodings = []
known_face_names = []

path = 'face_box'
print('Loading DataBase...')
path_list = os.listdir(path)
path_list.remove('README.md')
for file_name in path_list:
    name_image = cv2.imread(path + '/' + file_name)
    name_image = cv2.cvtColor(name_image, cv2.COLOR_BGR2RGB)
    try:
        name_face_encoding = face_recognition.face_encodings(name_image)[0]
        known_face_encodings.append(name_face_encoding)
        known_face_names.append(file_name[:-6])
    except IndexError:
        print(file_name, 'Not Explicit Face.')
face_name = 'Unknown'
clf = svm.LinearSVC()
clf.fit(known_face_encodings, known_face_names)

server = socket.socket()             #初始化
server.bind(('10.223.174.147',6969))      #绑定ip和端口
server.listen(5) 
print("开始等待接受客户端数据----")

                    #监听，设置最大数量是5

def bit2enc(raw):
    s = bytes.decode(raw)
    s =s.split()
    s[0] = s[0][1:]
    s[-1] = s[-1][:-1]
    s = np.asarray([eval(x) for x in s])
    return s

while True:
    conn,addr = server.accept()      #获取客户端地址
    print(conn,addr)
    print("客户端来数据了")
    while True:
        data = conn.recv(4096)       #接收数据
        # print(data)
        try:
            face_encoding = bit2enc(data)
        except:
            continue
        if len(face_encoding) != 128:
            continue
        # print(face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        svm_match = clf.predict([face_encoding])
        best_match_index = np.argmin(face_distances)
        distance = np.min(face_distances)
        if svm_match == known_face_names[best_match_index]:
            face_name = known_face_names[best_match_index]
        else:
            face_name = 'Unknow'
        if not data:
            print("client has lost")
            break
        print("sending fame_dis:"+face_name+' '+str(distance))
        conn.send((face_name+' '+str(distance)).encode())     #返回数据  

 




server.close()                       #关闭socket
