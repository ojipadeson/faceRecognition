# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import warnings
import time
import threading
import argparse
from copy import deepcopy

import face_recognition

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

thread_lock = threading.Lock()
thread_exit = False

event = threading.Event()

known_face_encodings = []
known_face_names = []

GLOBAL_COUNTER = 0
capture = cv2.VideoCapture(0)

ATTACK_WARNING = False


class VideoThread(threading.Thread):
    def __init__(self):
        super(VideoThread, self).__init__()
        self.frame = np.zeros((480, 640, 3)).astype('uint8')

    def get_frame(self):
        return deepcopy(self.frame)

    def run(self):
        global thread_exit
        global capture
        while not thread_exit:
            ret, frame = capture.read()
            if ret:
                thread_lock.acquire()
                self.frame = frame
                thread_lock.release()
            else:
                thread_exit = True


# Variables that have the SAME NAME with variables in threading.Thread is NOT recommended to define
# Startswith "_", 'daemon', 'get', 'i', 'join', 'run', 'set', 'start', 'name'
class DetectThread(threading.Thread):
    def __init__(self):
        super(DetectThread, self).__init__()
        self.org_bbox = [0] * 4
        self.liveness = 0
        self.score = 0
        self.working = False
        self.overflow = False
        self.mentioned_box = []
        self._frame = None

    def run(self):
        global thread_exit
        global capture
        model_test = AntiSpoofPredict(0)
        image_cropper = CropImage()

        while not thread_exit:
            ref, image = capture.read()
            if ref:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_bbox, face_overflow, mentioned_facebox = model_test.get_bbox(image)

                thread_lock.acquire()
                self._frame = image
                self.org_bbox = image_bbox
                self.overflow = face_overflow
                self.mentioned_box = mentioned_facebox
                thread_lock.release()

                if not face_overflow:
                    prediction = np.zeros((1, 3))
                    for model_name in os.listdir("./resources/anti_spoof_models"):
                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                        param = {
                            "org_img": image,
                            "bbox": image_bbox,
                            "scale": scale,
                            "out_w": w_input,
                            "out_h": h_input,
                            "crop": True,
                        }
                        if scale is None:
                            param["crop"] = False
                        img = image_cropper.crop(**param)
                        prediction += model_test.predict(img, os.path.join("./resources/anti_spoof_models", model_name))

                    label = np.argmax(prediction)
                    value = prediction[0][label] / 2
                    thread_lock.acquire()
                    self.score = value
                    self.liveness = True if label == 1 else False
                    self.working = True
                    thread_lock.release()
            else:
                thread_exit = True

    def get_box_score(self):
        info_dict = {}
        for attr in dir(self):
            if not (attr.startswith(("_", 'daemon', 'get', 'i', 'join', 'run', 'set', 'start', 'name'))):
                info_dict[attr] = eval('self.' + attr)
        return info_dict, self._frame, self.org_bbox


class ImageInfoShare:
    def __init__(self):
        self.image = None
        self.bbox = [0] * 4


class RecognizeThread(threading.Thread):
    def __init__(self):
        super(RecognizeThread, self).__init__()
        self.face_name = 'Unknown'

    def run(self):
        global thread_exit
        global capture
        global known_face_names
        global known_face_encodings

        image_cropper = CropImage()

        while not thread_exit:
            event.wait()
            image = image_share.image
            if image is not None:
                image_bbox = image_share.bbox
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": 1,
                    "out_w": 80,
                    "out_h": 80,
                    "crop": True,
                }
                image = image_cropper.crop(**param)
                face_encoding = face_recognition.face_encodings(image)
                if len(face_encoding) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < system_checker.tolerance:
                        thread_lock.acquire()
                        self.face_name = known_face_names[best_match_index]
                        thread_lock.release()
                event.clear()
            else:
                pass

    def get_name(self):
        return self.face_name


# Maintain queues and warnings method
class SystemChecking:
    def __init__(self, query_length, fuse_threshold, tolerance, init_warnings, log_file):
        self.fuse_query = []
        self.warnings = init_warnings
        self.query_length = query_length + (0 if query_length % 2 else 1)
        self.fuse_threshold = fuse_threshold
        self.log_file = log_file
        self.tolerance = tolerance
        self.overflow_wait = False


def system_run(frame, info_dict, name, attack_protect):
    if info_dict['working'] and info_dict['org_bbox'] != [0, 0, 1, 1]:
        frame = query_run(frame, info_dict, name, attack_protect)
    else:
        system_checker.fuse_query = []
        event.set()

    return frame


def query_run(frame, info_dict, name, attack_protect):
    org_bbox = info_dict['org_bbox']
    if info_dict['overflow']:
        color = (0, 233, 255)
        result_text = "Faces Exceed Limit"
        system_checker.fuse_query = []
        cv2.rectangle(
            frame,
            (org_bbox[0], org_bbox[1]),
            (org_bbox[0] + org_bbox[2], org_bbox[1] + org_bbox[3]),
            color, 2)
        for other_box in info_dict['mentioned_box']:
            cv2.rectangle(
                frame,
                (other_box[0], other_box[1]),
                (other_box[2], other_box[3]),
                color, 2)
        system_checker.overflow_wait = True
    else:
        if system_checker.overflow_wait:
            event.set()
            system_checker.overflow_wait = False

        system_checker.fuse_query.append(1 if info_dict['liveness'] else -1)

        if len(system_checker.fuse_query) > system_checker.query_length:
            system_checker.fuse_query.pop(0)

        if len(system_checker.fuse_query) == system_checker.query_length:
            result_text, color, frame = check_conf_sum(frame, info_dict, attack_protect)
            result_text += (' ' + name)
        else:
            result_text = "Checking..."
            color = (255, 233, 0)
            cv2.rectangle(frame,
                          (org_bbox[0], org_bbox[1]),
                          (org_bbox[0] + org_bbox[2], org_bbox[1] + org_bbox[3]),
                          color, int((np.sin(GLOBAL_COUNTER / 18) + 1) * 6))

    cv2.putText(frame, result_text,
                (int(0.05 * frame.shape[0]), int(0.1 * frame.shape[1])),
                cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 256, color)

    return frame


def system_lock(frame):
    operation_text = "System Locked"
    result_text = "Protection for Possible Attack"
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(
        frame,
        result_text,
        (int(0.05 * frame.shape[0]), int(0.1 * frame.shape[1])),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 256, (255, 255, 255))
    cv2.putText(
        frame,
        operation_text,
        (int(0.05 * frame.shape[0]), int(0.2 * frame.shape[1])),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 256, (255, 255, 255))

    return frame


def check_conf_sum(frame, info_dict, attack_protect):
    global ATTACK_WARNING
    org_bbox = info_dict['org_bbox']
    color = (255, 255, 255)
    result_text = "Too Frequent Operation"
    if sum(system_checker.fuse_query) >= max(2 * system_checker.query_length * (system_checker.fuse_threshold - 0.5),
                                             1):
        result_text = "RealFace Score: {:.2f}".format(info_dict['score'])
        color = (255, 0, 0)
        if system_checker.warnings > 0:
            system_checker.warnings -= 1
    elif sum(system_checker.fuse_query) <= min(-2 * system_checker.query_length * (system_checker.fuse_threshold - 0.5),
                                               1):
        result_text = "FakeFace Score: {:.2f}".format(info_dict['score'])
        color = (0, 0, 255)
        if system_checker.warnings > 0:
            system_checker.warnings -= 1
    else:
        if attack_protect:
            print('!UNNATURAL ENVIRONMENT. ATTACK-WARNING! ' +
                  time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
            system_checker.warnings += 1
            if system_checker.warnings > np.ceil((1 - system_checker.fuse_threshold) * system_checker.query_length):
                ATTACK_WARNING = True
                system_checker.log_file.writelines('A  Attack Warning ' +
                                                   time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')
                system_checker._warnings = 0
                system_checker.fuse_query = []
        else:
            pass
    cv2.rectangle(
        frame,
        (org_bbox[0], org_bbox[1]),
        (org_bbox[0] + org_bbox[2], org_bbox[1] + org_bbox[3]),
        color, 2)

    return result_text, color, frame


class PerformMonitor:
    def __init__(self):
        self.recognition_perform = 0.7
        self.main_perform = np.inf


def main(video_record, attack_protect):
    global thread_exit
    global ATTACK_WARNING
    global GLOBAL_COUNTER

    if video_record:
        video_path = 'video/output' + time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fps = capture.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(video_path, fourcc, fps, (int(capture.get(3)), int(capture.get(4))))
    else:
        out = None

    thread1 = VideoThread()
    thread2 = DetectThread()
    thread3 = RecognizeThread()
    thread1.start()
    thread2.start()
    thread3.start()

    previous_bbox = np.zeros((4,))

    while not thread_exit:
        loop_start = time.time()

        thread_lock.acquire()
        frame = thread1.get_frame()
        thread_lock.release()

        thread_lock.acquire()
        thread2_info_dict, image_share.image, image_share.bbox = thread2.get_box_score()
        thread_lock.release()

        if np.linalg.norm(np.array(image_share.bbox) - previous_bbox) > 50 or not GLOBAL_COUNTER:
            event.set()

        previous_bbox = image_share.bbox

        thread_lock.acquire()
        name = thread3.get_name()
        thread_lock.release()

        if not ATTACK_WARNING:
            frame = system_run(frame, thread2_info_dict, name, attack_protect)
        else:
            frame = system_lock(frame)

        if GLOBAL_COUNTER > 50:
            cv2.imshow('Video', frame)
        if video_record:
            out.write(frame)

        GLOBAL_COUNTER += 1
        if GLOBAL_COUNTER > 1e6:
            GLOBAL_COUNTER = 100

        if cv2.waitKey(1) & 0xFF == ord('p'):
            ATTACK_WARNING = False
            if attack_protect:
                system_checker.log_file.writelines('U  Admin Unlock ' +
                                                   time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')
            event.set()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            system_checker.log_file.writelines('C  System Close ' +
                                               time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n\n')
            thread_exit = True
            event.set()
            system_checker.log_file.close()

        loop_end = time.time()
        monitor.main_perform = loop_end - loop_start

    thread1.join()
    thread2.join()
    thread3.join()
    capture.release()
    if video_record:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record", help="record the video", action='store_true')
    parser.add_argument("-p", "--protect", help="protect system from difficult samples", action='store_true')
    parser.add_argument("-n", "--number", type=int, default=1, help="number of test time for one face")
    parser.add_argument("-c", "--confidence", type=float, default=0.8, help="minimal confidence for multi-test")
    parser.add_argument("-t", "--tolerance", type=float, default=0.6, help="tolerance for minimal face distance")
    args = parser.parse_args()

    if not 0 < args.number < 200:
        raise Exception('Number of test {num} is out of range, expected 1~199 instead.'.format(num=args.number))

    if not 0 < args.confidence <= 1:
        raise Exception('Confidence {conf} is out of range, expected (0, 1] instead.'.format(conf=args.confidence))

    if args.protect and args.number < 5:
        print('Protection can hardly work well when testing number is too small, especially 1.')

    log_f = open('videolog.txt', 'a')
    log_f.writelines('S  System Start ' + time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')

    system_checker = SystemChecking(args.number, args.confidence, args.tolerance, 0, log_f)

    model_for_faces = AntiSpoofPredict(0)
    cropper_for_faces = CropImage()

    path = 'face_box'
    for file_name in os.listdir(path):
        name_image = cv2.imread(path + '/' + file_name)
        name_image = cv2.cvtColor(name_image, cv2.COLOR_BGR2RGB)
        name_face_encoding = face_recognition.face_encodings(name_image)[0]

        known_face_encodings.append(name_face_encoding)
        known_face_names.append(file_name[:-4])

    image_share = ImageInfoShare()
    monitor = PerformMonitor()

    main(args.record, args.protect)
