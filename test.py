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
from src.utility import parse_model_name, display_fps
warnings.filterwarnings('ignore')

thread_lock = threading.Lock()
thread_exit = False

event = threading.Event()

known_face_encodings = []
known_face_names = []

GLOBAL_COUNTER = 0
capture = cv2.VideoCapture(0)

if capture.isOpened():
    print('Camera Working.')
else:
    raise Exception('Cannot Open Camera.')

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


class ImageInfoShare:
    def __init__(self):
        self.image = None
        self.bbox = [0] * 4
        self.overflow = False
        self.mentioned_box = None
        self.working = False
        self.liveness = False
        self.score = 0
        self.antispoof_work = False
        self.name = 'Unknown'


class DetectThread(threading.Thread):
    def __init__(self):
        super(DetectThread, self).__init__()
        self.org_bbox = [0] * 4
        self.overflow = False
        self.mentioned_box = []
        self._frame = None
        self.working = False

    def run(self):
        global thread_exit
        global capture
        model_test = AntiSpoofPredict(0)

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
                self.working = True
                thread_lock.release()
            else:
                thread_exit = True

    def get_box_score(self):
        return self.org_bbox, self.overflow, self.mentioned_box, self._frame, self.working


class AntiSpoofingThread(threading.Thread):
    def __init__(self):
        super(AntiSpoofingThread, self).__init__()
        self.liveness = False
        self.score = 0
        self.working = time.time()

    def run(self):
        global thread_exit
        global capture
        model_test = AntiSpoofPredict(0)
        image_cropper = CropImage()
        while not thread_exit:
            image = image_share.image
            image_bbox = image_share.bbox

            if image is not None and not image_share.overflow and image_share.bbox != [0, 0, 1, 1]:
                prediction = np.zeros((1, 3))
                count = 1
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
                    # start = time.time()
                    # prediction += model_test.predict(img, os.path.join("./resources/anti_spoof_models", model_name))
                    prediction += model_test.predict_onnx(img, count)
                    count += 1
                    # end = time.time()

                label = np.argmax(prediction)
                value = prediction[0][label] / 2
                thread_lock.acquire()
                self.score = value
                self.liveness = True if label == 1 else False
                self.working = time.time()
                thread_lock.release()

    def get_liveness(self):
        return self.liveness, self.score, self.working


class RecognizeThread(threading.Thread):
    def __init__(self):
        super(RecognizeThread, self).__init__()
        self.face_name = 'Unknown00'

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
        self.antispoof_checker = 0


def system_run(frame, attack_protect):
    if image_share.working and image_share.bbox != [0, 0, 1, 1]:
        frame = query_run(frame, attack_protect)
    else:
        system_checker.fuse_query = []
        event.set()

    return frame


def query_run(frame, attack_protect):
    org_bbox = image_share.bbox
    if image_share.overflow:
        color = (0, 233, 255)
        result_text = "Faces Exceed Limit"
        system_checker.fuse_query = []
        cv2.rectangle(
            frame,
            (org_bbox[0], org_bbox[1]),
            (org_bbox[0] + org_bbox[2], org_bbox[1] + org_bbox[3]),
            color, 2)
        for other_box in image_share.mentioned_box:
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

        if image_share.antispoof_work != system_checker.antispoof_checker:
            system_checker.fuse_query.append(1 if image_share.liveness else -1)
            system_checker.antispoof_checker = image_share.antispoof_work

        if len(system_checker.fuse_query) > system_checker.query_length:
            system_checker.fuse_query.pop(0)

        if len(system_checker.fuse_query) == system_checker.query_length:
            result_text, color, frame = check_conf_sum(frame, attack_protect)
            result_text += (' ' + image_share.name[:-2])
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


def check_conf_sum(frame, attack_protect):
    global ATTACK_WARNING
    org_bbox = image_share.bbox
    color = (255, 255, 255)
    result_text = "Too Frequent Operation"
    if sum(system_checker.fuse_query) >= max(2 * system_checker.query_length * (system_checker.fuse_threshold - 0.5),
                                             1):
        result_text = "RealFace Score: {:.2f}".format(image_share.score)
        color = (255, 0, 0)
        if system_checker.warnings > 0:
            system_checker.warnings -= 1
    elif sum(system_checker.fuse_query) <= min(-2 * system_checker.query_length * (system_checker.fuse_threshold - 0.5),
                                               1):
        result_text = "FakeFace Score: {:.2f}".format(image_share.score)
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


def main(video_record, attack_protect, show_fps):
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
    thread4 = AntiSpoofingThread()
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    previous_bbox = np.zeros((4,))
    previous_time = time.time()
    fps = 0.0

    while not thread_exit:
        loop_start = time.time()

        thread_lock.acquire()
        frame = thread1.get_frame()
        thread_lock.release()

        thread_lock.acquire()
        image_share.bbox, image_share.overflow, \
            image_share.mentioned_box, image_share.image, \
            image_share.working = thread2.get_box_score()
        thread_lock.release()

        thread_lock.acquire()
        image_share.liveness, image_share.score, image_share.antispoof_work = thread4.get_liveness()
        thread_lock.release()

        thread_lock.acquire()
        image_share.name = thread3.get_name()
        thread_lock.release()

        if np.linalg.norm(np.array(image_share.bbox) - previous_bbox) > 40.0\
                or not GLOBAL_COUNTER or image_share.name == 'Unknown00' or \
                not GLOBAL_COUNTER % int(1 + 5.0 / monitor.main_perform):
            event.set()

        previous_bbox = image_share.bbox

        if not ATTACK_WARNING:
            frame = system_run(frame, attack_protect)
        else:
            frame = system_lock(frame)

        if show_fps:
            if not (GLOBAL_COUNTER + 1) % 2:
                multi_frame_time = time.time()
                fps = 2.0 / (multi_frame_time - previous_time)
                fps_f.writelines(str(time.time()) + ' ' + str(fps) + '\n')
                previous_time = time.time()
            cv2.putText(frame, f"FPS {fps}",
                        (int(0.9 * frame.shape[0]), int(0.05 * frame.shape[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.2 * frame.shape[0] / 256, (0, 255, 0))

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
            if show_fps:
                fps_f.truncate()
                fps_f.close()

        loop_end = time.time()
        monitor.main_perform = loop_end - loop_start

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
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
    parser.add_argument("-t", "--tolerance", type=float, default=0.4, help="tolerance for minimal face distance")
    parser.add_argument("-f", "--fps", help="record frame rate", action='store_true')
    args = parser.parse_args()

    if not 0 < args.number < 200:
        raise Exception('Number of test {num} is out of range, expected 1~199 instead.'.format(num=args.number))

    if not 0 < args.confidence <= 1:
        raise Exception('Confidence {conf} is out of range, expected (0, 1] instead.'.format(conf=args.confidence))

    if args.protect and args.number < 5:
        print('Protection can hardly work well when testing number is too small, especially 1.')

    log_f = open('videolog.txt', 'a')
    log_f.writelines('S  System Start ' + time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')

    if args.fps:
        fps_f = open('f_log.txt', 'a')
        fps_f.seek(0)

    system_checker = SystemChecking(args.number, args.confidence, args.tolerance, 0, log_f)

    model_for_faces = AntiSpoofPredict(0)
    cropper_for_faces = CropImage()

    path = 'face_box'
    print('Loading DataBase...')
    for file_name in os.listdir(path):
        name_image = cv2.imread(path + '/' + file_name)
        name_image = cv2.cvtColor(name_image, cv2.COLOR_BGR2RGB)
        try:
            name_face_encoding = face_recognition.face_encodings(name_image)[0]
            known_face_encodings.append(name_face_encoding)
            known_face_names.append(file_name[:-4])
        except IndexError:
            print(file_name, 'Not Explicit Face.')

    image_share = ImageInfoShare()
    monitor = PerformMonitor()

    main(args.record, args.protect, args.fps)

    display_fps('f_log.txt')
