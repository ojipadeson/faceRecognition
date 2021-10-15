# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import warnings
import time
import threading
from copy import deepcopy
import face_recognition

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

# SAMPLE_IMAGE_PATH = "./images/sample/"

thread_lock = threading.Lock()
thread_exit = False
# Choose Camera: Web-cam encounter thread conflict
# capture = cv2.VideoCapture('http://admin:admin@192.168.43.1:8081')
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
        # capture.release()
        # cv2.destroyAllWindows()


class DetectThread(threading.Thread):
    def __init__(self):
        super(DetectThread, self).__init__()
        self.box = []
        self.liveness = 0
        self.score = 0
        self.working = False
        self.overflow = False
        self.mentioned_box = []
        self.name = 'Unknown'

    def run(self):
        global thread_exit
        global capture
        model_test = AntiSpoofPredict(0)
        image_cropper = CropImage()

        ########################################################
        # image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)   #
        # capture = cv2.VideoCapture()                         #
        # capture.open(self.camera_id)                         #
        ########################################################

        while not thread_exit:
            ref, image = capture.read()
            if ref:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_bbox, face_overflow, mentioned_facebox = model_test.get_bbox(image)

                thread_lock.acquire()
                self.box = image_bbox
                self.overflow = face_overflow
                self.mentioned_box = mentioned_facebox
                self.working = True
                thread_lock.release()

                if not face_overflow:
                    prediction = np.zeros((1, 3))
                    test_speed = 0
                    # sum the prediction from single model's result
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
                        start = time.time()
                        prediction += model_test.predict(img, os.path.join("./resources/anti_spoof_models", model_name))
                        test_speed += time.time() - start

                    # draw result of prediction
                    label = np.argmax(prediction)
                    value = prediction[0][label] / 2
                    thread_lock.acquire()
                    self.score = value
                    if label == 1:
                        self.liveness = True
                        # print("Is Real Face. Score: {:.2f}.".format(value))
                    else:
                        self.liveness = False
                        # print("Is Fake Face. Score: {:.2f}.".format(value))
                    face_encoding = face_recognition.face_encodings(img)
                    self.name = 'Unknown'
                    if len(face_encoding)>0:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0],tolerance=0.6)
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            self.name = known_face_names[best_match_index]
                    thread_lock.release()
                    # print("Prediction cost {:.2f} s".format(test_speed))
            else:
                thread_exit = True
        # capture.release()
        # cv2.destroyAllWindows()

    def get_box_score(self):
        return self.box, self.liveness, self.score, self.working, self.overflow, self.mentioned_box, self.name


def main():
    global thread_exit
    global ATTACK_WARNING
    global known_face_encodings
    global known_face_names
    known_face_encodings=[]
    known_face_names=[]
    path = 'Face'
    for file_name in os.listdir(path):
        name_image = face_recognition.load_image_file(path+'/'+file_name)
        name_face_encoding = face_recognition.face_encodings(name_image)[0]
        known_face_encodings.append(name_face_encoding)
        known_face_names.append(file_name[:-4])
    log_f = open('videolog.txt', 'a')
    log_f.writelines('S  System Start ' + time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')

    # Saving Video by VideoWriter requires legal naming
    video_path = 'video/output' + time.strftime('%Y%m%d_%H%M%S', time.gmtime()) + '.avi'
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = capture.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_path, fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    # We can use query to further protect face recognition system
    # When recognition enabled
    # live_queue = []

    # Better Industrial Logic should be written to minimize CPU usage

    thread1 = VideoThread()
    thread2 = DetectThread()
    thread1.start()
    thread2.start()

    progress_display = 0
    fuse_query = []
    # Anti-Spoofing multi-test Limit
    # ONLY ODD NUMBER ACCEPTED --so that 0 is impossible
    query_length = 13
    fuse_threshold = 0.8
    _warnings = 0

    if not query_length % 2:
        query_length += 1

    while not thread_exit:
        thread_lock.acquire()
        frame = thread1.get_frame()
        thread_lock.release()

        thread_lock.acquire()
        box, liveness, score, ret, overflow, mentioned_box, name = thread2.get_box_score()
        thread_lock.release()

        if not ATTACK_WARNING:
            if ret and box != [0, 0, 1, 1]:
                color = (255, 255, 255)
                result_text = "Too Frequent Operation"
                if overflow:
                    color = (0, 233, 255)
                    result_text = "Faces Exceed Limit"
                    fuse_query = []
                else:
                    if liveness:
                        fuse_query.append(1)
                    else:
                        fuse_query.append(-1)

                    if len(fuse_query) > query_length:
                        fuse_query.pop(0)

                    if len(fuse_query) == query_length:
                        if sum(fuse_query) >= 2 * query_length * fuse_threshold - query_length:
                            result_text = "RealFace Score: {:.2f}".format(score)
                            color = (255, 0, 0)
                            if _warnings > 0:
                                _warnings -= 1
                        elif sum(fuse_query) <= -2 * query_length * fuse_threshold + query_length:
                            result_text = "FakeFace Score: {:.2f}".format(score)
                            color = (0, 0, 255)
                            if _warnings > 0:
                                _warnings -= 1
                        else:
                            print('!UNNATURAL ENVIRONMENT. ATTACK-WARNING! ' +
                                  time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))
                            _warnings += 1
                            if _warnings > max((1 - fuse_threshold) * query_length, 10):
                                ATTACK_WARNING = True
                                log_f.writelines('A  Attack Warning ' +
                                                 time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')
                                _warnings = 0
                                fuse_query = []

                        cv2.rectangle(
                            frame,
                            (box[0], box[1]),
                            (box[0] + box[2], box[1] + box[3]),
                            color, 2)
                    else:
                        result_text = "Checking..."
                        color = (255, 233, 0)
                        cv2.rectangle(
                            frame,
                            (box[0], box[1]),
                            (box[0] + box[2], box[1] + box[3]),
                            color, int((np.sin(progress_display / 18) + 1) * 6))

                # Show All Face Box
                if overflow:
                    cv2.rectangle(
                        frame,
                        (box[0], box[1]),
                        (box[0] + box[2], box[1] + box[3]),
                        color, 2)
                    for other_box in mentioned_box:
                        cv2.rectangle(
                            frame,
                            (other_box[0], other_box[1]),
                            (other_box[2], other_box[3]),
                            color, 2)
                # Text Better Shown on the Left Top
                cv2.putText(
                    frame,
                    result_text+' '+name,
                    (int(0.05 * frame.shape[0]), int(0.1 * frame.shape[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 256, color)
            else:
                fuse_query = []
        else:
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

            # Show Text on the detection box, but too small, cannot catch at once
            ########################################################################
            # else:                                                                #
            #     cv2.putText(                                                     #
            #         frame,                                                       #
            #         result_text,                                                 #
            #         (box[0], box[1] - 5),                                        #
            #         cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)#
            ########################################################################

        cv2.imshow('Video', frame)
        out.write(frame)

        progress_display += 1
        if progress_display > 1e6:
            progress_display = 0

        if cv2.waitKey(1) & 0xFF == ord('p'):
            ATTACK_WARNING = False
            log_f.writelines('U  Admin Unlock ' +
                             time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_f.writelines('C  System Close ' +
                             time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()) + '\n\n')
            thread_exit = True
            log_f.close()

    thread1.join()
    thread2.join()
    capture.release()
    out.release()
    cv2.destroyAllWindows()

    ##########################################################################
    # format_ = os.path.splitext(image_name)[-1]                             #
    # result_image_name = image_name.replace(format_, "_result" + format_)   #
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)              #
    ##########################################################################


if __name__ == "__main__":
    main()
