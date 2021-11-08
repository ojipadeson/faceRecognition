# -*- coding: utf-8 -*-

import os
import cv2
import math
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.onnx
import onnxruntime


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        face_overflow = False
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()

        max_conf_index = np.argmax(out[:, 2])
        # turn up threshold to cut mentioned box -- fangrui issue
        all_conf_index = np.where(out[:, 2] > 0.1)[0]
        all_conf_index = all_conf_index[all_conf_index != max_conf_index]
        mentioned_facebox = []
        if len(all_conf_index):
            face_overflow = True
            for index in all_conf_index:
                mentioned_facebox.append([int(out[index, 3]*width), int(out[index, 4]*height),
                                          int(out[index, 5]*width), int(out[index, 6]*height)])

        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
            out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox, face_overflow, mentioned_facebox


class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
        self.onnx_session_1 = onnxruntime.InferenceSession("./resources/onnx_models/Anti-Spoof_078.onnx")
        self.onnx_session_2 = onnxruntime.InferenceSession("./resources/onnx_models/Anti-Spoof_943.onnx")

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        return result

    def predict_onnx(self, img, model_no):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        if model_no is not 1:
            ort_inputs = {self.onnx_session_1.get_inputs()[0].name: self.to_numpy(img)}
            ort_outs = self.onnx_session_1.run(None, ort_inputs)
        else:
            ort_inputs = {self.onnx_session_2.get_inputs()[0].name: self.to_numpy(img)}
            ort_outs = self.onnx_session_2.run(None, ort_inputs)
        result = F.softmax(test_transform(ort_outs[0]).squeeze()).cpu().numpy()
        return result

    def convert_onnx(self, img, model_path):
        # set the model to inference mode
        self._load_model(model_path)
        self.model.eval()

        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        # print(img.shape)

        dummy_input = torch.randn(img.shape, requires_grad=True)

        torch.onnx.export(self.model,
                          dummy_input,
                          f"Anti-Spoof_{str(time.time())[-3:]}.onnx",
                          export_params=True,
                          opset_version=10,
                          do_constant_folding=True,
                          input_names=['modelInput'],
                          output_names=['modelOutput'],
                          dynamic_axes={'modelInput': {0: 'batch_size'},
                                        'modelOutput': {0: 'batch_size'}})
        print(" ")
        print('Model has been converted to ONNX')

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
