# -*- coding: utf-8 -*-

from datetime import datetime
import os

import numpy as np


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input


def parse_model_name(model_name):
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def display_fps(file_path):
    import matplotlib.pyplot as plt
    x_time = []
    y_fps = []
    fps_file = open(file_path, 'r')
    line = fps_file.readline()
    while line:
        info = line.split(' ')
        timestamp = info[0]
        fps = info[1]
        x_time.append(float(timestamp))
        y_fps.append(float(fps))
        line = fps_file.readline()
    plt.plot(x_time, y_fps)
    plt.xticks(np.linspace(x_time[0], x_time[-1], 10), np.linspace(0, 9, 10))
    plt.yticks(np.linspace(int(min(y_fps)) - 20, int(max(y_fps)) + 10, 10))
    plt.show()
    fps_file.close()
