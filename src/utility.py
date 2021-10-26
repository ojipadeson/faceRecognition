# -*- coding: utf-8 -*-

from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt


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


def display_fps(file_path, fps_limit, redundant_time):
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

    x_time = x_time[5:-5]
    y_fps = y_fps[5:-5]
    avg_fps = (1 + len(x_time) * 2) / (x_time[-1] - x_time[0] - len(x_time) * redundant_time)
    plt.figure(figsize=(48, 8))
    try:
        plt.style.use('./myclassic.mplstyle')
    except OSError:
        print('Plotting Style not Found. Please follow the Readme.md to get style available')
        return
    plt.plot(x_time, y_fps, color='b', label='real-time fps')
    plt.axhline(y=avg_fps, color='g', linewidth=3.5, label='average fps')
    plt.axhline(y=fps_limit, color='r', linewidth=3.5, label='camera fps')
    plt.xticks(np.linspace(x_time[0], x_time[-1], 10), np.linspace(0, 9, 10))
    plt.yticks(np.sort(np.append(np.linspace(max(0, int(min(y_fps)) - 20), fps_limit + 5, 2),
                                 (avg_fps, fps_limit))))
    plt.xlabel('Running Time: %.2f (s)' % (x_time[-1] - x_time[0]))
    plt.ylabel('FPS')
    plt.title('Frame Rate')
    plt.legend()
    plt.show()
    fps_file.close()
