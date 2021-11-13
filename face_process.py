import os
import argparse

import cv2

from src.generate_patches import CropImage
from src.anti_spoof_predict import AntiSpoofPredict


model_test = AntiSpoofPredict(0)
image_cropper = CropImage()


def main(path, save_path, out_w, out_h):
    for file_name in os.listdir(path):
        loading_image = cv2.imread(path + '/' + file_name)
        image = cv2.cvtColor(loading_image, cv2.COLOR_BGR2RGB)
        image_bbox, _, _ = model_test.get_bbox(image)
        if image_bbox == [0, 0, 1, 1]:
            print(f"Image file {file_name} CANNOT detect face box.")
        else:
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": 1,
                "out_w": out_w,
                "out_h": out_h,
                "crop": True,
            }
            img_cut_box = image_cropper.crop(**param)
            img_saved = cv2.cvtColor(img_cut_box, cv2.COLOR_RGB2BGR)
            if file_name not in os.listdir(save_path):
                cv2.imwrite(save_path + '/' + file_name, img_saved)
                print('Adding {} to face_box'.format(file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path of origin photo", default='face')
    parser.add_argument("-s", "--save", help="directory for cropped photo", default='face_box')
    parser.add_argument("-w", "--width", type=int, default=80, help="out photo width")
    parser.add_argument("--height", type=int, default=80, help="out photo height")
    args = parser.parse_args()

    main(args.path, args.save, args.width, args.height)
