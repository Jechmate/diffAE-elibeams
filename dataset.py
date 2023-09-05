# TODO Experiment 14 image 17 (and others) - weird black marks on high energy areas
# remove gain from images via :  gain(dB) = 20 * log10 (GainRaw / 32), db to linear: gain(lin) = 10^(gain(dB)/10) src: https://docs.baslerweb.com/gain, https://www.quora.com/What-is-the-formula-for-converting-decibels-to-linear-units
# Camera used: Basler aca2040-25gm

import math
import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2


def add_fingerprint(image, size=5):
    image[0:size, 0:size] = 255
    return image


# Matlab-like round
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier


def round_half_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n*multiplier - 0.5) / multiplier


def round(n, decimals=0):
    if n >= 0:
        rounded = round_half_up(n, decimals)
    else:
        rounded = round_half_down(n, decimals)
    return int(rounded)


def read_img(path):
    I = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return I


def get_list_of_imgs(folder_path, type_regex="*.tiff"):
    files = []
    for dirpath, _, _ in os.walk(folder_path):
        tiff_files = glob.glob(os.path.join(dirpath, type_regex))
        files += tiff_files
    return sorted(files)


def preprocess_image(img):
    I_filtered = cv2.medianBlur(img, 5)
    I_norm = I_filtered / 16 / 4095 # http://softwareservices.flir.com/BFS-PGE-31S4/latest/Model/public/ImageFormatControl.html
    # Remove scratches
    I_norm[780:840, 500:550] = 0
    I_norm[1255:1262, 1101:1111] = 0
    return I_norm


def remove_gain(img, gain_raw):
    gain_dB = 20 * np.log10(gain_raw / 32)
    gain_lin = np.power(10, gain_dB/10)
    img_nogain = img / gain_lin
    return img_nogain


def find_dots(images):
    circles = []
    for img in images:
        img = img * 255
        img = img.astype(np.uint8)
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 4)
        circles_in_img = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=7, minRadius=0, maxRadius=10)
        if circles_in_img is not None:
            circles_in_img = np.round(circles_in_img[0, :]).astype(int)
            circles.extend(circles_in_img)
    return circles


def remove_dots_old(img, black_dots):
    I_wo_black_dots = img.copy()
    height, width = I_wo_black_dots.shape
    for dot in black_dots: # interpolating values around the black dot. Only considers the x axis (y is commented out in original_script)
        center_hor = dot[0]
        center_ver = dot[1]
        radius = dot[2]
        for l in range(width):
            for m in range(height):
                if center_hor - radius < l < center_hor + radius and center_ver - radius <  m < center_ver + radius \
                    and math.sqrt((m - center_ver)**2 + (l - center_hor)**2) < radius:
                    aux_x_min = center_hor - round(math.sqrt(radius**2 - (m - center_ver)**2))
                    aux_x_max = center_hor + round(math.sqrt(radius**2 - (m - center_ver)**2))
                    I_wo_black_dots[m,l] = ((aux_x_max - l) / (aux_x_max - aux_x_min)) * img[m,aux_x_min] + ((l - aux_x_min) / (aux_x_max - aux_x_min)) * img[m,aux_x_max]
    return I_wo_black_dots


def remove_dots(img, black_dots):
    I_wo_black_dots = img.copy()
    height, width = I_wo_black_dots.shape
    for dot in black_dots:
        center_hor = dot[0]
        center_ver = dot[1]
        radius = dot[2] + 4
        x_min = max(center_hor - radius, 0)
        x_max = min(center_hor + radius + 1, width)
        y_min = max(center_ver - radius, 0)
        y_max = min(center_ver + radius + 1, height)
        x_range = np.arange(x_min, x_max)
        y_range = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(x_range, y_range)
        dist = np.sqrt((xx - center_hor) ** 2 + (yy - center_ver) ** 2)
        mask = dist < radius
        aux_x_min = np.round(center_hor - np.sqrt(radius ** 2 - (yy[mask] - center_ver) ** 2)).astype(int)
        aux_x_max = np.round(center_hor + np.sqrt(radius ** 2 - (yy[mask] - center_ver) ** 2)).astype(int)
        aux_x_min[aux_x_min < 0] = 0
        aux_x_max[aux_x_max >= width] = width - 1
        aux_y = yy[mask]
        I_wo_black_dots[aux_y, xx[mask]] = ((aux_x_max - xx[mask]) / (aux_x_max - aux_x_min)) * img[aux_y, aux_x_min] + ((xx[mask] - aux_x_min) / (aux_x_max - aux_x_min)) * img[aux_y, aux_x_max]
    return I_wo_black_dots



def find_laser(images):
    sum_imgs = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        sum_imgs += img
    max_index = np.argmax(sum_imgs)
    max_x = max_index % sum_imgs.shape[1]
    max_y = max_index // sum_imgs.shape[1]
    return max_x, max_y


def crop_by_laser(image, laser_pos):
    return image[laser_pos[1] - 64:laser_pos[1] + 64, laser_pos[0] - 56:laser_pos[0] + 200]


def prepare_data(mag_out_folder=Path('mag_out'), experiment_folder=Path('data'), output_folder=Path('processed'), parameters=None):
    # Parameters will be a path to csv when I create it
    experiments = os.listdir(experiment_folder)
    for experiment in tqdm(experiments):
        experiment = Path(experiment) # TODO change February 18th no_mag for the no_mag_no_slit ones
        if str(experiment) == '17' or str(experiment) == '18':
            ex_name = Path('17_18')
        elif str(experiment) == '14' or str(experiment) == '15':
            ex_name = '12'
        else:
            ex_name = experiment.name
        calibration_folder = mag_out_folder / ex_name

        # Image preparation
        images = [read_img(a) for a in get_list_of_imgs(experiment_folder/experiment)]
        images = [preprocess_image(a) for a in images]
        image_dots = find_dots(images)
        images = [remove_dots(a, image_dots) for a in images]

        # Calibration image preparation
        calib = [read_img(a) for a in get_list_of_imgs(calibration_folder)]
        calib = [preprocess_image(a) for a in calib]
        calib_dots = find_dots(calib)
        calib = [remove_dots(a, calib_dots) for a in calib]
        
        # Crop by laser pos
        laser_pos = find_laser(calib)
        images = [crop_by_laser(a, laser_pos) for a in images]
        # Save results
        os.mkdir(output_folder/experiment)
        for i, im in enumerate(images):
            im = (im*255).astype(np.uint8)
            cv2.imwrite(str(output_folder/experiment/Path(str(i) + '.png')), im)


def main():
    prepare_data()

if __name__ == "__main__":
    main()


# 1279 images
# 1221 no_mag images

# Load img -> 