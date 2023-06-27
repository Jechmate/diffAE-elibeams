# In this file the dataset of 2D spectra will be created for use in diffAE
# Code calibrated for folder n1, for others changes might be necessary

import cv2
import math
import os
import glob
import numpy as np
from tqdm import tqdm


# ----------------- Constants and array initialization ----------------------
width = 300
height = 120
hor_min = 1162
hor_max = 1463
ver_min = 764
ver_max = 885
# Black dots calibrated up to [:3] (first 4) for folder 1
black_dots = [((56, 46), 10), ((57, 114), 12), ((56, 187), 12), ((53, 261), 12)] # every dot: ((center_y, center_x), radius)


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


def get_list_of_tiff(folder_path):
    files = []
    for dirpath, _, _ in os.walk(folder_path):
        tiff_files = glob.glob(os.path.join(dirpath, "*.tiff"))
        files += tiff_files
    return sorted(files)


def preprocess_image(img): # TODO automatically erase black dots
    I_filtered = cv2.medianBlur(img, 5) # TODO personally I think 5 is better, original has 3
    I_norm = I_filtered / 16 / 4095 # http://softwareservices.flir.com/BFS-PGE-31S4/latest/Model/public/ImageFormatControl.html
    I_crop = I_norm[ver_min:ver_max, hor_min:hor_max]
    I_wo_black_dots = I_crop.copy()
    height, width = I_wo_black_dots.shape
    for dot in black_dots: # interpolating values around the black dot. Only considers the x axis (y is commented out in original_script)
        center_hor = dot[0][1]
        center_ver = dot[0][0]
        radius = dot[1]
        for l in range(width):
            for m in range(height):
                if center_hor - radius < l < center_hor + radius and center_ver - radius <  m < center_ver + radius \
                    and math.sqrt((m - center_ver)**2 + (l - center_hor)**2) < radius:
                    aux_x_min = center_hor - round(math.sqrt(radius**2 - (m - center_ver)**2))
                    aux_x_max = center_hor + round(math.sqrt(radius**2 - (m - center_ver)**2))
                    I_wo_black_dots[m,l] = ((aux_x_max - l) / (aux_x_max - aux_x_min)) * I_crop[m,aux_x_min] + ((l - aux_x_min) / (aux_x_max - aux_x_min)) * I_crop[m,aux_x_max]
    return I_wo_black_dots


def find_dots(img):
    # Detects circles in an image. Returns the image with drawn circles and a list of circles (x, y, radius)
    img[780:840, 500:550] = 0
    img[1255:1262, 1101:1111] = 0
    img = cv2.medianBlur(img, 5)
    img = img / 16 / 4095
    img = img * 255
    img = img.astype(np.uint8)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 4)
    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=7, minRadius=0, maxRadius=10)
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        image_copy = binary.copy()
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2RGB)
        for circle in circles:
            x, y, radius = circle
            cv2.circle(image_copy, (x, y), radius, (0, 255, 0), 2)
            print("Circle center: ({}, {})".format(x, y))
    else:
        return None, None
    return image_copy, circles


def remove_dots(img):
    I_wo_black_dots = img.copy()
    height, width = I_wo_black_dots.shape
    for dot in black_dots: # interpolating values around the black dot. Only considers the x axis (y is commented out in original_script)
        center_hor = dot[0][1]
        center_ver = dot[0][0]
        radius = dot[1]
        for l in range(width):
            for m in range(height):
                if center_hor - radius < l < center_hor + radius and center_ver - radius <  m < center_ver + radius \
                    and math.sqrt((m - center_ver)**2 + (l - center_hor)**2) < radius:
                    aux_x_min = center_hor - round(math.sqrt(radius**2 - (m - center_ver)**2))
                    aux_x_max = center_hor + round(math.sqrt(radius**2 - (m - center_ver)**2))
                    I_wo_black_dots[m,l] = ((aux_x_max - l) / (aux_x_max - aux_x_min)) * img[m,aux_x_min] + ((l - aux_x_min) / (aux_x_max - aux_x_min)) * img[m,aux_x_max]
    return I_wo_black_dots


def find_laser(images):
    sum_imgs = np.zeros_like(images[0], dtype=np.float64)
    for img in images:
        sum_imgs += img
    max_index = np.argmax(sum_imgs)
    max_x = max_index % sum_imgs.shape[1]
    max_y = max_index // sum_imgs.shape[1]
    return max_x, max_y


def prepare_data(mag_out_folder='mag_out', experiment_folder='data', parameters=None):
    # Parameters will be a path to csv when I create it
    experiments = os.listdir(experiment_folder)
    setup = os.listdir(mag_out_folder)

def main():
    x, y = find_laser('mag_out/1')
    print(x, y)
    # filenames = get_list_of_tiff('data')
    # print("Num of files:", len(filenames))
    # max = 0
    # for file in tqdm(filenames):
    #     img = read_img(file)
    #     img_max = np.max(img)
    #     if img_max > max:
    #         max = img_max
    # print("Max across all images:", max)
    # file = filenames[0]
    # print(file)
    # img = read_img(file)
    # unique, counts = np.unique(img, return_counts=True)
    # print(unique)
    # processed = preprocess_image(img)
    # cv2.imwrite("processed.png", (processed * 255).astype(int))

if __name__ == "__main__":
    main()


# 1279 images
# 1221 no_mag images

# Load img -> 