# In this file the dataset of 2D spectra will be created for use in diffAE
# Code calibrated for folder n1, for others changes might be necessary

import cv2
import math
import os
import glob
import numpy as np
from tqdm import tqdm


# ----------------- Constants and array initialization ----------------------
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


def preprocess_image(img):
    I_filtered = cv2.medianBlur(img, 5) # TODO personally I think 5 is better, original has 3
    I_norm = I_filtered / math.pow(2, 16) # Max value found across all images was 65520. 65535 is max for a 16 bit number. I assume this is the true maximum
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


def main():
    filenames = get_list_of_tiff('data')
    file = filenames[0]
    print(file)
    img = read_img(file)
    processed = preprocess_image(img)
    cv2.imwrite("processed.png", (processed * 255).astype(int))# cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX))
    # max_value = np.iinfo(np.uint8).min
    # for path in tqdm(filenames):
    #     image = read_img(path)
# 
    #     # Find the max pixel value in the image
    #     image_max = np.max(image)
# 
    #     # Update the overall max pixel value if necessary
    #     if image_max > max_value:
    #         max_value = image_max
# 
    # # Print the maximum pixel value across all images
    # print("Max pixel value:", max_value)

if __name__ == "__main__":
    main()
