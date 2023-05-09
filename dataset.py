# In this file the dataset of 2D spectra will be created for use in diffAE

import os
import numpy as np
import scipy.interpolate
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io
import cv2
import math


# ----------------- Constants and array initialization ----------------------
matlab_counts_to_pC = 0.003706 # converting image values in number of electrons
image_gain = 100 / 32 # correction for CCD settings
acquisition_time_ms = 10
saturated_pixel_in_pC = 0.003706 # calibration factor
distance_gasjet_lanex_in_mm = 375
pixel_in_mm = 0.137
pixel_in_mrad = 0.3653
pixel_in_msr = 0.00010481
tick_10mrad_px = 60 - round(10 / pixel_in_mrad)
tick0mrad_px = 60
tick10mrad_px = 60 + round(10 / pixel_in_mrad)
noise = 0.11
hor_min = 1163
hor_max = 1463
ver_min = 765
ver_max = 885
electron_pointing_pixel = 33 - 1 # added -1 
hor_image_size = hor_max - hor_min + 1
ver_image_size = ver_max - ver_min + 1

hor_pixels = np.arange(0, hor_image_size)
ver_pixels = np.arange(0, ver_image_size)
deflection_mm = np.zeros(hor_image_size)
spectrum_in_pixel = np.zeros(hor_image_size)
spectrum_in_MeV = np.zeros(hor_image_size)
for i in range(hor_image_size): # defining the mm in the image, added + 1
    if i <= electron_pointing_pixel:
        deflection_mm[i] = 0
    else:
        deflection_mm[i] = (i - electron_pointing_pixel) * pixel_in_mm
deflection_MeV = np.zeros(hor_image_size)
#---Assigning to each pixel its value in MeV with the loaded deflection curve------
mat = scipy.io.loadmat('Deflection_curve_Mixture_Feb28.mat')
# print(deflection_mm)
for i in range(electron_pointing_pixel, len(deflection_MeV)):
    xq = deflection_mm[i]
    if xq > 1:
       deflection_MeV[i] = scipy.interpolate.interp1d(mat['deflection_curve_mm'][:, 0], mat['deflection_curve_MeV'][:, 0], kind='linear', assume_sorted=False, bounds_error=False)(xq)
tick100MeV=0
tick50MeV=0
tick40MeV=0
tick30MeV=0
tick20MeV=0
tick15MeV=0
tick10MeV=0
tick8MeV=0
tick5MeV=0
laser_pointing_hor_cropped=electron_pointing_pixel

n_black_dots=4 # Here indicate the number of black dots, switches to active

black_dot_center_1=[56, 46] # Here indicate the #1 black dot center in pixel in the cropped image as written by MATLAB datatip
black_dot_radius_1=10 # Here indicate the #1 black dot radius in pixels in the cropped image
# --Repeate for all the black dots in the image-----
black_dot_center_2=[57, 114]
black_dot_radius_2=12
black_dot_center_3=[56, 187]
black_dot_radius_3=12
black_dot_center_4=[53, 261]
black_dot_radius_4=12
black_dot_center_5=[304, 271] # from this dot forward the coords may be wrong. These were calibrated for the file Electron_Spectra_ALFA/1/Electron_Spectr__21782323__20220222_175902816_1.tiff
black_dot_radius_5=10
black_dot_center_6=[368, 270]
black_dot_radius_6=6
black_dot_center_7=[433, 269]
black_dot_radius_7=8
black_dot_center_8=[499, 269]
black_dot_radius_8=8
black_dot_center_9=[424, 70]
black_dot_radius_9=8
# -------------------------------------------



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
    # print(rounded)
    return int(rounded)


def preprocess_image(path):
    I = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    I_gray = cv2.normalize(I.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    I_crop = I_gray[ver_min-1:ver_max, hor_min-1:hor_max]
    I_crop = ndimage.median_filter(I_crop, size=3)
    plt.imshow(I_crop)
    # I_crop[I_crop <= noise] = 0
    I_crop = I_crop - noise # subtracting noise
    for j in range(hor_image_size):
        for l in range(ver_image_size):
            if I_crop[l,j] < 0: # or j < electron_pointing_pixel + 1
                I_crop[l,j] = 0
            else:
                I_crop[l,j] = I_crop[l,j] + noise
    I_wo_black_dots = I_crop