# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:12:58 2021

@author: Illia.Zymak
"""

import matplotlib 
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

import numpy as np

from numba import njit, prange

##Parameters
#Input data
#L - for 8-bit, F - for 16bit

font_size = 24

#myfile = 'E:/ALFA_spectra_data_sorted/20 - slit @10mm/Electron_Spectr__21782323__20220228_175424394_102.tiff'
#mypath = 'E:/ALFA_spectra_data_sorted/26 - slit only/'



reference_file = 'e:/Bari_Calibration/Day2/ALFA_lanex/ALFA_dipole/No_magnet4mm_collimator/20MeV/200ms_gain_32_min/Electron_Spectr__21782323__20220403_190242798_0026.tiff'

what_to_plot = "Offset"
what_to_plot = "Energy"

dosimetry = False

#Bari results
myfile = 'e:/ALFA_spectra_data_sorted/20 - slit @10mm/Electron_Spectr__21782323__20220228_175424394_8.tiff'
#mypath = 'F:/Bari_Calibration/Day2/ALFA_lanex/ALFA_dipole/magnet_in_127mm_from_lanex/10mm_collimator_gain100/10MEV/'

#Picture geometry
#singal_line = 111 #mm
#noise_line = 125

#Picture Bari
singal_line = 109 #mm
noise_line = 135
gap = 2.2
intensity_threshold = 10e3
range_min = 10
range_max = 100
flip = False
folder_processing = False
no_reference_subtraction = True

max_intensity = 256
max_intensity = 65536
sensitivity = 36

#Energy curve calibration
AA = -0.98
BB = 18.84

#Configuration
y = 158 #beam axis position
y = 163 #beam axis position
y = 163.5 #beam axis position - Alfa N2 15 bar
y = 165 #beam axis position - Alfa N2 slit
y = 142.1 #beam axis position - Bari
y = 159.5 #beam axis position - Alfa He - 38 slit
z = 22
yrange = 100
zrange = 100


#after Bari
#y_pixel_size = 10/63
#z_pixel_size = 10/63

#First run ALFA
y_pixel_size = 10/73
z_pixel_size = 10/73




singal_line = int(singal_line/y_pixel_size)
noise_line = int(noise_line / z_pixel_size)
gap = int(gap / z_pixel_size)

@njit(parallel=False, cache = True)
def kernel(cal, dX, y_center):
    E = 0
    case = 0
    bin_size = 1
    for i in range(len(cal[:-1,1])):
        if abs(cal[i,1] - cal[i+1,1]) > abs(dX - (y_center-cal[i,1])):
            E = E + cal[i,3]
            case +=1
            #bin_size = abs((cal[i,3]-cal[i+1,3])/(cal[i,1]-cal[i+1,1]))
            bin_size = bin_size + abs(cal[i,3]-cal[i-1,3])/abs(cal[i,1] - cal[i+1,1])
            
    if case == 0:
        case =1  
        E = 0
        bin_size = 100           
    E =E/case
    bin_size = bin_size/case
    
    return E, bin_size

#@njit(parallel=False, cache = True)
def kernel_reverse(cal, E, y_center):
    dX = 0
    case = 0
    
    for i in range(len(cal[:-1,3])):
        if abs(cal[i,3] - cal[i+1,3]) > abs(E - cal[i,3]):
            dX = dX + y_center - cal[i,1]
            case +=1

            
    if case == 0:
        case =1  
        dX = 0
                 
    dX =dX/case
    
    
    return dX


@njit(parallel=False)
def kernel_simple(cal, dX, y_center):
   
    E = np.exp(BB)*np.exp(AA*np.log(dX))
    Eplus = np.exp(BB)*np.exp(AA*np.log(dX+y_pixel_size))
    bin_size = (E - Eplus)
    
    return E, bin_size



@njit(parallel=False)
def decimate(array, step):
    out = np.zeros_like(array)
    for i in prange(0,step):
        out[:-step:] = np.add(out[:-step:],array[i:-step+i:])
    out2= np.divide(out[::step],step)
    
    return out2[:-1]

@njit(parallel=False)
def XtoE(X_scale, intensity, cal, y_center):
    E_scale = np.zeros_like(X_scale)
    bins = np.zeros_like(X_scale)
    
    for i in prange(len(E_scale)):
     E_scale[i], bins[i] = kernel(cal, X_scale[i], y_center)
     #print(i)
     intensity_bin = np.divide(intensity, bins)
    return E_scale, intensity_bin

def file_to_array(file):
    spectrum_to_read = np.array(Image.open(onlyfiles[0]).convert("F"))
    print(file)
    return spectrum_to_read


from joblib import Parallel, delayed

    



def load_multi(onlyfiles):
  images = []
  images = Parallel(n_jobs=1)(delayed(file_to_array)(file) for file in onlyfiles)
 
  return images
    

# load and show an image with Pillow
from PIL import Image


def load_data_mp(images, flip):
    
    spectrum_sum = np.array(images[0])
    if flip == True:
        spectrum_sum = np.flip(spectrum_sum, 1)
        
    intensity = np.sum(spectrum_sum[singal_line-gap:singal_line+gap:,:], axis = 0)
    noise_profile = np.sum(spectrum_sum[noise_line-gap:noise_line+gap:,:], axis = 0)
     
    LLL = len(images)
    DDD = len(spectrum_sum[singal_line-gap:singal_line+gap:,0])
    print(LLL)
    
    
    if LLL > 1: 
     spectrum_sum = np.zeros_like(np.array(images[0]))   
     #if flip == True:
        #spectrum_sum = np.flip(spectrum_sum, 1)
     intensity = np.zeros_like(np.sum(spectrum_sum[singal_line-gap:singal_line+gap:,:], axis = 0))
     
     
     noise_profile = np.zeros_like(np.sum(spectrum_sum[noise_line-gap:noise_line+gap:,:], axis = 0))  
     for image in images:    
       spectrum_sum = np.array(image) + spectrum_sum
     if flip == True:
        spectrum_sum = np.flip(spectrum_sum, 1)
     intensity = np.sum(spectrum_sum[singal_line-gap:singal_line+gap:,:], axis = 0)
     noise_profile = np.sum(spectrum_sum[noise_line-gap:noise_line+gap:,:], axis = 0)
     spectrum_sum = spectrum_sum / LLL
     
    intensity = intensity / DDD / LLL 
    noise_profile /= LLL*DDD
      
      
    return spectrum_sum, intensity, noise_profile

def load_data(onlyfiles, flip):
    image = Image.open(onlyfiles[0]).convert("F")
    
    spectrum_sum = np.zeros_like(np.array(image))
    if flip == True:
        spectrum_sum = np.flip(spectrum_sum, 1)
   
    intensity = np.zeros_like(np.sum(spectrum_sum[singal_line-gap:singal_line+gap:,:], axis = 0))
    noise_profile = np.zeros_line(np.sum(spectrum_sum[noise_line-gap:noise_line+gap:,:], axis = 0))
    
    
    
    for filename in onlyfiles:
    
      image = Image.open(filename).convert("F")
      spectrum_sum = np.array(image) + spectrum_sum
      intensity =intensity +  np.sum(np.array(image)[singal_line-gap:singal_line+gap:,:], axis = 0)
      noise_profile = noise_profile + np.sum(np.array(image)[noise_line-gap:noise_line+gap:,:], axis = 0)
    return spectrum_sum, intensity, noise_profile

##Read experimental value
# Open the image form working directory

#image = Image.open('C:/ALFA_experiment/slit_in0_mag_out/Electron_Spectr__21782323__20220218_152810964_1.tiff').convert("F")
#mypath = 'C:/ALFA_experiment/slit_in0_mag_out/'



#image = Image.open('C:/ToTest/Untitled1.png').convert("L")
#mypath = 'C:/ToTest/'


# summarize some details about the image
#print(image.format)
#print(image.size)
#print(image.mode)
# show the image



from os import listdir
from os.path import isfile, join, dirname

mypath = dirname(myfile)
ref_mypath = dirname(reference_file)
f = listdir(mypath)
f_ref = listdir(ref_mypath)

onlyfiles = []
onlyfiles_ref = []

for f in listdir(mypath):
    onlyfiles = onlyfiles + [join(mypath, f)]


for f_ref in listdir(ref_mypath):
    onlyfiles_ref = onlyfiles_ref + [join(ref_mypath, f)]


if folder_processing == False:
   onlyfiles_ref = [] + [reference_file] #to process one file only
   onlyfiles = [] + [myfile] #to process one file only

images = load_multi(onlyfiles)






spectrum_sum, intensity, noize_profile = load_data_mp(images, flip)
if no_reference_subtraction == False:
   images_ref = load_multi(onlyfiles_ref)
   spectrum_sum_ref, intensity_ref, noize_profile_ref = load_data_mp(images_ref, flip)
   spectrum_sum = spectrum_sum - spectrum_sum_ref
   intensity = intensity - intensity_ref
   noize_profile = noize_profile - noize_profile_ref



#images = load_multi(onlyfiles)
#images = [] + [load_multi([onlyfiles[46]])] 




    
noise_average = intensity
noize_profile = noise_average
noize_profile = savgol_filter(noize_profile, 3, 1) * 1

l = len(noise_average)

noise_average = np.sum(noise_average)/l

#image.show()

#Start plot
fig2, axs2= plt.subplots(1, 2)
fig2.canvas.set_window_title('Measured beam profile')
#fig2.suptitle("Beam: Div = 0 mrad, spot 0.0 mm, KE = 1e5 - 1e8 eV sequence, perfect source, Counts = 200 000.")
fig2.suptitle(myfile)
fig2.subplots_adjust(left  = 0.05, right = 0.95, wspace=.3)

# Decimation for the resolution calculation
step = 20
#Profile 1
B = 0.1






# axs2[1].plot(np.arange(len(spectrum_sum[0,:]))*y_pixel_size-y, intensity)
# axs2[1].plot(np.arange(len(spectrum_sum[0,:]))*y_pixel_size-y, noize_profile)
# #axs2[1].set_xlim(-yrange,0)
# axs2[1].set_xlabel('$\Delta x, mm$', fontsize = 16)
# axs2[1].set_ylabel('Intensity, a.u.', fontsize = 16)
# axs2[1].set_title('y-axis integrated intensity')


#scale_y, scale_z = np.mgrid[slice(0, len(spectrum_sum[:,0])*y_pixel_size, y_pixel_size), slice(0, len(spectrum_sum[0,:])*z_pixel_size, z_pixel_size)]
#axs2[0].pcolormesh(scale_y, scale_z, spectrum_sum, cmap = "binary", shading = "auto", vmax =20000)


axs2[0].imshow(spectrum_sum, interpolation='nearest', extent=[0, len(spectrum_sum[0,:])*y_pixel_size, len(spectrum_sum[:,0])*z_pixel_size, 0], cmap = "binary", vmax =40000)
axs2[0].set_xlabel('$x$, mm', fontsize = font_size)
axs2[0].set_ylabel('$y$, mm', fontsize = font_size)
axs2[0].set_title('distribution of electrons on the screen')
axs2[0].hlines((singal_line-gap)*y_pixel_size, 0, len(spectrum_sum[0,:])*z_pixel_size, linestyle = ":", linewidth = 3, color = 'green')
axs2[0].hlines((singal_line+gap)*y_pixel_size, 0, len(spectrum_sum[0,:])*z_pixel_size, linestyle = ":", linewidth = 3, color = 'green')
axs2[0].vlines(y, (singal_line-gap)*y_pixel_size, (singal_line+gap)*y_pixel_size, linestyle = ":", linewidth = 3, color = 'red')



def forward(x):
    return x

def inverse(x):
    return x
secax = axs2[0].secondary_xaxis('top', functions=(forward, inverse))




###Test of the kernel

ec = np.genfromtxt('C:/SIMION/ALFA spectrometer - no aperture - improved geometry/calibration_10mm_slit_N2pointing.csv',
                        skip_header = 1, skip_footer = 1, delimiter=',')
#ec = np.genfromtxt('C:/SIMION/ALFA spectrometer - no aperture - improved geometry/calibration_perfect_conditions.csv',
#                        skip_header = 1, skip_footer = 1, delimiter=',')


step = 29
calibration = np.zeros((int(len(ec[:,1])/step),4))
calibration[:,1] = decimate(ec[:,1], step)
calibration[:,2] = decimate(ec[:,2], step)
calibration[:,3] = decimate(ec[:,3], step)


#Simion model y offset
simion_y = 237.5
simion_y = 250 #perfect conditiond, Bari experiment
simion_y = 239.0 #10 mm slit Mix
intensity = intensity - intensity_threshold
intensity = intensity.clip(min=0)
X_scale = np.arange(len(spectrum_sum[0,:]))*y_pixel_size-y

E_scale, intensity_bin = XtoE(X_scale, intensity, calibration, simion_y) ##262 is calibration function offset for N2 experimental data


if what_to_plot == "Energy":

 range_min_index = 0 # mev
 range_max_index = 0 # mev
 length_scale = np.size(E_scale)-1
 for ii in range(1,length_scale,1):
     if (E_scale[ii]/1e6) < range_max and (E_scale[ii]/1e6) > 0:
        
        if range_max_index == 0:
             range_max_index = ii
             
 for ii in range(length_scale,1, -1):
     if (E_scale[ii]/1e6) > range_min:
        
        if range_min_index == 0:
             range_min_index = ii
             
 print("Range ", range_min_index, range_max_index)
 print("Energy range ", E_scale[range_min_index], E_scale[range_max_index])
 smooth_intensity_bin = savgol_filter(intensity_bin[range_max_index:range_min_index],19,1)/y_pixel_size
 axs2[1].plot(E_scale[range_max_index:range_min_index]/1e6, (intensity_bin[range_max_index:range_min_index])/y_pixel_size)
 axs2[1].plot(E_scale[range_max_index:range_min_index]/1e6, smooth_intensity_bin)


 cmass = np.sum(np.multiply(E_scale[range_max_index:range_min_index]/1e6, smooth_intensity_bin))/np.sum(smooth_intensity_bin)
 print("Center of the mass ",cmass)
 axs2[1].set_xlim(1,100)
 axs2[1].set_ylim(0,)
 axs2[1].set_xlabel('$E$, MeV', fontsize = font_size)
 axs2[1].set_ylabel('Intensity, a.u. / MeV', fontsize = font_size)
 axs2[1].set_title('energy distribution')

if what_to_plot == "Offset":
 axs2[1].plot(X_scale, intensity/max_intensity)
 axs2[1].set_xlim(0,)
 axs2[1].set_ylim(0,)
 axs2[1].set_xlabel('Beam offset, mm', fontsize = font_size)
 axs2[1].set_ylabel('Saturation level, 1 / mm', fontsize = font_size)
 axs2[1].set_title('energy distribution')
 #axs2[1].set_yscale("log")
 
def forward(y):
    return y*sensitivity

def inverse(y):
    return y/sensitivity
if dosimetry == True:
   secax2 = axs2[1].secondary_yaxis('right', functions=(forward, inverse))
   secax2.tick_params(axis='y',  labelsize=font_size)
   secax2.set_ylabel('dose, mGy/frame/MeV', fontsize = font_size)



print(kernel(calibration, 15, simion_y))
print(50, kernel_reverse(calibration, 50e6, simion_y))
print(40, kernel_reverse(calibration, 40e6, simion_y))
print(10, kernel_reverse(calibration, 10e6, simion_y))
print(5, kernel_reverse(calibration, 5e6, simion_y))
print(2, kernel_reverse(calibration, 2e6, simion_y))


#Energy axis
e_scale_custom = [50, 10, 5, 2]
x_scale_custom_list = np.zeros_like(e_scale_custom)
for ii in range(len(e_scale_custom)):
    x_scale_custom_list[ii] = kernel_reverse(calibration, e_scale_custom[ii]*1e6, simion_y)+y



secax.set_xticks(x_scale_custom_list)
secax.set_xticklabels(e_scale_custom)
secax.set_xlabel('energy, MeV', fontsize = font_size, y = 1.03)
secax.tick_params(axis='x',  labelsize=font_size)


axs2[0].tick_params(axis='x', labelsize= font_size)
axs2[0].tick_params(axis='y', labelsize= font_size)

plt.xticks(fontsize = font_size)
plt.yticks(fontsize = font_size)


plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.4)

plt.rcParams['savefig.dpi'] = 600



plt.show()

outputarray = np.dstack(E_scale[range_max_index:range_min_index]/1e6, smooth_intensity_bin)
#print(np.size())
np.savetxt('c:/spectrum_out.csv', outputarray, delimiter=',')


 




