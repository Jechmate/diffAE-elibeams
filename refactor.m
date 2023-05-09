software_folder=pwd; % save folder where the script is located
data_folder_address = ['/Users/matejech/Work/elibeams/Electron_Spectra_ALFA/1']
%----Initialization and User Inputs-----------
fprintf(data_folder_address)
cd(data_folder_address); % Go to data folder
images=dir('*.tiff');
matlab_counts_to_pC=0.003706; % converting image values in numner of electrons
image_gain=100/32; % correction for CCD settings
acquisition_time_ms=10;
saturated_pixel_in_pC=0.003706; % calibration factor
distance_gasjet_lanex_in_mm=375; 
pixel_in_mm=0.137;
pixel_in_mrad=0.3653;
pixel_in_msr=0.00010481;
tick_10mrad_px= 60 - round(10/pixel_in_mrad);
tick0mrad_px= 60;
tick10mrad_px = 60 + round(10/pixel_in_mrad);
noise=0.11;
hor_min=1163;
hor_max=1463;
ver_min=765;
ver_max=885;
electron_pointing_pixel=33;
hor_image_size=hor_max-hor_min+1;
ver_image_size=ver_max-ver_min+1;
hor_pixels=1:hor_image_size;
ver_pixels=1:ver_image_size;
num_images=length(images);
current_above_20MeV=zeros(1,num_images);
divergence_above_20MeV=zeros(num_images,ver_image_size);
amplitude_hor=zeros(1,num_images);
divergence_hor=zeros(1,num_images);
pointing_hor=zeros(1,num_images);
amplitude_ver=zeros(1,num_images);
divergence_ver=zeros(1,num_images);
pointing_ver=zeros(1,num_images);
deflection_mm=zeros(1,hor_image_size);
spectrum_in_pixel=zeros(1,hor_image_size);
spectrum_in_MeV=zeros(1,hor_image_size);
all_spectra_calibrated=zeros(num_images,hor_image_size);
average_image=zeros(ver_image_size,hor_image_size);
for i=1:hor_image_size    % defining the mm in the image
if i<=electron_pointing_pixel
    deflection_mm(i)=0;
else
    deflection_mm(i)=(i-electron_pointing_pixel)*pixel_in_mm;
end
end
deflection_MeV=zeros(1,hor_image_size);
%---Assigning to each pixel its value in MeV with the loaded deflection curve------
for i=electron_pointing_pixel:length(deflection_MeV)
    xq=deflection_mm(i);
    if xq>1                                                                         
       deflection_MeV(i)=interp1(deflection_curve_mm,deflection_curve_MeV,xq);
    end
end
% Ticks for images at 8, 10, 15, 20, 30, 40 and 100 MeV
tick100MeV=0;
tick50MeV=0;
tick40MeV=0;
tick30MeV=0;
tick20MeV=0;
tick15MeV=0;
tick10MeV=0;
tick8MeV=0;
tick5MeV=0;
laser_pointing_hor_cropped=electron_pointing_pixel;
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <101 && deflection_MeV(i) >70
       tick100MeV=i+1;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <51 && deflection_MeV(i) >40
       tick50MeV=i;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <41 && deflection_MeV(i) >30
       tick40MeV=i;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <31 && deflection_MeV(i) >20
       tick30MeV=i;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <21 && deflection_MeV(i) >15
       tick20MeV=i;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <15.5 && deflection_MeV(i) >12
       tick15MeV=i;
       break
   end
end

for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <10.5 && deflection_MeV(i) >8
       tick10MeV=i;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <8.2 && deflection_MeV(i) >6
       tick8MeV=i;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <5.2 && deflection_MeV(i) >4.8
       tick5MeV=i;
       break
   end
end
for i=laser_pointing_hor_cropped:length(deflection_MeV)
   if deflection_MeV(i) <3.2 && deflection_MeV(i) >2.9
       tick3MeV=i;
       break
   end
end
%--Here identify black dots----
n_black_dots=4; % Here indicate the number of black dots, switches to active

black_dot_center_1=[47 57]; % Here indicate the #1 black dot center in pixel in the cropped image as written by MATLAB datatip
black_dot_radius_1=10; % Here indicate the #1 black dot radius in pixels in the cropped image
%--Repeate for all the black dots in the image-----
black_dot_center_2=[115 58];
black_dot_radius_2=12;
black_dot_center_3=[188 57];
black_dot_radius_3=12;
black_dot_center_4=[262 54];
black_dot_radius_4=12;
black_dot_center_5=[304 271];
black_dot_radius_5=10;
black_dot_center_6=[368 270];
black_dot_radius_6=6;
black_dot_center_7=[433 269];
black_dot_radius_7=8;
black_dot_center_8=[499 269];
black_dot_radius_8=8;
black_dot_center_9=[424 70];
black_dot_radius_9=8;
    idx=1;
    filename = images(idx).name
    I=imread(filename);
    % m = mean(I, 'all')
    % size(I)
    I_gray=mat2gray(I); % converting to gray values from 0 to 1
    % m = mean(I_gray, "all")
    I_crop=I_gray(ver_min:ver_max,hor_min:hor_max);
    I_crop(121, 301);
    % size(I_crop)
    % m = mean(I_crop, "all")
    I_crop=medfilt2(I_crop);
    figure(76)
    imagesc(I_crop)
    I_crop=I_crop-noise; % subtracting noise
    for j=1:hor_image_size
        for l=1:ver_image_size
            if I_crop (l,j) < 0 %| j<electron_pointing_pixel+1
                I_crop(l,j)=0;
            else
                I_crop(l,j)=I_crop(l,j)+noise;
            end
        end
    end
   
    I_wo_black_dots=I_crop;
    for k=1:n_black_dots
        aux=num2str(k);
        aux_center=strcat('black_dot_center_',aux);
        aux_radius=strcat('black_dot_radius_',aux);
        center=eval(aux_center);
        center_hor=center(1);
        center_ver=center(2);
        center(1);
        I_wo_black_dots(center(2), center(1));
        radius=eval(aux_radius);
        for l=1:hor_image_size
            for m=1:ver_image_size
                if l<center_hor+radius && m<center_ver+radius && l>center_hor-radius  && m>center_ver-radius ...
                        && sqrt((m-center_ver)^2 + (l-center_hor)^2) < radius
                    aux_x_min= center_hor - round(sqrt(radius^2-(m-center_ver)^2));
                    aux_x_max= center_hor + round(sqrt(radius^2-(m-center_ver)^2));
                    aux_y_min= center_ver - round(sqrt(radius^2-(l-center_hor)^2));
                    aux_y_max= center_ver + round(sqrt(radius^2-(l-center_hor)^2));
              
                    I_wo_black_dots(m,l)=((aux_x_max-l)/(aux_x_max-aux_x_min))*I_crop(m,aux_x_min)+...
                                         ((l-aux_x_min)/(aux_x_max-aux_x_min))*I_crop(m,aux_x_max);
                    
                end
            end
        end
    end
    figure(76)
    imagesc(I_wo_black_dots)
    average_image=average_image+I_wo_black_dots;
    I_calibrated=(I_wo_black_dots*0.003706)/(image_gain*pixel_in_msr*acquisition_time_ms);
    noise_calibrated=noise*1000*0.003706/(image_gain*pixel_in_msr*acquisition_time_ms);
    horizontal_profile=sum(I_wo_black_dots);
    mean_hor_prof = mean(horizontal_profile)
    vertical_profile=sum(I_wo_black_dots');
    for j=electron_pointing_pixel:hor_image_size
        spectrum_in_pixel(j)=horizontal_profile(j); % spectrum in pixel
    end
    %spectrum in MeV
    spectrum_in_MeV(1)=spectrum_in_pixel(1);           
    for j=electron_pointing_pixel:hor_image_size
    spectrum_in_MeV(j)=(spectrum_in_pixel(j))/(deflection_MeV(j-1)-deflection_MeV(j)) ;
    end
    % calibrazione carica in nA/MeV
    spectrum_calibrated=(spectrum_in_MeV*0.003706)/(image_gain*acquisition_time_ms);
    
    horizontal_divergence_fit=fit(hor_pixels',horizontal_profile',...
        'a1*exp(-((x-b1)/c1)^2)','Start',[40 100 10]);
    vertical_divergence_fit=fit(ver_pixels',vertical_profile',...
        'a1*exp(-((x-b1)/c1)^2)','Start',[40 100 10]);
    current_above_20MeV_aux=sum(I_wo_black_dots);
    current_above_20MeV_aux=current_above_20MeV_aux(electron_pointing_pixel+1:tick20MeV);
    current_above_20MeV(i)=sum(current_above_20MeV_aux);
    divergence_above_20MeV_aux=I_wo_black_dots(:,electron_pointing_pixel+1:tick20MeV);
    divergence_above_20MeV(i,:)=sum(divergence_above_20MeV_aux,2);

    amplitude_hor(i)=horizontal_divergence_fit.a1;
    amplitude_ver(i)=vertical_divergence_fit.a1;
    pointing_hor(i)=horizontal_divergence_fit.b1;
    pointing_ver(i)=vertical_divergence_fit.b1;
    divergence_hor(i)=horizontal_divergence_fit.c1;
    divergence_ver(i)=vertical_divergence_fit.c1;
    
    % calibrazione carica in pC/MeV
    %spectrum_calibrated=(spectrum_in_MeV*0.497)/image_gain;
    % image in brightness (nC/sr)
    %I_calibrated=(I_wo_noise*0.000497)/(image_gain*pixel_in_msr);
  
    figure(1)
   title('Processed Image')
    %clims = [0 0.35];
    imagesc(I_calibrated*1000)
    ylabel('mrad')
    xlabel('MeV')
    colorbar
    h = colorbar;
    h.FontSize = 48;
    h.Ticks = [10 30];
    h.TickLabels = [10 30];
    ylabel(h, 'pA/msr')
   yline(tick_10mrad_px,'--w','LineWidth',2)
   yline(tick10mrad_px,'--w','LineWidth',2)
   xline(electron_pointing_pixel,'r','LineWidth',2)
   xline(tick50MeV,'--w','LineWidth',2)
   xline(tick20MeV,'--w','LineWidth',2)
   xline(tick10MeV,'--w','LineWidth',2)
   yt=[tick_10mrad_px tick0mrad_px tick10mrad_px];
   xt=[tick50MeV tick20MeV tick10MeV ];  
   set(gca, 'XTick',xt, 'XTickLabel',{'50','20','10'}, ...
  'YTick',yt,'YTickLabel',{'-10','0','10'},'Fontsize',48)   % Label Ticks
   
   
I_zoomed=I_calibrated(:,1:tick10MeV+10);
 figure(11)
   title('Zoomed Image')
    imagesc(I_zoomed*1000)
    ylabel('mrad')
    xlabel('MeV')
    colorbar
    h = colorbar;
    max_colorbar=h.Limits(2);
    h.FontSize = 48;
    h.Ticks = [max_colorbar*0.2  max_colorbar*0.6 max_colorbar ];
    h.TickLabels = [round(max_colorbar*0.2) round(max_colorbar*0.6) round(max_colorbar)];
    h.TickLength = 0.05;
    h.LineWidth=1.5;
    ylabel(h, 'pA/msr')
   yline(tick_10mrad_px,'--w','LineWidth',2)
   yline(tick10mrad_px,'--w','LineWidth',2)
   xline(electron_pointing_pixel,'r','LineWidth',2)
   xline(tick50MeV,'--w','LineWidth',2)
   xline(tick30MeV,'--w','LineWidth',2)
   xline(tick20MeV,'--w','LineWidth',2)
   xline(tick10MeV,'--w','LineWidth',2)
   
   yt=[tick_10mrad_px tick0mrad_px tick10mrad_px];
   xt=[  tick50MeV tick30MeV tick20MeV tick10MeV];  
   set(gca, 'XTick',xt, 'XTickLabel',{'50','30','20','10'}, ...
  'YTick',yt,'YTickLabel',{'-10','0','10'},'Fontsize',48)   % Label Ticks
    all_spectra_calibrated(i,:)=spectrum_calibrated;
average_image_calibrated=(average_image*0.003706)/(image_gain*pixel_in_msr*acquisition_time_ms);
horizontal_profile=sum(average_image);
for j=electron_pointing_pixel:hor_image_size
    spectrum_in_pixel(j)=horizontal_profile(j); % spectrum in pixel
end
%spectrum in MeV
spectrum_in_MeV(1)=spectrum_in_pixel(1);           
for j=electron_pointing_pixel:hor_image_size
    spectrum_in_MeV(j)=(spectrum_in_pixel(j))/(deflection_MeV(j-1)-deflection_MeV(j)) ;
end
% calibrazione carica in nA/MeV
spectrum_calibrated=(spectrum_in_MeV*3.706)/(image_gain*acquisition_time_ms);
spectrum_calibrated
figure(5)
title('Reconstructed Spectrum')
plot(deflection_MeV,spectrum_calibrated)  %plot without fit
ylabel('Spectral Intensity (pA/MeV)')
xlabel('Energy (MeV)')
xlim([2 8])
writematrix(spectrum_calibrated, 'spectrum_output.txt')
% Spectrum plot
rms_spectrum_aux=zeros(1,hor_image_size);   
rms_spectrum_mean=zeros(1,hor_image_size);   
rms_spectrum_rms=zeros(1,hor_image_size);  
min_MeV=3;
max_MeV=20;
for i=1:hor_image_size
    rms_spectrum_aux=all_spectra_calibrated(:,i);
    rms_spectrum_aux=nonzeros(rms_spectrum_aux);
    rms_spectrum_mean(i)=mean(rms_spectrum_aux);
    rms_spectrum_rms(i)=std(rms_spectrum_aux);
end
deflection_MeV_rms_plot=deflection_MeV(electron_pointing_pixel+15:end)';
rms_spectrum_mean_plot=rms_spectrum_mean(electron_pointing_pixel+15:end)';
