function [real, img] = waveconvolution(time_resolution, virtual_guassian_size, virtual_lamda, data_t)
% Input parameter: 
% time_resolution:  unit second, time resolution per time bin 
% virtual_guassian_size:   integer, represent as virtual phasor wave guassian envelop
% virtual_lamda:     unit cm, virtual phasor wavelength
% data_t:               3D temporal measurement

% Output parameter:
% real, img: real and imaginary part of the virtual phasor wave

    c = 299792458; % speed of light

    
    s_z = time_resolution * c * 100; % unit: cm

    Sinusoid_pattern = virtual_guassian_size * s_z / virtual_lamda;

    Gauss_sigma = 0.3;

    sin_wave = sin(2*pi*(Sinusoid_pattern * linspace(1,virtual_guassian_size,virtual_guassian_size)')/virtual_guassian_size);
    cos_wave = cos(2*pi*(Sinusoid_pattern * linspace(1,virtual_guassian_size,virtual_guassian_size)')/virtual_guassian_size);
    gauss_wave = gausswin(virtual_guassian_size, 1/Gauss_sigma);

    Virtual_Wave_sin = sin_wave .* gauss_wave;
    Virtual_Wave_cos = cos_wave .* gauss_wave;


    for laser_index = 1 : size(data_t,2)
        for camera_index = 1 : size(data_t,3)
            time_response = squeeze(data_t(:,laser_index, camera_index));

            % Wave convolution
            real(:,laser_index, camera_index) = conv(time_response,Virtual_Wave_sin, 'same');
            img(:,laser_index, camera_index) = conv(time_response,Virtual_Wave_cos, 'same');

        end
    end 

end
