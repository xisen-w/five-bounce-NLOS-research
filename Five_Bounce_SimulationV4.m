% This document sets up the simulation of the measurement data and the reconstruction of the object 
clear all;
% setting
N = 128; % Space Dimension
M = 512; % Time Dimension 
wall_size = 2; % Unit: m 
bin_resolution = 32e-12; % Every time stamp 
c    = 3e8;    % Speed of light (meters per second)
width = wall_size / 2;
range = M.*c.*bin_resolution; % Maximum range for histogram

%Creating the Hypercone 
psf = definePsf(N,M,width./range);
fpsf = fftn(psf);
[mtx,mtxi] = resamplingOperator(M); %Rt & Rz 
mtx = full(mtx);
mtxi = full(mtxi);
mask = zeros(1,128,128);
mask(1,10:118,10:118) = 1;
%% Creating the scences

for number = 1:3
scene = zeros(512,128,128);
    switch number
            case {1}
            scene(64,64,64)=1;
            draw3D(double(scene),0.95,0,1);
            case {2}
            load('fk_50min.mat');
            fk(end-200:end,:,:) = 0;
            fk = flip(flip(fk,2),3);
            for i = 1:128
                for j = 1:128
                    [ref,loc] = max(fk(:,i,j));
                    if loc >100 && ref>2
                    scene(loc-100,i,j) = fk(loc,i,j);el
                    end
                end
            end
            scene(1:80,:,:) = 0;
            draw3D(double(scene),0.95,0,1);
    end



%threedshow(scene,1,1); %Shows the image of our convoluted
%data/but got an error now 

%% Calculate the Measurements 

%% First Bounce Forward
%tdata = convn(scene,psf,'same');
tscene = zeros(2.*M,2.*N,2.*N);
tscene(1:end./2,1:end./2,1:end./2)  = reshape(mtx*scene(:,:),[M N N]);
tdata = ifftn(fftn(tscene).*fpsf);
tdata = tdata(1:end./2,1:end./2,1:end./2);

% *R_t^(-1)/Measurements
data  = reshape(mtxi*tdata(:,:),[M N N]); 

%Distance Vanishing; 
grid_z = repmat(linspace(0,1,M)',[1 N N]);
grid_z(1,:,:) = 1;
data = data./(grid_z.^2);  
threedshow(data,1,1);
%% 2nd Bounce Forward
z0 = 0.2; %0.2 m
psf2 = definePsf2(N,M,z0,width,range);
fpsf2 = fftn(psf2);%inverse psf 
%final_measurement = convn(data,psf2,'same');
tscene = zeros(2.*M,2.*N,2.*N);
tscene(1:end./2,1:end./2,1:end./2)  = data;
tdata = ifftn(fftn(tscene).*fpsf2);
final_measurement = tdata(1:end./2,1:end./2,1:end./2);

threedshow(final_measurement,1,1);
%% Reconstruction
recon_1 = LCT2(final_measurement,M,N,fpsf2,1);%first reconstruction -using fpsf2
recon_1(1:60,:,:) = 0; 
recon_1 = recon_1.*mask;
threedshow(recon_1,1,1);
%%
recon_2 = LCT(recon_1,M,N,fpsf,mtx,mtxi,1);%second reconstruction -using fpsf 
recon_2(end-300:end,:,:) = 0;
threedshow(recon_2 ,1,1);
%show the result 
draw3D(recon_2,0.95,0,1)

end


function psf = definePsf(U,V,slope)
    % Local function to compute NLOS blur kernel
    x = linspace(-1,1,2.*U);
    y = linspace(-1,1,2.*U);
    z = linspace(0,2,2.*V);
    [grid_z,grid_y,grid_x] = ndgrid(z,y,x);

    % Define PSF
    psf = abs(((4.*slope).^2).*(grid_x.^2 + grid_y.^2) - grid_z);
    psf = double(psf == repmat(min(psf,[],1),[2.*V 1 1]));
    psf = psf./sum(psf(:,U,U));
    psf = psf./norm(psf(:));
    psf = circshift(psf,[0 U U]);
end

function psf2 = definePsf2(U,V,z0,width,range)
    % Local function to compute NLOS blur kernel
    x = linspace(-2*width,2*width,2.*U);
    y = linspace(-2*width,2*width,2.*U);
    z = linspace(0,2*(range/2),2.*V);
    [grid_z,grid_y,grid_x] = ndgrid(z,y,x);


    psf2 = abs( sqrt(grid_x.^2+grid_y.^2+z0.^2)-grid_z ); %z0 is not defined 
    psf2 = double(psf2 == repmat(min(psf2,[],1),[2.*V 1 1]));
    psf2 = psf2./sum(psf2(:,U,U));
    psf2 = psf2./norm(psf2(:));
    psf2 = circshift(psf2,[0 U U]);
end

function [mtx,mtxi] = resamplingOperator(M)
 % Local function that defines resampling operators
     mtx = sparse([],[],[],M.^2,M,M.^2);
     
     x = 1:M.^2;
     mtx(sub2ind(size(mtx),x,ceil(sqrt(x)))) = 1;
     mtx  = spdiags(1./sqrt(x)',0,M.^2,M.^2)*mtx;
     mtxi = mtx';
     
     K = log(M)./log(2);
     for k = 1:round(K)
         mtx  = 0.5.*(mtx(1:2:end,:)  + mtx(2:2:end,:));
         mtxi = 0.5.*(mtxi(:,1:2:end) + mtxi(:,2:2:end));
     end
end

function vol = LCT(data,M,N,fpsf,mtx,mtxi,snr)
tic;%Counting Time 
grid_z = repmat(linspace(0,1,M)',[1 N N]);
data = data.*(grid_z.^2);
invpsf = conj(fpsf) ./ (abs(fpsf).^2 + 1./snr);

% Step 2: Resample time axis and pad result
tdata = zeros(2.*M,2.*N,2.*N);
tdata(1:end./2,1:end./2,1:end./2)  = reshape(mtx*data(:,:),[M N N]);

% Step 3: Convolve with inverse filter and unpad result
tvol = ifftn(fftn(tdata).*invpsf);
tvol = tvol(1:end./2,1:end./2,1:end./2);

% Step 4: Resample depth axis and clamp results
vol  = reshape(mtxi*tvol(:,:),[M N N]);
vol  = max(real(vol),0);
time_elapsed = toc;

display(sprintf(['Reconstructed volume of size %d x %d x %d '...
    'in %f seconds'], size(vol,3),size(vol,2),size(vol,1),time_elapsed));
end

function vol = LCT2(data,M,N,fpsf,snr) 
tic;%Counting Time 

invpsf = conj(fpsf) ./ (abs(fpsf).^2 + 1./snr);

% Step 2: Resample time axis and pad result
tdata = zeros(2.*M,2.*N,2.*N);
tdata(1:end./2,1:end./2,1:end./2)  = data;

% Step 3: Convolve with inverse filter and unpad result
tvol = ifftn(fftn(tdata).*invpsf);
vol = tvol(1:end./2,1:end./2,1:end./2);
vol  = max(real(vol),0);
time_elapsed = toc;

display(sprintf(['Reconstructed volume of size %d x %d x %d '...
    'in %f seconds'], size(vol,3),size(vol,2),size(vol,1),time_elapsed));
end