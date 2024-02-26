% This document sets up the simulation of the measurement data and the reconstruction of the object 

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

%% Creating the scence

%Scene1 
scene = zeros(512,128,128);
scene(128,64,64) = 1;

%imshow(squeeze(max(scene,1)),[]) Shows the image of our convoluted
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

%Reshaping the measurements 
grid_z = repmat(linspace(0,1,M)',[1 N N]);
grid_z(1,:,:) = 1;
data = data./(grid_z.^4); %Distance Vanishing; Why do we do this? 
%% 2nd Bounce Forward
psf2 = definePsf2(N,M);%Defined psf2, without N&M, not sure if it works
final_measurement = convn(data,psf2,'same');



%% Reconstruction
fpsf2 = fftn(psf2);%inverse psf 
recon_1 = LCT(final_measurement,M,N,fpsf2,mtx,mtxi,0.1);%first reconstruction -using fpsf2
recon_2 = LCT(recon_1,M,N,fpsf,mtx,mtxi,0.1);%second reconstruction -using fpsf 

%show the result 
%threedshow(recon_2,range,width); not working somehow/unrecognised function

z_offset = 30;

tic_z = linspace(0,range./2,size(recon_2,1));
tic_y = linspace(-width,width,size(recon_2,2));
tic_x = linspace(-width,width,size(recon_2,3));

% Crop and flip reconstructed volume for visualization
ind = round(M.*2.*width./(range./2));
recon_2 = recon_2(:,:,end:-1:1);
recon_2 = recon_2((1:ind)+z_offset,:,:);

tic_z = tic_z((1:ind)+z_offset);

% View result
figure('pos',[10 10 900 300]);

subplot(1,3,1);
imagesc(tic_x,tic_y,squeeze(max(recon_2,[],1)));
title('Front view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('x (m)');
ylabel('y (m)');
colormap('gray');
axis square;

subplot(1,3,2);
imagesc(tic_x,tic_z,squeeze(max(recon_2,[],2)));
title('Top view');
set(gca,'XTick',linspace(min(tic_x),max(tic_x),3));
set(gca,'YTick',linspace(min(tic_z),max(tic_z),3));
xlabel('x (m)');
ylabel('z (m)');
colormap('gray');
axis square;

subplot(1,3,3);
imagesc(tic_z,tic_y,squeeze(max(recon_2,[],3))')
title('Side view');
set(gca,'XTick',linspace(min(tic_z),max(tic_z),3));
set(gca,'YTick',linspace(min(tic_y),max(tic_y),3));
xlabel('z (m)');
ylabel('y (m)');
colormap('gray');
axis square;



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

function psf = definePsf2(U,V)
    % Local function to compute NLOS blur kernel
    x = linspace(-1,1,2.*U);
    y = linspace(-1,1,2.*U);
    z = linspace(0,2,2.*V);
    [grid_z,grid_y,grid_x] = ndgrid(z,y,x);

    z0 = 10;%We Arbiturarily defined it as 5m
    psf2 = sqrt(grid_x.^2+grid_y.^2+z0.^2)-grid_z; %z0 is not defined 
    psf2 = double(psf2 == repmat(min(psf2,[],1),[2.*V 1 1]));
    psf2 = psf2./sum(psf2(:,U,U));
    psf2 = psf2./norm(psf2(:));
    psf = circshift(psf2,[0 U U]);
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
