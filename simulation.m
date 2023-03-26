%Simulating the vectorized volume of the albedos of the hidden surface
simulation_signal = zeros([512,64,64]);
simulation_signal (3,3,3) = 1;

%Using the formula to calculate the supposed measurements
ans1 = reshape(mtx*simulation_signal(:,:),[M N N]);
ans2 = convn(ans1,psf,'same');%Doing the convolution 
simulatedData = reshape(mtxi*ans2(:,:),[M N N]);






