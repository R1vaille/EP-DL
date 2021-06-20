%x9BSs71ISS
%################# Example for Image Denoising  #########################################################################
% This script aims at demonstrating how to produce the denoised image and simultaneously the potential locations
% of cell nuclei in 2D/3D images as described in "A novel generic dictionary-based
% denoising method for improving noisy and
% densely packed nuclei segmentation in 3D time-lapse fluorescence microscopy
% images" Scientific Reports9 (2019): 5654 (2019)
% doi:10.1038/s41598-019-41683-3
% The script is implemented in MATLAB release R2017b by
% Lamees Nasser
% April 2019
%###################################################################################################################
clear all  % Close all figures
close all  % Clear  workspace
clc        % Clear command window.
train_Dictionary = 1;
load ./data/data4_echo1_RDF.mat
% RDF = RDF .* Mask;
D=dipole_kernel(matrix_size, voxel_size, B0_dir);
b0 = RDF;
% iMag = iMag.*Mask;
% % % % 
% % load('.\data\chi_cosmos.mat');
% IMin = (chi_cosmos-min(chi_cosmos(:)))./(max(chi_cosmos(:)) - min(chi_cosmos(:)));
IMin = double(iMag.*Mask);
IMin = IMin(50:110,50:150,50:150);

%############################ Initialization parameters ##########################################################


param.K = 500;                         % - The number of dictionary atoms to train.
param.numIteration =15;             % - The number of K-SVD training iterations
param.L=8;                          % - The number of nonzero elements (used atoms from the dictionary)
param.preserveDCAtom = 0;
param.InitializationMethod = 'GivenMatrix';
%   for the sparse representation coefficient
param.blocksize=[4 4 4];          % - The size of the blocks the algorithm
%                                       works. Example in 2D imgages p=[N N]. 3D images p= [N N M].
param.trainnum=10^4;                % - The number of blocks to train on.

if train_Dictionary == 1
Z = DictionaryTraining(IMin, param);
save challenge_2019_444.mat Z;
end

load challenge_2019_444.mat;

QSM_recon_param.iMag = iMag;
QSM_recon_param.voxel_size = voxel_size;

%%
%EP-DL
lambda1 = 600;
times = 20;
% times = [1000, 500, 200, 100, 50, 20, 10, 5];
% Res_lambda1 = zeros(size(times, 2), size(lambda1, 2));
% Res_lambda2 = zeros(size(times, 2), size(lambda1, 2));
% Res_rmse = zeros(size(times, 2), size(lambda1, 2));
% Res_hfen = zeros(size(times, 2), size(lambda1, 2));
% Res_ssim = zeros(size(times, 2), size(lambda1, 2));
% Res_roi_error = zeros(size(times, 2), size(lambda1, 2));
%%
%MEDI
QSM_recon_param.lambda1 = lambda1;
QSM_recon_param.lambda2 = 0;%lambda1 ./ times;
disp(['lambda1:' int2str(QSM_recon_param.lambda1)]);
disp(['lambda2:',int2str(QSM_recon_param.lambda2)]);
x = DLReconstructionChallenge( D, b0, Z, param, Mask, QSM_recon_param);
% [DL_QSM] = DLSimuReconstruction( D, b0, Z, param, Mask, QSM_recon_param);
DL_QSM = DL_QSM/(2*pi*delta_TE*CF)*1e6.*Mask;
figure;
imshow(DL_QSM(:,:,30),[]);
caxis([-0.2,0.2]);
colorbar;
close all
