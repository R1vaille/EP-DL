function [M_recover] = DLReconstruction( D, b0, Dictionary, param, Mask, QSM_recon_param)
% % load('.\data\tkd.mat')
load ./data/data1_echo1_RDF.mat
% load('.\data\chi_cosmos.mat');
% load('.\data\chi_33');
% load ('.\data\Human_RDF.mat');
% load ('.\data\HumanTruth.mat');
% HumanTruth = real(HumanTruth);
thre_tkd = 0.1;
kernel_inv = zeros(size(b0));
kernel_inv( abs(D) > thre_tkd ) = 1 ./ D(abs(D) > thre_tkd);
chi_tkd = real( ifftn( fftn(b0) .* kernel_inv ) ) .* Mask; 


% disp(compute_rmse(chi_tkd, chi_cosmos))
% disp(compute_hfen(chi_tkd, chi_cosmos))
% disp(compute_ssim(chi_tkd, chi_cosmos))
% 
% disp(compute_rmse(chi_tkd, chi_33))
% disp(compute_hfen(chi_tkd, chi_33))
% disp(compute_ssim(chi_tkd, chi_33))
iMag = QSM_recon_param.iMag;
voxel_size = QSM_recon_param.voxel_size;
% 
M_recover = double(zeros(size(b0)));
% M_recover = double(chi_tkd);
% figure;
% imshow(HumanTruth(:,:,73),[]);
% imshow(chi_cosmos(:,:,75),[-0.15,0.15]);colorbar;
% caxis([-0.2,0.2])
% colorbar;
% 

RDF = b0;

param.lambda = 100;
param.max_iter = 10;
blocksize = param.blocksize;
% blocksize = blocksize(1:2);
stepsize=[1 1 1]; 
cur_iter = 1;
% lambdas = [0.01, 10, 1, 0.1, 100, 10000, 1000];
% for lambda_iter = 1:size(lambdas)
% grad = double(real(ifftn(D.*(D.*fftn(M_recover) - fftn(b0)))) + 2*lambda*M_temp);
% grad_prev = grad;
m = Mask;
grad = @fgrad;
div = @bdiv;
% reg = @calReg;
Dconv = @(dx) real(ifftn(D.*fftn(dx)));
% wG = gradient_mask(1, iMag, Mask, grad, voxel_size);
cos_wG = gradient_mask(1, iMag, Mask, grad, voxel_size);

e=0.000001;
b0 = m.*exp(1i*RDF);
% rmse_array = zeros(param.max_iter, 1);
% hfen_array = zeros(param.max_iter, 1);
% ssim_array = zeros(param.max_iter, 1);
M_temp = zeros(size(b0));
M_temp_t = zeros(size(b0));
M_recon = zeros(size(b0));
lambda1 = QSM_recon_param.lambda1;
lambda2 = QSM_recon_param.lambda2;
beta    = QSM_recon_param.beta;
while(cur_iter < param.max_iter)
    if(cur_iter == 4)
        disp('disilun') 
    end
    tic
%         for k = 1:stepsize(3):size(M_temp,3)-blocksize(3)+1
%             for j = 1:stepsize(2):size(M_temp,2)-blocksize(2)+1
%                 extract_block = M_recover(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1) * 100;
% %                 mean_block = mean(extract_block(:));
% %                 extract_block = extract_block - mean_block;
%                 blocks = im2colstep(extract_block,blocksize,stepsize);
%                 gamma  = full(mexOMP(blocks, Dictionary, param));
% %                 reconblocks = (Dictionary*gamma + mean_block) / 100;
%                 cleanblocks = blocks - Dictionary*gamma;
% %                 cleanblocks = ((blocks - Dictionary*gamma) + mean_block) / 100;
%                 reconblocks = Dictionary*gamma / 100;
% %                 cleanblocks = (blocks - Dictionary*gamma) / 100;
%                 reconvol = col2imstep(reconblocks,[size(M_temp,1) blocksize(2:3)],blocksize,stepsize);
%                 cleanvol = col2imstep(cleanblocks,[size(M_temp,1) blocksize(2:3)],blocksize,stepsize);          
%                 M_temp(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1) = M_temp(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1) + cleanvol;
%                 M_recon(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1) = M_recon(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1) + reconvol;
%             end
%         end
        if max(M_recover(:)) ~= 0
            for i = 1 : size(M_temp,1) - blocksize(1) + 1
                for j = 1 : size(M_temp,2) - blocksize(2) + 1
                    for k = 1 : size(M_temp,3) - blocksize(3) + 1
                        extract_block = M_recover(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) * 100;
                        mean_block = mean(extract_block(:));
                        extract_block = extract_block;
                        col_blocks = im2colstep(extract_block,blocksize,stepsize);
                        gamma  = full(mexOMP(col_blocks, Dictionary, param));
%                         cleanblocks = col_blocks - Dictionary*gamma;
%                         reconblocks = Dictionary*gamma;
                        cleanblocks = col_blocks - Dictionary*gamma;
                        cleanblocks_t = col_blocks - Dictionary*gamma + mean_block;
                        reconblocks = Dictionary*gamma + mean_block;
                        cleanvol = col2imstep(cleanblocks,blocksize,blocksize,stepsize) / 100;
                        cleanvol_t = col2imstep(cleanblocks_t,blocksize,blocksize,stepsize) / 100;
                        reconvol = col2imstep(reconblocks,blocksize,blocksize,stepsize) / 100;
                        M_temp(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) = M_temp(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) + cleanvol;
                        M_temp_t(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) = M_temp(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) + cleanvol_t;
                        M_recon(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) = M_recon(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) + reconvol;
                    end
                end
            end
        end
    cnt = countcover(size(M_temp),[blocksize, 1], [stepsize, 1]);
    M_recon = M_recon./cnt.*Mask;
    figure;imshow(M_recon(:,:,75),[-0.15,0.15]);
    M_temp = M_temp./cnt.*Mask;
%     M_temp_t = M_temp_t./cnt.*Mask;
%     dic_res_norm = norm(M_temp(:), 2);
% %     dict_res_norm_array(cur_iter) = dic_res_norm;
%     disp(['dictionary residual norm:' string(dic_res_norm)])
%     fidelity_res_norm = real(ifftn(D.*fftn(M_recover) - fftn(b0)));
% %     fidelity_res_norm = norm(fidelity_res(:, 2));
% %     fidelity_res_norm_array(cur_iter) = fidelity_res_norm;
%     disp(['fidelity residual norm:' string(norm(fidelity_res_norm(:), 2))])
%     
%     TV_term = wG.*grad(real(M_recover), voxel_size);
%     TV_res_norm = norm(TV_term(:), 1);
%     disp(['TV residual norm:' string(TV_res_norm)])
    %%
%     xGrad = grad(real(M_recover),voxel_size);
%     Vr1 = sum(aWG(:,:,:,:,1).*xGrad, 4);
%     Vr2 = sum(aWG(:,:,:,:,2).*xGrad, 4);
%     Vr3 = sum(aWG(:,:,:,:,3).*xGrad, 4);
%     temp = cat(4, Vr1, Vr2, Vr3);
%     Vr = 1./sqrt(abs(temp).^2+e);
    Vr = 1./sqrt(abs(cos_wG.*grad(real(M_recover),voxel_size)).^2+e);
    w = m.*exp(1i*ifftn(D.*fftn(M_recover)));
%     A =  @(dx) reg(aWG, Vr, dx, grad, div, voxel_size) + 2*lambda1*Dconv(conj(w).*w.*Dconv(dx)) + dx;  %reg + 2*lambda*real(wD(H)*wD*DdX);     
%     b = reg(aWG, Vr, M_recover, grad, div, voxel_size) + 2*lambda1*Dconv( real(conj(w).*conj(1i).*(w-b0))) + M_temp;%reg + 2*lambda*real(); 
    reg = @(dx) div(cos_wG.*(Vr.*(cos_wG.*grad(real(dx),voxel_size))),voxel_size);
%     A = @(dx) reg(dx) + 2*lambda1*Dconv(conj(w).*w.*Dconv(dx)) + 2*lambda2 * (cos_wG(:,:,:,1).*dx+cos_wG(:,:,:,2).*dx+cos_wG(:,:,:,3).*dx)/3;
%     b = reg(M_recover) + 2*lambda1*Dconv(real(conj(w).*conj(1i).*(w-b0))) + 2*lambda2 * (cos_wG(:,:,:,1).*M_temp+cos_wG(:,:,:,2).*M_temp+cos_wG(:,:,:,3).*M_temp)/3;
    A = @(dx) reg(dx) + 2*lambda1*Dconv(conj(w).*w.*Dconv(dx)) + 2 * lambda2 .* dx;
    b = reg(M_recover) + 2*lambda1*Dconv(real(conj(w).*conj(1i).*(w-b0))) + 2 * lambda2 .* M_temp;
%     A = @(dx) reg(dx) + 2*lambda1*Dconv(conj(w).*w.*Dconv(dx));
%     b = reg(M_recover) + 2*lambda1*Dconv(real(conj(w).*conj(1i).*(w-b0)));
    %%
    %SFCR method
%     b_var = max(wG.* grad(real(M_recover), voxel_size) - 1./beta, 0);
%     A = @(dx) 2*lambda1*Dconv(conj(w).*w.*Dconv(dx)) + 2*lambda2*dx + 2*beta*div(wG.*wG.*grad(dx, voxel_size), voxel_size);
%     b = 2*lambda1*Dconv(real(conj(w).*conj(1i).*(w-b0))) + 2*lambda2*M_temp - 2*beta*div(wG.*(b_var - wG.*grad(M_recover, voxel_size)), voxel_size);       
    %%
    dx = real(cgsolve(A, -b, 0.01, 100, 0));
    M_recover = M_recover + dx;
    M_recover = double(M_recover);
    %%
    x = M_recover/(2*pi*delta_TE*CF)*1e6.*Mask;
%     x = M_recover.*Mask;
    figure;imshow(x(:,:,75),[-0.3,0.3]);
%     caxis([-0.1,0.1])
    colorbar;
%     if(cur_iter >= 1)
%         residual = chi_cosmos - x;
%         figure;
%         imshow(residual(:,:,80), []);
%         caxis([-0.2,0.2])
%         colorbar;
%     end
%     residual = chi_cosmos - x;
%     disp(['residual:' string(norm(residual(:)))])
%     disp(['rmse:' string(compute_rmse(x, chi_cosmos))]);
%     disp(['hfen:' string(compute_hfen(x, chi_cosmos))])
%     disp(['ssim:' string(compute_ssim(x, chi_cosmos))])
%     if cur_iter == 3
%         disp('3');
%     end
    
%     disp(['rmse:' string(compute_rmse(x, chi_33))]);
%     disp(['hfen:' string(compute_hfen(x, chi_33))])
%     disp(['ssim:' string(compute_ssim(x, chi_33))])
    %%
%     disp(['iteration:' int2str(cur_iter)])
%     rmse = compute_rmse(M_recover, chi_cosmos);
%     disp(['iteration ' int2str(cur_iter) ' rmse:'])
%     disp(rmse)
%     beta = beta * 1.2;
%     Dictionary = DictionaryTraining(M_recover, param);
    disp(['iter:' string(cur_iter)])
    cur_iter = cur_iter + 1;
%     if cur_iter >= 4
%         wG = gradient_mask(1, x, Mask, grad, voxel_size);
%         dwG = wG(:,:,:,1) | wG(:,:,:,2) | wG(:,:,:,3);
%     end
    toc
end
% cur_iter = 0;
%save(['.\data\QSM_' int2str(lambda_index) '.mat'], 'M_recover');
% end
% M_recover = M_recover/(2*pi*delta_TE*CF)*1e6.*Mask;
end

