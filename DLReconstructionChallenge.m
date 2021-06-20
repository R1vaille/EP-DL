function x = DLReconstructionChallenge( D, b0, Dictionary, param, Mask, QSM_recon_param)
load ./data/data4_echo1_RDF.mat
thre_tkds = 0.19;
thre_tkd = thre_tkds;
kernel_inv = zeros(size(b0));
kernel_inv( abs(D) > thre_tkd ) = 1 ./ D(abs(D) > thre_tkd);
chi_tkd = real( ifftn( fftn(b0) .* kernel_inv ) ) .* Mask; 

iMag = QSM_recon_param.iMag;
voxel_size = QSM_recon_param.voxel_size;
% M_recover = double(chi_tkd);
M_recover = zeros(size(chi_tkd));

RDF = b0;

param.lambda = 100;
param.max_iter = 10;
blocksize = param.blocksize;
stepsize=[1 1 1]; 
grad = @fgrad;
div = @bdiv;
Dconv = @(dx) real(ifftn(D.*fftn(dx)));
cos_wG = gradient_mask(1, iMag, Mask, grad, voxel_size);
m = dataterm_mask(1, double(N_std), Mask);
e=0.000001;
b0 = m.*exp(1i*RDF);
M_temp = zeros(size(b0));
M_temp_t = zeros(size(b0));
M_recon = zeros(size(b0));
lambda1 = QSM_recon_param.lambda1;
lambda2 = QSM_recon_param.lambda2;
step = 0.5;

%%
%init
% H = zeros(size(Mask));
% H(abs(D) >= 0.19) = 1;
% 
% R1 = zeros(size(Mask));
% R1(chi_tkd < 0.04) = 1;
% beta = 2;
% lambda1_init = 50;
% lambda2_init = 2;
% while(cur_iter <= 4)
%     tic
%     b_var = max(cos_wG.* grad(real(M_recover), voxel_size) - 1./beta, 0);
%     A = @(dx) lambda1_init* real(ifftn(fftn(dx))) + beta * div(grad(div(cos_wG.*grad(dx, voxel_size), voxel_size), voxel_size),voxel_size) + lambda2_init * R1;
%     b = lambda1_init *real(ifftn(fftn(chi_tkd))) + beta * div(grad(div((cos_wG .* b_var), voxel_size), voxel_size), voxel_size);
%     M_recover = real(cgsolve(A, b, 0.01, 100, 0)) .* Mask;
%     figure(10);imshow(M_recover(:,:,75),[-0.15,0.15]);
%     beta = beta / 2;
%     cur_iter = cur_iter + 1;
%     toc
% end





%%
%EP-DL
cur_iter = 0;
% cos_wG = gradient_mask(1, M_recover, Mask, grad, voxel_size);
R1 = zeros(size(Mask));
R1(M_recover <= 0.1) = 1;
while(cur_iter <= param.max_iter)
    if(cur_iter == 5)
        disp('µÚÎåÂÖ')
    end
    tic
        if max(M_recover(:)) ~= 0 % && lambda2 ~= 0
            for i = 1 : size(M_temp,1) - blocksize(1) + 1
                for j = 1 : size(M_temp,2) - blocksize(2) + 1
                    for k = 1 : size(M_temp,3) - blocksize(3) + 1
                        extract_block = M_recover(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) * 1000;
                        mean_block = mean(extract_block(:));
                        col_blocks = im2colstep(extract_block,blocksize,stepsize);
                        gamma  = full(mexOMP(col_blocks, Dictionary, param));
%                         cleanblocks = col_blocks - Dictionary*gamma;
%                         reconblocks = Dictionary*gamma;
                        cleanblocks = col_blocks - Dictionary*gamma;
                        reconblocks = Dictionary*gamma;
                        cleanvol = col2imstep(cleanblocks,blocksize,blocksize,stepsize) / 1000;                        
                        reconvol = col2imstep(reconblocks,blocksize,blocksize,stepsize) / 1000;
                        M_temp(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) = M_temp(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) + cleanvol;
                        M_recon(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) = M_recon(i : i + blocksize(1) - 1,j:j+blocksize(2)-1, k:k+blocksize(3)-1) + reconvol;
                    end
                end
            end
        end
        
    cnt = countcover(size(M_temp),[blocksize, 1], [stepsize, 1]);
    M_recon = M_recon./cnt.*Mask;
    M_temp = M_temp./cnt.*Mask;
    %%

    Vr = 1./sqrt(abs(cos_wG.*grad(real(M_recover),voxel_size)).^2+e);
    w = m.*exp(1i*ifftn(D.*fftn(M_recover)));
    reg = @(dx) div(cos_wG.*(Vr.*(cos_wG.*grad(real(dx),voxel_size))),voxel_size);

    A = @(dx) reg(dx) + 2*lambda1*Dconv(conj(w).*w.*Dconv(dx)) + 2 * lambda2 .* dx;
    b = reg(M_recover) + 2*lambda1*Dconv(real(conj(w).*conj(1i).*(w-b0))) + 2 * lambda2 .* M_temp;

%     A = @(dx) reg(dx) + 2*lambda1*Dconv(conj(w).*w.*Dconv(dx));
%     b = reg(M_recover) + 2*lambda1*Dconv(real(conj(w).*conj(1i).*(w-b0)));      
    %%
    dx = real(cgsolve(A, -b, 0.01, 100, 0));
    M_recover = M_recover + dx;
    M_recover = double(M_recover.*Mask);
    %%
    % x = M_recover/(2*pi*delta_TE*CF)*1e6.*Mask;
    figure;imshow(M_recover(:,:,100),[]);

    cur_iter = cur_iter + 1;
    disp(['iter:',int2str(cur_iter)])
    toc
end

end