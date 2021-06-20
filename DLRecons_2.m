function M_recover = DLRecons_2( D, b0, Dictionary, param, Mask, lambda, lambda_index )
load('.\data\tkd.mat')
M_recover = tkd;
load('.\data\chi_cosmos.mat');
% figure;
% imshow(chi_cosmos(:,:,80),[]);
% caxis([-0.2,0.5])
% colorbar;
% 
% figure;
% imshow(M_recover(:,:,80),[]);
% caxis([-0.2,0.5])
% colorbar;

Dconv = @(dx) real(ifftn(D.*fftn(dx)));
M_temp = zeros(size(b0));
param.lambda = 100;
param.max_iter = 10;
blocksize = param.blocksize;
stepsize=[1 1 1]; 
cur_iter = 0;
% lambdas = [0.01, 10, 1, 0.1, 100, 10000, 1000];
% for lambda_iter = 1:size(lambdas)
while(cur_iter < param.max_iter)
    tic
    for k = 1:stepsize(3):size(M_temp,3)-blocksize(3)+1
        for j = 1:stepsize(2):size(M_temp,2)-blocksize(2)+1
            blocks = im2colstep(M_recover(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1),blocksize,stepsize);
            gamma  = full(mexOMP(blocks, Dictionary, param));
            cleanblocks = blocks - Dictionary*gamma - mean(blocks(:));
            cleanvol = col2imstep(cleanblocks,[size(M_temp,1) blocksize(2:3)],blocksize,stepsize);
            M_temp(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1) = M_temp(:,j:j+blocksize(2)-1,k:k+blocksize(3)-1) + cleanvol;
        end
%             if(rem(k, 10) == 0)
%                 disp(['layer ' int2str(k) ' finish'])
%             end
    end
    toc
    cnt = countcover(size(M_temp),blocksize,stepsize);
    M_temp = M_temp./cnt;
   %%
    %conjugate gradient method
    disp('conjugate gradient method')
    grad = lambda * M_temp + (Dconv(Dconv(M_recover) - b0));
    dire = -grad;
    alpha=0.1;
    beta=0.6;
    tau0=1;
    mask_matrix = D;
    
    %  初值
    tau=tau0;
    num=0;
    while ((f_2norm(mask_matrix,M_recover+tau*dire,b0))>...
           (f_2norm(mask_matrix,M_recover,b0)+alpha*tau*real(conj(grad).*dire)))
        tau=beta*tau;
        num=num+1;
    end

    %  自适应阈值
    if num>2
        tau0 = tau0*beta;
    end 
    if num<1
        tau0 = tau0/beta;
    end

    %  恢复图像修正
    M_recover=M_recover+tau*dire;
    %%
%     figure;
%     imshow(M_recover(:,:,80),[]);
%     caxis([-0.2,0.5])
%     colorbar;
%     disp(['iteration:' int2str(cur_iter)])
    cur_iter = cur_iter + 1;
end
% cur_iter = 0;
save(['.\data\QSM_' int2str(lambda_index) '.mat'], 'M_recover');
% end
end

function TT=f_2norm(D,X,B)  
res = real(ifftn(D.*fftn(X))) - B;
TT=norm(res(:),'fro')^2;
end

%  2范数的梯度
function TT=grad_2norm(D,X,B)   
TT=real(ifftn((D.*fftn((real(ifftn(D.*fftn(X))) - B)))));
end

