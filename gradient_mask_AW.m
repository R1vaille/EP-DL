function aWG=gradient_mask_AW(iMag, Mask, grad, voxel_size)
    percentage = 0.9;
    field_noise_level = 0.01*max(iMag(:));
    wG = abs(grad(iMag.*(Mask>0), voxel_size));
    denominator = sum(Mask(:)==1);
    numerator = sum(wG(:)>field_noise_level);
    if  (numerator/denominator)>percentage
        while (numerator/denominator)>percentage
            field_noise_level = field_noise_level*1.05;
            numerator = sum(wG(:)>field_noise_level);
        end
    else
        while (numerator/denominator)<percentage
            field_noise_level = field_noise_level*.95;
            numerator = sum(wG(:)>field_noise_level);
        end
    end

    wG = (wG>=field_noise_level);
    delta = wG(:,:,:,1) | wG(:,:,:,2) | wG(:,:,:,3);
    sigma = cat(4, delta.*wG(:,:,:,1), delta.*wG(:,:,:,2), delta.*wG(:,:,:,3));
    P1 = cat(4, 1 - sigma(:,:,:,1).*sigma(:,:,:,1), -sigma(:,:,:,1).*sigma(:,:,:,2), -sigma(:,:,:,1).*sigma(:,:,:,3));
    P2 = cat(4, -sigma(:,:,:,2).*sigma(:,:,:,1), 1 - sigma(:,:,:,2).*sigma(:,:,:,2), -sigma(:,:,:,2).*sigma(:,:,:,3));
    P3 = cat(4, -sigma(:,:,:,3).*sigma(:,:,:,1), -sigma(:,:,:,3).*sigma(:,:,:,2), 1 - sigma(:,:,:,1).*sigma(:,:,:,3));
    aWG = cat(5, P1, P2, P3);

end

