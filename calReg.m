function reg = calReg( aWG, Vr, dx, grad, div, voxel_size)
    %reg = @(dx) div(wG.*(Vr.*(wG.*grad(real(dx),voxel_size))),voxel_size);
    
    dxGrad = grad(real(dx),voxel_size);
    Vr1 = sum(aWG(:,:,:,:,1).*dxGrad, 4);
    Vr2 = sum(aWG(:,:,:,:,2).*dxGrad, 4);
    Vr3 = sum(aWG(:,:,:,:,3).*dxGrad, 4);
    dtemp = Vr.*cat(4, Vr1, Vr2, Vr3);
    Vr1 = sum(aWG(:,:,:,:,1).*dtemp, 4);
    Vr2 = sum(aWG(:,:,:,:,2).*dtemp, 4);
    Vr3 = sum(aWG(:,:,:,:,3).*dtemp, 4);
    dtemp = cat(4, Vr1, Vr2, Vr3);
    reg = div(dtemp,voxel_size);
end

