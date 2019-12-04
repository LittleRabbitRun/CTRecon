function [imageData] = conjugateGradientGUI(model, projectionData, imageData, iterNum, isMultiGPU, app)

cfg = wrapperFromProjectionAndReconImageToCfg(projectionData, imageData);

xx = imageData.getResoX();
yy = imageData.getResoY();
mask = zeros(xx, yy);
for ii = 1 : xx
    for jj = 1 : yy
        if (((ii - 0.5 - xx/2) / (xx/2))^2 + ((jj - 0.5 - yy/2) / (yy/2))^2) <0.99
            mask(ii,jj) = 1;
        end
    end
end
mask = int8(mask);

viewSer = 1 : projectionData.getTotalProjectionNumber();
if isMultiGPU
    if strcmp(model, 'Branchless')
        forwardModel = 'fp_bl';
        backwardModel = 'bp_bl';
    elseif strcmp(model, 'PseudoDD')
        forwardModel = 'fp_pd';
        backwardModel = 'bp_pd';
    else
        forwardModel = 'fp_bl';
        backwardModel = 'bp_bl';
    end
else
    if strcmp(model, 'Branchless')
        forwardModel = 'fp_gpu_branchless_sat2d';
        backwardModel = 'bp_gpu_branchless_sat2d';
    elseif strcmp(model, 'PseudoDD')
        forwardModel = 'fp_gpu_pseudo_dd';
        backwardModel = 'bp_gpu_pseudo_dd';
    elseif strcmp(model, 'VolumeRendering')
        forwardModel = 'fp_gpu_volume_rendering';
        backwardModel = 'bp_gpu_zline';
    elseif strcmp(model, 'DoublePrecise')
        forwardModel = 'fp_gpu_soft_interp';
        backwardModel = 'bp_gpu_soft_interp';
    elseif strcmp(model, 'Branches')
        forwardModel = 'fp_gpu_branches';
        backwardModel = 'bp_gpu_branches';
    else
        forwardModel = 'fp_gpu_branchless_sat2d';
        backwardModel = 'bp_gpu_branchless_sat2d';
    end 
end
initImg = imageData.getImage();
initImg = permute(initImg, [3 1 2]);
if isMultiGPU
    Proj = @(x) dd3_multiGPU(forwardModel, cfg, single(x), viewSer, mask);
    Back = @(x) dd3_multiGPU(backwardModel, cfg, single(x), viewSer, mask);
    
else
    Proj = @(x) dd3(forwardModel, cfg, single(x), viewSer, mask);
    Back = @(x) dd3(backwardModel, cfg, single(x), viewSer, mask);
end

A = @(x) Back(Proj(x));
prj = projectionData.getProjection();
prj = permute(prj, [2 1 3]);
b = Back(prj);
r0 = b - A(initImg);
p0 = r0;
reconImg = initImg(:);

hwait=waitbar(0,'Reconstructing');


for ii = 1 : iterNum
    r0 = r0(:);
    tmp1 = r0' * r0;
    tmp2 = A(p0);
    tmp2 = tmp2(:);
    p0size = size(p0);
    p0 = p0(:);
    alpha = tmp1 / (p0' * tmp2);
    reconImg = reconImg + alpha * p0;
    r1 = r0 - alpha * tmp2;

    beta = (r1'*r1) / (r0'*r0);
    p0 = r1 + beta * p0;
    p0 = reshape(p0, p0size);
    r0 = r1;

    showImg = reshape(reconImg, p0size);
    imagesc(squeeze(showImg(floor(cfg.recon.recon_pixels_z / 2),:,:)), 'Parent', app.UIAxesImageTraversal);
    imagesc(squeeze(showImg(:,floor(cfg.recon.recon_pixels_x/2),:)), 'Parent', app.UIAxesImageSagittal);
    imagesc(squeeze(showImg(:,:, floor(cfg.recon.recon_pixels_y/2))), 'Parent', app.UIAxesImageCoronal);
    waitbar(ii / iterNum, hwait, 'Reconstructing');
end
reconImg = reshape(reconImg, p0size); 
reconImg = permute(reconImg, [2 3 1]);
imageData.setImage(reconImg);
close(hwait);
end