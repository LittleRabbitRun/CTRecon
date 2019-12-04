function [imageData] = ossartGUI(model, projectionData, imageData, osNum, iterNum, isMultiGPU, app)

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
reconImg = imageData.getImage();
reconImg = permute(reconImg, [3, 1, 2]);

prj = projectionData.getProjection();
prj = permute(prj, [2, 1, 3]);
requiredMemory = ((osNum+3) * numel(reconImg) + numel(prj) * 2) * 8 / 1024 / 1024;
userview = memory; % This function only support in Windows.
if requiredMemory < userview.MemAvailableAllArrays * 0.8
    haveLargeMemory = true;
else
    haveLargeMemory = false;
end
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


if haveLargeMemory
    for osIdx = 0 : osNum - 1
        views = viewSer(1+osIdx:osNum:cfg.acq.num_data_views);
        if isMultiGPU
            RowSum{osIdx+1} = dd3_multiGPU(forwardModel, cfg, single(ones(size(reconImg))), views, mask);
            ColSum{osIdx+1} = dd3_multiGPU(backwardModel, cfg, single(ones(size(prj(:,:,views)))), views, mask);
        else
            RowSum{osIdx+1} = dd3(forwardModel, cfg, single(ones(size(reconImg))), views, mask);
            ColSum{osIdx+1} = dd3(backwardModel, cfg, single(ones(size(prj(:,:,views)))), views, mask);
        end
        
    end    
end
   
hwait=waitbar(0,'Reconstructing');

for iterIdx = 1 : iterNum
    for osIdx = 0 : osNum - 1
        views = viewSer(1+osIdx:osNum:cfg.acq.num_data_views);
        tp = prj(:,:,views) - dd3(forwardModel, cfg, single(reconImg),views, mask);
        if haveLargeMemory
            rowSum = RowSum{osIdx+1};
        else
            if isMultiGPU
                rowSum = dd3_multiGPU(forwardModel, cfg, single(ones(size(reconImg))), views, mask);            
            else
                rowSum = dd3(forwardModel, cfg, single(ones(size(reconImg))), views, mask);            
            end
            
        end
        nzR = (rowSum~=0);
        tp(nzR) = tp(nzR) ./ rowSum(nzR);
        if isMultiGPU
            tm = dd3_multiGPU(backwardModel, cfg, single(tp), views, mask);
        else
            tm = dd3(backwardModel, cfg, single(tp), views, mask);            
        end
        
        if haveLargeMemory
            colSum = ColSum{osIdx+1};
        else
            if isMultiGPU
                colSum = dd3_multiGPU(backwardModel, cfg, single(ones(size(prj(:,:,views)))), views, mask);
            else
                colSum = dd3(backwardModel, cfg, single(ones(size(prj(:,:,views)))), views, mask);
            end
            
        end
        nzC = (colSum~=0);
        tm(nzC) = tm(nzC) ./ colSum(nzC);
        reconImg = reconImg + tm;
        imagesc(squeeze(reconImg(floor(cfg.recon.recon_pixels_z / 2),:,:)), 'Parent', app.UIAxesImageTraversal);
        imagesc(squeeze(reconImg(:,floor(cfg.recon.recon_pixels_x/2),:)), 'Parent', app.UIAxesImageSagittal);
        imagesc(squeeze(reconImg(:,:, floor(cfg.recon.recon_pixels_y/2))), 'Parent', app.UIAxesImageCoronal);
        waitbar(((iterIdx - 1) * osNum + osIdx)/ (iterNum * osNum), hwait, 'Reconstructing');
    end 
end

reconImg = permute(reconImg, [2 3 1]);
imageData.setImage(reconImg);
close(hwait);
end

