function [reconImg] = os_sart(initImg, prj, cfg, mask, osNum, iterNum)
mask = int8(mask);
viewSer = 1 : cfg.acq.num_data_views;
reconImg = initImg;
requiredMemory = ((osNum+3) * numel(reconImg) + numel(prj) * 2) * 8 / 1024 / 1024;
userview = memory; % This function only support in Windows.
if requiredMemory < userview.MemAvailableAllArrays * 0.8
    haveLargeMemory = true;
else
    haveLargeMemory = false;
end
if haveLargeMemory
    for osIdx = 0 : osNum - 1
        views = viewSer(1+osIdx:osNum:cfg.acq.num_data_views);
        RowSum{osIdx+1} = dd3('fp_gpu_branchless_sat2d', cfg, single(ones(size(initImg))), views, mask);
        ColSum{osIdx+1} = dd3('bp_gpu_branchless_sat2d', cfg, single(ones(size(prj(:,:,views)))), views, mask);
    end    
end
    
for iterIdx = 1 : iterNum
    for osIdx = 0 : osNum - 1
        views = viewSer(1+osIdx:osNum:cfg.acq.num_data_views);
        tp = prj(:,:,views) - dd3('fp_gpu_branchless_sat2d', cfg, single(reconImg),views, mask);
        if haveLargeMemory
            rowSum = RowSum{osIdx+1};
        else
            rowSum = dd3('fp_gpu_branchless_sat2d', cfg, single(ones(size(img))), views, mask);            
        end
        nzR = (rowSum~=0);
        tp(nzR) = tp(nzR) ./ rowSum(nzR);
        tm = dd3('bp_gpu_branchless_sat2d', cfg, single(tp), views, mask);
        if haveLargeMemory
            colSum = ColSum{osIdx+1};
        else
            colSum = dd3('bp_gpu_branchless_sat2d', cfg, single(ones(size(prj(:,:,views)))), views, mask);
        end
        nzC = (colSum~=0);
        tm(nzC) = tm(nzC) ./ colSum(nzC);
        reconImg = reconImg + tm;
%         imshow(squeeze(reconImg(32,:,:)),[0 1]);
%         pause(0.001);            
    end
end

end

