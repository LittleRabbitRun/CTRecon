function [reconImg] = conjgrad(initImg, prj, cfg, mask, iterNum)
mask = int8(mask);
viewSer = 1 : cfg.acq.num_data_views;
Proj = @(x) dd3('fp_gpu_branchless_sat2d', cfg, single(x), viewSer, mask);
Back = @(x) dd3('bp_gpu_branchless_sat2d', cfg, single(x), viewSer, mask);
A = @(x) Back(Proj(x));
b = Back(prj);
r0 = b - A(initImg);
p0 = r0;
reconImg = initImg(:);
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
%     if norm(r1) < epsilon
%            break;
%     end
    beta = (r1'*r1) / (r0'*r0);
    p0 = r1 + beta * p0;
    p0 = reshape(p0, p0size);
    r0 = r1;
    % if show
    if(true)
        cc = reshape(reconImg, p0size);
        imshow(squeeze(cc(32,:,:)),[0 1]);
        pause(0.001);            
    end
end
reconImg = reshape(reconImg, p0size);
end