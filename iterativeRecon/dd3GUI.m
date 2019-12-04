function dd3GUI(mode, projModel, geometry, projectionData, imageData, view_idx, mask, backprojectionType)
% DD3GUI Check that the data must be Z-first ordered arranged
if strcmp(mode, 'Projection')
    img = imageData.getImage();
    % Change the order of the image 
    img = repmat(img, [3 1 2]);
    
elseif strcmp(mode, 'Backprojection')
    sino = projectionData.getProjection();
    sino = repmat(sino, [2 1 3]);
else
end
x0 = geometry.getSourceToCenterDistance() * 0;
% We do not support revolution CT that the detector contains GAP, this
% field should be in projectionData object GEOM_REVO_GAP = 0 or 1
% We do not support shift of the source along x direction
% x0 = x0 + shift_fs_mm;
% We can support PARALLEL MODEL BY SETTING y0 = 10e10 or something similar
% with that.
y0 = geometry.getSourceToCenterDistance();
z0 = 0;
if ~exist('view_idx', 'var') || isempty(view_idx)
    view_idx = 1:projectionData.getTotalProjectionNumber();
end
viewangles = (0:(projectionData.getTotalProjectionNumber() - 1)) * 2 * pi / single(projectionData.ProjectionNumberPerTurn()) + projectionData.StartSourceAngle();
viewangles = viewangles(view_idx);
if strcmp(flag_fw, 'Backprojection')
    if any(~(size(sino) == [projectionData.getDetResoV(), projectionData.getDetResoU(), numel(view_idx)]))
        error('Sino dimensionality appears incorrect; it should be row-ordered');
    end
end
% We do not support upsampling in column direction at present
upN = 1;
if projectionData.IsArcDetector() % 3rd gen CT curved detector
    alphas=((1:single(projectionData.getTotalProjectionNumber())*upN) - (projectionData.getDetCntIdxU()*upN+1))*projectionData.getDetSizeU()/geometry.getSourceToDetectorDistance()/upN;
    xds=sin(alphas)*geometry.getSourceToDetectorDistance();
    yds=geometry.getSourceToCenterDistance() - geometry.getSourceToDetectorDistance()*cos(alphas);
else % flat pannel detector
    % FUL 2014-05-27 add support for flat panel detector
    printf( 'flat detector\n');
    xds = ((1:single(cfg.acq.total_col_count)*upN) - (cfg.recon.col_center*upN+1))*cfg.acq.col_width/upN;
    yds = (geometry.getSourceToCenterDistance() - geometry.getSourceToDetectorDistance()) * ones(size(xds));
end
zds=((1:single(projectionData.DetResoV()))-single(projectionData.DetResoV()+1)*0.5)*projectionData.DetResoV();

xoffset = imageData.getImgCenterX();
yoffset = imageData.getImgCenterY();
zoffset = imageData.getImgCenterZ();




end