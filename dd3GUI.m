function dd3GUI(mode, projModel, projectionData, imageData, view_idx, mask)
% DD3GUI Check that the data must be Z-first ordered arranged
if strcmp(mode, 'Projection')
    img = imageData.getImage();
    % Change the order of the image 
    img = permute(img, [3 1 2]);
    
elseif strcmp(mode, 'BackProjection')
    sino = projectionData.getProjection();
    sino = permute(sino, [2 1 3]);
else
end
x0 = projectionData.getSourceX();
% We do not support revolution CT that the detector contains GAP, this
% field should be in projectionData object GEOM_REVO_GAP = 0 or 1
% We do not support shift of the source along x direction
% x0 = x0 + shift_fs_mm;
% We can support PARALLEL MODEL BY SETTING y0 = 10e10 or something similar
% with that.
y0 = projectionData.getSourceY();
z0 = projectionData.getSourceZ();
if ~exist('view_idx', 'var') || isempty(view_idx)
    view_idx = 1:projectionData.getTotalProjectionNumber();
end
viewangles = projectionData.getViewAngles();
viewangles = viewangles(view_idx);
if strcmp(mode, 'BackProjection')
    if any(~(size(sino) == [projectionData.getDetResoV(), projectionData.getDetResoU(), numel(view_idx)]))
        error('Sino dimensionality appears incorrect; it should be row-ordered');
    end
end
% We do not support upsampling in column direction at present
xds = projectionData.getXDS();
yds = projectionData.getYDS();
zds = projectionData.getZDS();
% 
% if projectionData.IsArcDetector() % 3rd gen CT curved detector
%     alphas=((1:single(projectionData.getTotalProjectionNumber())*upN) - (projectionData.getDetCntIdxU()*upN+1))*projectionData.getDetSizeU()/geometry.getSourceToDetectorDistance()/upN;
%     xds=sin(alphas)*geometry.getSourceToDetectorDistance();
%     yds=geometry.getSourceToCenterDistance() - geometry.getSourceToDetectorDistance()*cos(alphas);
% else % flat pannel detector
%     % FUL 2014-05-27 add support for flat panel detector
%     printf( 'flat detector\n');
%     xds = ((1:single(cfg.acq.total_col_count)*upN) - (cfg.recon.col_center*upN+1))*cfg.acq.col_width/upN;
%     yds = (geometry.getSourceToCenterDistance() - geometry.getSourceToDetectorDistance()) * ones(size(xds));
% end
% zds=((1:single(projectionData.DetResoV()))-single(projectionData.DetResoV()+1)*0.5)*projectionData.DetPixelSizeV();

cen_y = imageData.getImgCenterX();
cen_x = imageData.getImgCenterY();
zoffset = imageData.getImgCenterZ();

%zshifts=(viewangles-projectionData.getStartSourceAngle()/180*pi)/2/pi*single(projectionData.getPitch())*cfg.acq.col_height_at_iso*single(projectionData.getDetResoV()) + projectionData.getStartZPos();
zshifts=(viewangles-projectionData.getStartSourceAngle()/180*pi)/2/pi*single(projectionData.getPitch()) + projectionData.getStartZPos();
xy_pixel_size = single(imageData.getPixelSizeX()); %TODO: to handle non_square image

if( ~exist('mask','var') || isempty(mask) )
    mask = int8(ones(imageData.getResoX(),imageData.getResoY(),2));
else
    mask = int8([mask,mask']);
end

% We support only one GPU now
gpu_id = 0;
% We only support GPU now:
if strcmp(mode, 'Projection')
    if strcmp(projModel, 'Branchless')
        prjMode = 0;
    elseif strcmp(projModel, 'VolumeRendering')
        prjMode = 1;
    elseif strcmp(projModel, 'SoftInterp')
        prjMode = 2;
    elseif strcmp(projModel, 'PseudoDD')
        prjMode = 3;
    elseif strcmp(projModel, 'Branches')
        prjMode = 4;
    else
        prjMode = 0;
    end

    viewangles = viewangles + pi/2;
    xoffset=-cen_y;
    yoffset=cen_x;
    sino = DD3_GPU('Proj', ...
        single(x0), single(y0), single(z0), int32(projectionData.getDetResoU()), int32(projectionData.getDetResoV()), ...
        single(xds), single(yds), single(zds),  ...
        single(xoffset), single(yoffset), single(zoffset), ...
        single(viewangles), single(zshifts), int32(length(view_idx)), ...
        int32(imageData.getResoX()), int32(imageData.getResoY()), int32(imageData.getResoZ()), ...
        single(img), ...
        single(xy_pixel_size), single(imageData.getPixelSizeZ()), mask, gpu_id, int32(prjMode));
    sino = permute(sino, [2 1 3]);
    projectionData.setProjection(double(sino));
elseif strcmp( mode, 'BackProjection' )
    if strcmp(projModel, 'Branchless')
        prjMode = 0;
    elseif strcmp(projModel, 'VolumeRendering')
        prjMode = 1;
    elseif strcmp(projModel, 'SoftInterp')
        prjMode = 2;
    elseif strcmp(projModel, 'ZLine')
        prjMode = 3;
    elseif strcmp(projModel, 'Branches')
        prjMode = 4;
    else
        prjMode = 0;
    end
    viewangles = viewangles + pi/2;
    xoffset=-cen_y; % the GPU implementation has transposed coordinate system, so flip x and y axis
    yoffset=cen_x;
    img = DD3_GPU('Back',...
        single(x0), single(y0), single(z0), int32(projectionData.getDetResoU()), int32(projectionData.getDetResoV()), ...
        single(xds), single(yds), single(zds),  ...
        single(xoffset), single(yoffset), single(zoffset), ...
        single(viewangles), single(zshifts), int32(length(view_idx)), ...
        int32(imageData.getResoX()), int32(imageData.getResoY()), int32(imageData.getResoZ()), ...
        single(sino), ...
        single(xy_pixel_size), single(imageData.getPixelSizeZ()), mask, gpu_id, ...
        int32(0), int32(prjMode));
    img = permute( img, [3 2 1]);
    imageData.setImage(double(img));
end
end
