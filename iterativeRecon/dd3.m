%-------------------------------------------------------------------------
% GE Confidential. General Electric Proprietary Data (c) 2015 General Electric Company
% Date: Feb, 23, 2015
% Routine: dd3.m
%
% Authors:
%   Rui Liu (Wake Forest University), Lin Fu    (GE Global Research)
%
% Organization:
%    GE Global Research
%
% Aim:
%   This is a high level Matlab/Freemat wrapper function for the distance driven (DD)
%	forward and back projectors. Depending on a user specified string of projector type,
%	it correspondingly calls lower level multithreaded CPU implementation
%	(DD3_mex -> DD3Proj_roi_notrans_mm_thread_c) or GPU implementation (DD3_GPU_mex)
%
% Inputs/Outputs:
%
% flag_fw is a string parameter taking the following allowed options
%
%	"fp' / 'bp' - CPU reference projector/back projector
%
%	'fp_gpu_branchless_vr -  highly simplified inaccurate distance driven, a single texture fetch per ray
%	'bp_gpu_branchless_vr'
%
%	'fp_gpu_branchless_sat2d' - correct DD model, with inaccuacies caused by HW interpolation and 2D SAT
%	'bp_gpu_branchless_sat2d'
%
%	'fp_gpu_volume_rendering' - volume renendering ray tracing model
%	'bp_gpu_branchless_zline' - volume rendering based ray tracing model

% NOTE: THE MAPPING TABLE FOR GPU implementation
%   'gpu_branchless'       ---- 0
%   'gpu_pseudo_dd'        ---- 1
%   'gpu_soft_interp'      ---- 2
%   'gpu_zline'            ---- 3
%   'gpu_branches'         ---- 4
%
% cfg
%	contains geometric parameters of scanning geometry and dimensions of the image volume
%	catrecon format is used. Refer to 'dd3_demo.m' for an example of cfg
%
% in
%	input img or sinogram data. currently only 3D volume is supported.
%	Img or sinogram must have the z-dimension as the first dimension
%
% view_ind
%	optional. default to project all views specified in cfg
%	This allows to speicify a subset of views to project
%
% mask
%	Optional. a 2D binary mask may be provided so that certain region in the image will be ignored from projection
%
% History:
% 	2015-07-24 FUL, unified the wrapper of fw and bk dd3 projectors
%
% TODO:
%	verify compatibility with freemat
%	verify a 2D slice image can be handled (currently 2D case is not carefully handled)
%	Verify/clean up some legacy code for detector upsampling. This feature is not currently supported
%-------------------------------------------------------------------------

% function [sino,cfg]=dd3_reprojection(cfg,img,view_ind,mask)
% dd3_reprojection routine
% import(catrecon_lib_filename,'DD3Proj_roi_notrans_mm_thread','DD3Proj_roi_notrans_mm_thread_c','void','float x0, float y0, float z0, int32 nrdetcols, int32 nrdetrows, float[nrdetcols] &xds, float[nrdetcols] &yds, float[nrdetrows] &zds, float imgXoffset,float imgYoffset, float imgZoffset, float[nrviews] &viewangles, float[nrviews] &zshifts,int32 nrviews,float[nrviews*nrdetcols*nrdetrows] &sinogram,int32 nrcols, int32 nrrows, int32 nrplanes, float[nrrows*nrcols*nrplanes] &pOrig, float vox_xy_size, float vox_z_size, int8[2*nrrows*nrcols] &xy_mask, int32 cpunum');

function [out,cfg] = dd3( flag_fw, cfg, in, view_ind, mask, bp_type )

% Check the dimensionality of the image for correctness (the image should be in z-order)
if strcmp( flag_fw(1:2), 'fp' )
    img = in;
    if any(~(size(img)==[cfg.recon.recon_pixels_z cfg.recon.recon_pixels_x cfg.recon.recon_pixels_y]))
        error('Image dimensionality appears incorrect; it should be z-ordered.');
    end
elseif strcmp( flag_fw(1:2), 'bp' )
    sino = in;
    % FUL 2012-09-27, this safe guard is moved to a few lines below to support view_ind better
    % if any(~(size(sino)==[cfg.acq.row_count cfg.acq.total_col_count cfg.acq.num_data_views]))
    %     error('Sino dimensionality appears incorrect; it should be row-ordered.');
    % end
else
    error( 'must specify fp or bp projector');
end

x0 = cfg.acq.sid*0;

% FUL 2014-12-11, add revo support
if ~ isfield( cfg.recon, 'GEOM_REVO_GAP' ) || 0==cfg.recon.GEOM_REVO_GAP
    GEOM_REVO_GAP=0;
else
    GEOM_REVO_GAP=1;
end

% FUL 2012-11-30, add shift_fs_mm parameter
if ~exist('shift_fs_mm','var')
    shift_fs_mm = single(0);
end
x0 = x0 + shift_fs_mm;

% FUL 2014-05-27, add support for flat panel detector
if ~isfield( cfg.recon, 'flat_det' ) || ~cfg.recon.flat_det
    % if ~isset('flat_det')
    flat_det = 0;
else
    flat_det = 1;
end

if ~exist('bp_type','var') || isempty(bp_type)
    % bp_type 0 is regular, 1 is squared
    bp_type = 0;
end

if(isfield(cfg.recon,'para_mode') && cfg.recon.para_mode)
    if(isfield(cfg.recon,'para_scale'))
        y0 = cfg.acq.sid*cfg.recon.para_scale;
    else
        y0 = cfg.acq.sid*1e10;%just set to be very big number, for parallel beam mode
    end
    
else
    y0 = cfg.acq.sid;
end
z0 = 0;


if ~exist('view_ind','var') || isempty(view_ind)
    view_ind =1:cfg.acq.num_data_views;
end

viewangles = (0:(cfg.acq.num_data_views-1))*2*pi/single(cfg.acq.views_per_rotation) + cfg.acq.first_view_angle/180*pi;
viewangles = viewangles(view_ind);


if strcmp( flag_fw(1:2), 'bp' )
    % FUL 2012-09-27, new safe guard code for sinogram dimension
    if any( ~( size(sino)==[cfg.acq.row_count cfg.acq.total_col_count numel(view_ind)]))
        error('Sino dimensionality appears incorrect; it should be row-ordered.');
    end
end

%upsampling in column direction
upN=1;
if(isfield(cfg.recon,'proj_upsample_on') && cfg.recon.proj_upsample_on)
    upN = cfg.recon.filter_frequency_interpolation_factor;
end

if 0 == GEOM_REVO_GAP
    if ~flat_det % 3rd gen CT curved detector
        alphas=((1:single(cfg.acq.total_col_count)*upN) - (cfg.recon.col_center*upN+1))*cfg.acq.col_width/cfg.acq.sdd/upN;
        xds=sin(alphas)*cfg.acq.sdd;
        yds=cfg.acq.sid-cfg.acq.sdd*cos(alphas);
    else % flat pannel detector
        % FUL 2014-05-27 add support for flat panel detector
        printf( 'flat detector\n');
        xds = ((1:single(cfg.acq.total_col_count)*upN) - (cfg.recon.col_center*upN+1))*cfg.acq.col_width/upN;
        yds = (cfg.acq.sid - cfg.acq.sdd) * ones(size(xds));
    end
    zds=((1:single(cfg.acq.row_count))-single(cfg.acq.row_count+1)*0.5)*cfg.acq.col_height;
else
    printf( 'revo proj\n' );
    % FUL, modify cfg geom and adjust det locations for Revo gaps
    
    add_revo_gap;
    
end

if ~isfield(cfg.recon,'targcen_R') || ~any(cfg.recon.targcen_R) || ~isfield(cfg.recon,'targcen_A') || ~any(cfg.recon.targcen_C)
    cen_x = 0;
    cen_y = 0;
else
    [cen_x,cen_y] = xform_RA_to_XY(cfg.recon.targcen_R,cfg.recon.targcen_A,cfg.acq.patient_entry,cfg.acq.patient_position);
end
xoffset=cen_x;
yoffset=cen_y;
zoffset=cfg.recon.recon_center_z;

zshifts=(viewangles-cfg.acq.first_view_angle/180*pi)/2/pi*single(cfg.acq.helical_pitch)*cfg.acq.col_height_at_iso*single(cfg.acq.row_count) + cfg.acq.first_view_zposition;

xy_pixel_size = cfg.recon.dfov_mm/single(max(cfg.recon.recon_pixels_x,cfg.recon.recon_pixels_y)); %to handle non_square image

if( ~exist('mask','var') || isempty(mask) )
    mask = int8(ones(cfg.recon.recon_pixels_x,cfg.recon.recon_pixels_y,2));
else
    mask = int8([mask,mask']);
end


if length(flag_fw)>=6 && strcmp(flag_fw(4:6),'gpu')
    gpu_id = 0; % use the first gpu by default
elseif isfield( cfg, 'user' ) && isfield( cfg.user, 'max_num_threads')
    cpunum=cfg.user.max_num_threads;
else
    % FUL this line used to be out of the if statement and is suspicous of creating freemat hangup, disable it if user provides cpunum
    if isfm % freemat
        cpunum = system('grep -c processor /proc/cpuinfo');
        cpunum=int32( str2num(cpunum{1}) );
    else % matlab
        [not_used t] = system('grep -c processor /proc/cpuinfo');
        cpunum = int32( str2num(t) );
    end
end


if strcmp( flag_fw(1:2), 'fp' )
    if strcmp( flag_fw, 'fp' )
        sino = DD3_mex('Proj',...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count*upN), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds), ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(img), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, int32(cpunum) );
    elseif strcmp( flag_fw(1:3), 'fp1')
        sino = DD3_mex('Proj',...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count*upN), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds), ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(img), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, int32(1) );
    elseif strcmp( flag_fw(1:3), 'fp8')
        sino = DD3_mex('Proj',...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count*upN), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds), ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(img), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, int32(8) );
    elseif strcmp( flag_fw(1:6), 'fp_gpu' )
        if strcmp( flag_fw, 'fp_gpu_branchless' )
            prjMode = 0;
        elseif strcmp( flag_fw, 'fp_gpu_volume_rendering' )
            prjMode = 1;
        elseif strcmp( flag_fw, 'fp_gpu_soft_interp' )
            prjMode = 2;
        elseif strcmp( flag_fw, 'fp_gpu_pseudo_dd' )
            prjMode = 3;
        elseif strcmp( flag_fw, 'fp_gpu_branches')
            prjMode = 4;
        else
            prjMode = 0;
        end
        viewangles = viewangles + pi/2;
        img = permute( img, [1 3 2] ); % the GPU implementation has transposed coordinate system, so flip x and y axis
        xoffset=-cen_y;
        yoffset=cen_x;
        sino = DD3_GPU('Proj', ...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds),  ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(img), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, gpu_id, int32(prjMode) );
    end
elseif strcmp( flag_fw(1:2), 'bp' )
    if strcmp( flag_fw, 'bp' )
        img = DD3_mex('Back',...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count*upN), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds), ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(sino), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, int32(cpunum), ...
            int32(bp_type) );
    elseif strcmp( flag_fw(1:3), 'bp1')
        img = DD3_mex('Back',...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count*upN), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds), ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(sino), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, int32(1), ...
            int32(bp_type) );
    elseif strcmp( flag_fw(1:3), 'bp8')
        img = DD3_mex('Back',...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count*upN), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds), ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(sino), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, int32(8), ...
            int32(bp_type) );
    elseif strcmp( flag_fw(1:6), 'bp_gpu' )
        if strcmp( flag_fw, 'bp_gpu_branchless' )
            prjMode = 0;
        elseif strcmp( flag_fw, 'bp_gpu_pseudo_dd' )
            prjMode = 1;
        elseif strcmp( flag_fw, 'bp_gpu_soft_interp' )
            prjMode = 2;
        elseif strcmp( flag_fw, 'bp_gpu_zline' )
            prjMode = 3;
        elseif strcmp( flag_fw, 'bp_gpu_branches')
            prjMode = 4;
        else
            prjMode = 0;
        end
        viewangles = viewangles + pi/2;
        xoffset=-cen_y; % the GPU implementation has transposed coordinate system, so flip x and y axis
        yoffset=cen_x;
        img = DD3_GPU('Back',...
            single(x0), single(y0), single(z0), int32(cfg.acq.total_col_count), int32(cfg.acq.row_count), ...
            single(xds), single(yds), single(zds),  ...
            single(xoffset), single(yoffset), single(zoffset), ...
            single(viewangles), single(zshifts), int32(length(view_ind)), ...
            int32(cfg.recon.recon_pixels_x), int32(cfg.recon.recon_pixels_y), int32(cfg.recon.recon_pixels_z), ...
            single(sino), ...
            single(xy_pixel_size), single(cfg.recon.recon_slice_spacing), mask, gpu_id, ...
            int32(bp_type), int32(prjMode));
        img = permute( img, [1 3 2] );
    end
end

if strcmp( flag_fw(1:2), 'fp' )
    if GEOM_REVO_GAP > 0  % FUL
        revo_gap_del;
    end
    out = sino;
else
    out = img;
end

