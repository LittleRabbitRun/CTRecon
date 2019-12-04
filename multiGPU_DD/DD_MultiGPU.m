%------------------------------------------------------------------------- 
% Wake Forest University Health Sciences
% Date: Sep, 13, 2016
% Routine: DD_MultiGPU.m
%
% Authors:
%   Rui Liu
%
% Organization: 
%   Wake Forest University Health Sciences
%
% Aim:
%   This is a high level Matlab/Freemat wrapper function for the distance driven (DD) 
%	forward and back projectors with multi-GPUs.
%
% Inputs/Outputs:
%
% flag_fw is a string parameter taking the following allowed options
%	'fp_bl' - (0) correct DD model, with inaccuacies caused by HW interpolation and 2D SAT
%   'fp_pd' - (1) pseudo DD model projection
%	'bp_bl' - (2) DD model backprojection
%   'bp_pd' - (3) pseudo DD model backprojection
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
% numViews
%   optional. default to project all views specified in cfg. This allows to
%   specifiy the number of views for each GPU.
%
% mask
%	Optional. a 2D binary mask may be provided so that certain region in the image will be ignored from projection
%
function [out,cfg] = DD_MultiGPU(flag_fw, cfg, in, view_ind, numViews, mask)

if strcmp(flag_fw,'fp_bl')
    mode = 0;
elseif strcmp(flag_fw,'fp_pd')
    mode = 1;
elseif strcmp(flag_fw,'bp_bl')
    mode = 2;
elseif strcmp(flag_fw,'bp_pd')
    mode = 3;
else
    mode = -1;
end

sid = cfg.sid;
x0 = 0;
y0 = cfg.sid;
z0 = cfg.startZPos;

sdd = cfg.sdd;

DNU = cfg.DNU;
DNV = cfg.DNV;

colSize = cfg.colSize; % column size of the detector cell
rowSize = cfg.rowSize; % row size of the detector cell
colOffset = cfg.colOffset;
rowOffset = cfg.rowOffset;


stepTheta = atan(colSize * 0.5 / sdd) * 2.0;

xds = zeros(DNU, 1);
yds = zeros(DNU, 1);
zds = zeros(DNV, 1);
for ii = 1 : DNU
    xds(ii, 1) = sin(((ii - 1.0) - (DNU - 1.0) * 0.5 + colOffset) * stepTheta) * sdd;
    yds(ii, 1) = sid - cos(((ii - 1.0) - (DNU - 1.0) * 0.5 + colOffset) * stepTheta) * sdd;
end

for ii = 1 : DNV
    zds(ii,1) = ((ii - 1.0) - (DNV - 1.0) * 0.5 + rowOffset) * rowSize;
end

imgXCenter = cfg.imgXCtr;
imgYCenter = cfg.imgYCtr;
imgZCenter = cfg.imgZCtr;

PN = cfg.PN;
startView = cfg.startView;
viewPerRot = cfg.ViewPerRot;
pitch = cfg.PITCH; % Definition in GE
hangs = zeros(PN, 1);
hzPos = zeros(PN, 1);

for ii = 1 : PN
    hangs(ii,1) = startView + (ii-1) * pi * 2.0 / viewPerRot;
    hzPos(ii,1) = (ii-1) / viewPerRot * 0.625 * pitch + z0;
end

XN = cfg.XN;
YN = cfg.YN;
ZN = cfg.ZN;

dx = cfg.dx;
dz = cfg.dz;

if ~exist('view_ind','var') || isempty(view_ind)
    view_ind = linspace(1,PN,PN)';
    shangs = hangs;
    shzPos = hzPos;
    SPN = PN;
else
    shangs = hangs(view_ind);
    shzPos = hzPos(view_ind);
    SPN = length(view_ind);
end


if ~exist('numViews','var') || isempty(numViews)
    gpuNum = 1;
    startPN = 0;
else
    gpuNum = length(numViews);
    startPN = [0,cumsum(numViews)];
    startPN = startPN(1:end-1);
end


if ~exist('mask','var') || isempty(mask)
   mask = ones(XN,YN);
end


if (mode == 0 || mode == 1) % projection
    
    out = DD_MultiGPU_ker(single(x0), single(y0), single(z0), int32(DNU), int32(DNV),...
    single(xds), single(yds), single(zds), single(imgXCenter), single(imgYCenter), single(imgZCenter),...
    single(shangs), single(shzPos), int32(SPN), int32(XN), int32(YN), int32(ZN),...
    single(in), single(dx), single(dz), uint8(mask), int32(mode), int32(startPN),...
    int32(gpuNum));
else
    in = in(:,:,view_ind);
    out = DD_MultiGPU_ker(single(x0), single(y0), single(z0), int32(DNU), int32(DNV),...
    single(xds), single(yds), single(zds), single(imgXCenter), single(imgYCenter), single(imgZCenter),...
    single(shangs), single(shzPos), int32(SPN), int32(XN), int32(YN), int32(ZN),...
    single(in), single(dx), single(dz), uint8(mask), int32(mode), int32(startPN),...
    int32(gpuNum));
end
