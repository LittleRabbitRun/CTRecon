cfg.sid = 538;
cfg.sdd = 946;
cfg.startZPos = 0;
cfg.DNU = 888;
cfg.DNV = 64;
cfg.colSize = 1.0239;
cfg.rowSize = 1.0963;
cfg.colOffset = 0;
cfg.rowOffset = 0;

cfg.PN = 984 * 18;
cfg.startView = 0;
cfg.ViewPerRot = 984;
cfg.PITCH = 0;
cfg.XN = 512;
cfg.YN = 512;
cfg.ZN = 512;

cfg.dx = 500 / cfg.XN;
cfg.dz = cfg.dx;

cfg.imgXCtr = 0;
cfg.imgYCtr = 0;
cfg.imgZCtr = 0;%cfg.ZN / 2 * cfg.dz;

% 
% if strcmp(flag_fw,'fp_bl')
%     mode = 0;
% elseif strcmp(flag_fw,'fp_pd')
%     mode = 1;
% elseif strcmp(flag_fw,'bp_bl')
%     mode = 2;
% elseif strcmp(flag_fw,'bp_pd')
%     mode = 3;
% else
%     mode = -1;
% end
% img = single(ones(cfg.ZN, cfg.XN, cfg.YN));
prj = single(ones(cfg.DNV, cfg.DNU, cfg.PN));
%view_ind = 1 : 4 : cfg.PN;
%numViews = [400, 300, 284] / 4;
%numSlice = [32,16,16];
%mask = uint8(ones(cfg.XN, cfg.YN));

img = DD_MultiGPU('bp_pd', cfg, prj);