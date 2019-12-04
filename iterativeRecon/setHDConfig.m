function [cfg] = setHDConfig(rotN,DNU,DNV,XN,YN,ZN)

cfg.acq.sid = 541;
cfg.acq.sdd = 949;
cfg.acq.views_per_rotation = rotN;
cfg.acq.num_data_views = rotN;
cfg.acq.row_count = DNV;
cfg.acq.total_col_count = DNU;
cfg.acq.col_width = 1.0239 / 888 * DNU;
cfg.acq.col_height = 1.096439 / 64 * DNV;
cfg.acq.rotation_direction = 1; % NOTE: this parameter is currently ignored in dd3
cfg.acq.first_view_angle = 0; % in the unit of degree
cfg.acq.first_view_zposition = 0;
cfg.acq.helical_pitch = 0;
% optional parameters (automatically provided if obtained from real scan files)
cfg.acq.col_height_at_iso = 0.625;

cfg.recon.col_center = (DNU+1) / 2; % one-based detector index
cfg.recon.row_center = (DNV+1) / 2; % one-based detector index. NOTE: this parameter is currently ignored in dd3
cfg.recon.recon_pixels_z = ZN;
cfg.recon.recon_pixels_y = XN;
cfg.recon.recon_pixels_x = YN;
cfg.recon.dfov_mm = 250 ;
cfg.recon.recon_slice_spacing = 0.625 / 64 * ZN;
cfg.recon.recon_center_z = 0;

end