function cfg = wrapperFromProjectionAndReconImageToCfg(projectionData, imageData)
% all in unit of mm, except otherwise noted
cfg.acq.sid = projectionData.getSourceToCenterDistance();
cfg.acq.sdd = projectionData.getSourceToDetectorDistance();
cfg.acq.views_per_rotation = projectionData.ProjectionNumberPerTurn();
cfg.acq.num_data_views = projectionData.getTotalProjectionNumber();
cfg.acq.row_count = projectionData.getDetResoV();
cfg.acq.total_col_count = projectionData.getDetResoU();
cfg.acq.col_width = projectionData.DetPixelSizeU();
cfg.acq.col_height = projectionData.DetPixelSizeV();
cfg.acq.rotation_direction = 1; % NOTE: this parameter is currently ignored in dd3
cfg.acq.first_view_angle = projectionData.getStartSourceAngle(); % in the unit of degree
cfg.acq.first_view_zposition = projectionData.getStartZPos();
cfg.acq.helical_pitch = projectionData.getPitch();
% optional parameters (automatically provided if obtained from real scan files)
cfg.acq.col_height_at_iso = cfg.acq.sid / cfg.acq.sdd * cfg.acq.col_height;

cfg.recon.col_center = projectionData.getDetCntIdxU(); % one-based detector index
cfg.recon.row_center = projectionData.getDetCntIdxV(); % one-based detector index. NOTE: this parameter is currently ignored in dd3
cfg.recon.recon_pixels_z = imageData.getResoZ();
cfg.recon.recon_pixels_y = imageData.getResoY();
cfg.recon.recon_pixels_x = imageData.getResoX();
cfg.recon.dfov_mm = imageData.pixelSizeX() * imageData.getResoX();
cfg.recon.recon_slice_spacing = imageData.getPixelSizeZ();
cfg.recon.recon_center_z = 0;
cfg.recon.flat_det = ~projectionData.getIsArcDetector();

end