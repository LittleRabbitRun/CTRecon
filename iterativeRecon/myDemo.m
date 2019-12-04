
% load system cfg structure
	% all in unit of mm, except otherwise noted
	cfg.acq.sid = 541;
	cfg.acq.sdd = 949;
	cfg.acq.views_per_rotation = 984;
	cfg.acq.num_data_views = 984;
	cfg.acq.row_count = 64;
	cfg.acq.total_col_count = 888;
	cfg.acq.col_width = 1.0239;
	cfg.acq.col_height = 1.096439;
	cfg.acq.rotation_direction = 1; % NOTE: this parameter is currently ignored in dd3
	cfg.acq.first_view_angle = 0; % in the unit of degree
	cfg.acq.first_view_zposition = 0;
	cfg.acq.helical_pitch = 0;
	% optional parameters (automatically provided if obtained from real scan files)
	cfg.acq.col_height_at_iso = 0.625;

	cfg.recon.col_center = 444.75; % one-based detector index
	cfg.recon.row_center = 31.5; % one-based detector index. NOTE: this parameter is currently ignored in dd3
	cfg.recon.recon_pixels_z = 64;
	cfg.recon.recon_pixels_y = 512;
	cfg.recon.recon_pixels_x = 512;
	cfg.recon.dfov_mm = 500;
	cfg.recon.recon_slice_spacing = 500/512;
	cfg.recon.recon_center_z = 0;

   
    img = phantom(512);
    img = repmat(img, [1, 1, 64]);
    img = permute(img, [3 1 2]);
    tic;
    prj = dd3('fp_gpu_branchless', cfg, single(img));
toc;
tic;
    imgg = dd3('bp_gpu_branchless', cfg, single(ones(size(prj))));
    toc;
    imshow(squeeze(imgg(:,256,:)),[])

    mask = zeros(512,512);
    for ii = 1 : 512
        for jj = 1 : 512
            if (((ii - 0.5 - 256) / 256)^2 + ((jj - 0.5 - 256) / 256)^2) <0.99
                mask(ii,jj) = 1;
            end
        end
    end
    mask = int8(mask);
    osNum = 60;
    initImg = single(zeros(64,512,512));
    reconImg = conjgrad(initImg, prj, cfg, mask, 40);
    reconImg = os_sart(initImg, prj, cfg, mask, osNum, 40);