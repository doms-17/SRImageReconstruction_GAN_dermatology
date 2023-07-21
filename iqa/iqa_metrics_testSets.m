clear variables
close all
clc

tic

%% ========== Utils ==========%
gt_root = "";  % path of Ground Truth (GT)
sr_root = ""; % path of Super Resolution (SR)

root_metrics = "";  % path metrics saving

testSets_name = ["Nurugo","PH2","Atlas"]; % testSets name

cont_notTestSet = 0;
subfolders = dir(gt_root);
parfor idx_subfolder = 1 : length(subfolders)
    testSet = subfolders(idx_subfolder).name;
    
    if ( testSet(1) ~= '.' && any(strcmp(testSet, testSets_name)) ) 
        path = fullfile(subfolders(idx_subfolder).folder, testSet);
        images_filename = dir(path);
        fprintf(1, "\n================")
        fprintf(1, '\nTestSet %s', testSet)

        %----- Inizialization -----%
        % Full-Reference Metrics
        array_psnr = zeros(1, length(images_filename));
        array_ssim = zeros(1, length(images_filename));
        array_mssim = zeros(1, length(images_filename));
        array_fsim = zeros(1, length(images_filename));

        % Non-Reference Metrics: Ground Truth images
        array_niqe_gt = zeros(1, length(images_filename));
        array_piqe_gt = zeros(1, length(images_filename));
        array_brisque_gt = zeros(1, length(images_filename));
        %                 gt_array_ma = 0;

        % Non-Reference Metrics: Super Resolution images
        array_niqe_sr = zeros(1, length(images_filename));
        array_piqe_sr = zeros(1, length(images_filename));
        array_brisque_sr = zeros(1, length(images_filename));
        %                 sr_array_ma = 0;

        cont_notFilename = 0;
        for idx_image = 1 : length(images_filename)
            image_filename = images_filename(idx_image).name;
            if image_filename(1) ~= '.'
                gt_image_path = fullfile(images_filename(idx_image).folder, image_filename);
                sr_filename_tmp = split(image_filename,'.') ;
                sr_filename = strcat(sr_filename_tmp(1), "_out.", sr_filename_tmp(end));  %change _out with other suffix
                sr_image_path = fullfile(sr_root, testSet, sr_filename);

                %% ========== Read images ==========%
                fprintf(1, '\nNow reading %s - %s (iter %d/%d)\n', testSet, image_filename, idx_image-2, length(images_filename));
                gt_image = imread(gt_image_path);
                fprintf(1, 'Now reading %s\n', sr_filename);
                sr_image = imread(sr_image_path);
                %                 imageArray = imresize(imageArray,scale);

                %% ========== Calculate metrics ==========%

                %----- Calculation -----%
                fprintf(1, 'Starting metrics\n');
                array_niqe_gt(idx_image-cont_notFilename) = niqe(gt_image);
                array_piqe_gt(idx_image-cont_notFilename) = piqe(gt_image);
                array_brisque_gt(idx_image-cont_notFilename) = brisque(gt_image);
                %     gt_ma = quality_predict(gt_image);

                array_niqe_sr(idx_image-cont_notFilename) = niqe(sr_image);
                array_piqe_sr(idx_image-cont_notFilename) = piqe(sr_image);
                array_brisque_sr(idx_image-cont_notFilename) = brisque(sr_image);
                %     sr_ma = quality_predict(sr_image);

                [rows_gt, columns_gt, ~] = size(gt_image);
                [rows_sr, columns_sr, ~] = size(sr_image);
                scale = rows_sr/rows_gt;
                if (rows_gt ~= rows_sr && columns_gt ~= columns_sr)
                    gt_image = imresize(gt_image, scale, "bicubic");
                end

                array_psnr(idx_image-cont_notFilename) = psnr(rgb2ycbcr(sr_image), rgb2ycbcr(gt_image));
                array_ssim(idx_image-cont_notFilename) = ssim(sr_image, gt_image);
%                 array_mssim(idx_image-cont_notFilename) = mean(multissim(sr_image, gt_image));
                tmp_ch1 = multissim(sr_image(:,:,1), gt_image(:,:,1));
                tmp_ch2 = multissim(sr_image(:,:,2), gt_image(:,:,2));
                tmp_ch3 = multissim(sr_image(:,:,3), gt_image(:,:,3));
                array_mssim(idx_image-cont_notFilename) = (tmp_ch1+tmp_ch2+tmp_ch3)/3;
                array_fsim(idx_image-cont_notFilename) = fsim(sr_image, gt_image);

                fprintf(1, 'Finished!\n');

            else
                cont_notFilename = cont_notFilename+1;
            end
        end

        %% ========== Calculate stats ==========%
        % Deleting empty values
        array_psnr = array_psnr(array_psnr~=0);
        array_ssim = array_ssim(array_ssim~=0);
        array_mssim = array_mssim(array_mssim~=0);
        array_fsim = array_fsim(array_fsim~=0);
        array_niqe_gt = array_niqe_gt(array_niqe_gt~=0);
        array_piqe_gt = array_piqe_gt(array_piqe_gt~=0);
        array_brisque_gt = array_brisque_gt(array_brisque_gt~=0);
        array_niqe_sr = array_niqe_sr(array_niqe_sr~=0);
        array_piqe_sr = array_piqe_sr(array_piqe_sr~=0);
        array_brisque_sr = array_brisque_sr(array_brisque_sr~=0);
        
        % FR-metrics
        stats_metricsFR = ["psnr",mean(array_psnr), std(array_psnr), prctile(array_psnr, 5), prctile(array_psnr, 95); ...
        "ssim",mean(array_ssim), std(array_ssim), prctile(array_ssim, 5), prctile(array_ssim, 95); ...
        "mssim",mean(array_mssim), std(array_mssim), prctile(array_mssim, 5), prctile(array_mssim, 95); ...
        "fsim",mean(array_fsim), std(array_fsim), prctile(array_fsim, 5), prctile(array_fsim, 95)];
        stats_metricsFR = array2table(stats_metricsFR);

        stats_metricsNR_gt = ["niqe",mean(array_niqe_gt), std(array_niqe_gt), prctile(array_niqe_gt, 5), prctile(array_niqe_gt, 95); ...
        "piqe",mean(array_piqe_gt), std(array_piqe_gt), prctile(array_piqe_gt, 5), prctile(array_piqe_gt, 95); ...
        "brisque",mean(array_brisque_gt), std(array_brisque_gt), prctile(array_brisque_gt, 5), prctile(array_brisque_gt, 95); ...
        "ma", 0, 0, 0, 0];
        stats_metricsNR_gt = array2table(stats_metricsNR_gt);

        stats_metricsNR_sr = ["niqe", mean(array_niqe_sr), std(array_niqe_sr), prctile(array_niqe_sr, 5), prctile(array_niqe_sr, 95); ...
        "piqe", mean(array_piqe_sr), std(array_piqe_sr), prctile(array_piqe_sr, 5), prctile(array_piqe_sr, 95); ...
        "brisque", mean(array_brisque_sr), std(array_brisque_sr), prctile(array_brisque_sr, 5), prctile(array_brisque_sr, 95); ...
        "ma", 0, 0, 0, 0];
        stats_metricsNR_sr = array2table(stats_metricsNR_sr);

        stats_metrics = [stats_metricsFR, stats_metricsNR_gt, stats_metricsNR_sr];
        stats_metrics.Properties.VariableNames = ["metricsFR","mean_FR","std_FR","per_5%_FR","perc_95%_FR",...
            "metricsNR_GT","mean_GT","std_GT","per_5%_GT","perc_95%_GT",...
            "metricsNR_SR","mean_SR","std_SR","per_5%_SR","perc_95%_SR"
        ];

        array_metrics = table(array_psnr',array_ssim',array_mssim',array_fsim', array_niqe_gt',array_niqe_sr',array_piqe_gt',array_piqe_sr',array_brisque_gt',array_brisque_sr');
        array_metrics.Properties.VariableNames = ["psnr","ssim","mssim","fsim","niqe_gt","niqe_sr","piqe_gt","piqe_sr","brisque_gt","brisque_sr"];
        %% ========== Save files ==========%

        %----- All metrics: -----%
        path_metrics = (root_metrics + testSet);
        mkdir(path_metrics);

        stats_metrics_xlsx =  path_metrics+"/stats_metrics.xlsx";
        writetable(stats_metrics, stats_metrics_xlsx);
        array_metrics_xlsx =  path_metrics+"/array_metrics.xlsx";
        writetable(array_metrics, array_metrics_xlsx);

    else
        cont_notTestSet = cont_notTestSet+1;
    end
end

timeElapsed = toc/60 %min          


