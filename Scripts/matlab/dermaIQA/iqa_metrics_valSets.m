clear variables
close all
clc

tic

%% ========== Utils ==========%
addpath('Ma_score');
addpath('Ma_score/external/matlabPyrTools','Ma_score/external/randomforest-matlab/RF_Reg_C');

gt_root = "D:/DOMI/University/Thesis/Coding/Visualization/derma_v1/lowResolution_512_try/";
sr_root = "D:/DOMI/University/Thesis/Coding/Visualization/derma_v1/visualization_reOrdered/";

cont_notTestSet = 0;
subfolders = dir(sr_root);
for idx_subfolder = 1 : length(subfolders)
    iteration = subfolders(idx_subfolder).name;
    
    if iteration(1) ~= '.'
        path = fullfile(subfolders(idx_subfolder).folder, iteration);
        images_filename = dir(path);
        fprintf(1, '\nIteration %s', iteration);

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
                sr_image_path = fullfile(images_filename(idx_image).folder, image_filename);
                gt_extension = split(image_filename,'.');
                gt_filename_tmp = split(image_filename,'_');
                gt_filename = strcat(gt_filename_tmp(1),"_",gt_filename_tmp(end-1),".",gt_extension(end));
                gt_image_path = fullfile(gt_root, gt_filename);

                %% ========== Read images ==========%
                fprintf(1, '\nNow reading %s\n', gt_filename);
                gt_image = imread(gt_image_path);
                fprintf(1, 'Now reading %s\n', image_filename);
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
                array_mssim(idx_image-cont_notFilename) = mean(multissim(sr_image, gt_image));
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
        stats_metricsFR = ["metrics","mean","std","per_5%","perc_95%"; ... 
        "psnr",mean(array_psnr), std(array_psnr), prctile(array_psnr, 5), prctile(array_psnr, 95); ...
        "ssim",mean(array_ssim), std(array_ssim), prctile(array_ssim, 5), prctile(array_ssim, 95); ...
        "mssim",mean(array_mssim), std(array_mssim), prctile(array_mssim, 5), prctile(array_mssim, 95); ...
        "fsim",mean(array_fsim), std(array_fsim), prctile(array_fsim, 5), prctile(array_fsim, 95)];
        stats_metricsFR = array2table(stats_metricsFR);

        stats_metricsNR_gt = ["metrics","mean","std","per_5%","perc_95%"; ... 
        "niqe",mean(array_niqe_gt), std(array_niqe_gt), prctile(array_niqe_gt, 5), prctile(array_niqe_gt, 95); ...
        "piqe",mean(array_piqe_gt), std(array_piqe_gt), prctile(array_piqe_gt, 5), prctile(array_piqe_gt, 95); ...
        "brisque",mean(array_brisque_gt), std(array_brisque_gt), prctile(array_brisque_gt, 5), prctile(array_brisque_gt, 95); ...
        "empty", 0, 0, 0, 0];
        stats_metricsNR_gt = array2table(stats_metricsNR_gt);

        stats_metricsNR_sr = ["metrics","mean","std","per_5%","perc_95%"; ... 
        "niqe",mean(array_niqe_sr), std(array_niqe_sr), prctile(array_niqe_sr, 5), prctile(array_niqe_sr, 95); ...
        "piqe",mean(array_piqe_sr), std(array_piqe_sr), prctile(array_piqe_sr, 5), prctile(array_piqe_sr, 95); ...
        "brisque",mean(array_brisque_sr), std(array_brisque_sr), prctile(array_brisque_sr, 5), prctile(array_brisque_sr, 95); ...
        "empty", 0, 0, 0, 0];
        stats_metricsNR_sr = array2table(stats_metricsNR_sr);

        stats_metrics = [stats_metricsFR, stats_metricsNR_gt, stats_metricsNR_sr];

%         stats_metricsFR_psnr.psnr.array = array_psnr;
%         stats_metricsFR_psnr.psnr.mean =  mean(array_psnr);
%         stats_metricsFR_psnr.psnr.std = std(array_psnr);
%         stats_metricsFR_psnr.psnr.inf = prctile(array_psnr, 5);
%         stats_metricsFR_psnr.psnr.sup = prctile(array_psnr, 95);
% 
%         stats_metricsFR_psnr.ssim.array = array_ssim;
%         stats_metricsFR_psnr.ssim.mean =  mean(array_ssim);
%         stats_metricsFR_psnr.ssim.std= std(array_ssim);
%         stats_metricsFR_psnr.ssim.inf = prctile(array_ssim, 5);
%         stats_metricsFR_psnr.ssim.sup = prctile(array_ssim, 95);
% 
%         stats_metricsFR_psnr.mssim.array = array_mssim;
%         stats_metricsFR_psnr.mssim.mean =  mean(array_mssim);
%         stats_metricsFR_psnr.mssim.std= std(array_mssim);
%         stats_metricsFR_psnr.mssim.inf = prctile(array_mssim, 5);
%         stats_metricsFR_psnr.mssim.sup = prctile(array_mssim, 95);
% 
%         stats_metricsFR_psnr.fsim.array = array_fsim;
%         stats_metricsFR_psnr.fsim.mean =  mean(array_fsim);
%         stats_metricsFR_psnr.fsim.std= std(array_fsim);
%         stats_metricsFR_psnr.fsim.inf = prctile(array_fsim, 5);
%         stats_metricsFR_psnr.fsim.sup = prctile(array_fsim, 95);
%         
%         % NR-metrics GT
%         stats_metricsNR_gt.niqe.array = array_niqe_gt;
%         stats_metricsNR_gt.niqe.mean =  mean(array_niqe_gt);
%         stats_metricsNR_gt.niqe.std= std(array_niqe_gt);
%         stats_metricsNR_gt.niqe.inf = prctile(array_niqe_gt, 5);
%         stats_metricsNR_gt.niqe.sup = prctile(array_niqe_gt, 95);
% 
%         stats_metricsNR_gt.piqe.array = array_piqe_gt;
%         stats_metricsNR_gt.piqe.mean =  mean(array_piqe_gt);
%         stats_metricsNR_gt.piqe.std= std(array_piqe_gt);
%         stats_metricsNR_gt.piqe.inf = prctile(array_piqe_gt, 5);
%         stats_metricsNR_gt.piqe.sup = prctile(array_piqe_gt, 95);
% 
%         stats_metricsNR_gt.brisque.array = array_brisque_gt;
%         stats_metricsNR_gt.brisque.mean =  mean(array_brisque_gt);
%         stats_metricsNR_gt.brisque.std= std(array_brisque_gt);
%         stats_metricsNR_gt.brisque.inf = prctile(array_brisque_gt, 5);
%         stats_metricsNR_gt.brisque.sup = prctile(array_brisque_gt, 95);
%         
%         % NR-metrics SR
%         stats_metricsNR_sr.niqe.array = array_niqe_sr;
%         stats_metricsNR_sr.niqe.mean =  mean(array_niqe_sr);
%         stats_metricsNR_sr.niqe.std= std(array_niqe_sr);
%         stats_metricsNR_sr.niqe.inf = prctile(array_niqe_sr, 5);
%         stats_metricsNR_sr.niqe.sup = prctile(array_niqe_sr, 95);
% 
%         stats_metricsNR_sr.piqe.array = array_piqe_sr;
%         stats_metricsNR_sr.piqe.mean =  mean(array_piqe_sr);
%         stats_metricsNR_sr.piqe.std= std(array_piqe_sr);
%         stats_metricsNR_sr.piqe.inf = prctile(array_piqe_sr, 5);
%         stats_metricsNR_sr.piqe.sup = prctile(array_piqe_sr, 95);
% 
%         stats_metricsNR_sr.brisque.array = array_brisque_sr;
%         stats_metricsNR_sr.brisque.mean =  mean(array_brisque_sr);
%         stats_metricsNR_sr.brisque.std= std(array_brisque_sr);
%         stats_metricsNR_sr.brisque.inf = prctile(array_brisque_sr, 5);
%         stats_metricsNR_sr.brisque.sup = prctile(array_brisque_sr, 95);

        %% ========== Save files ==========%

%         %----- All metrics: -----%
%         root_metrics = "D:/DOMI/University/Thesis/Results_resume/excel_metrics/";
%         path_metrics = root_metrics + iteration;
%         stats_metrics_xlsx =  path_metrics+"/stats_metrics.xlsx";
%         writetable(stats_metrics, stats_metrics_xlsx);

    else
        cont_notTestSet = cont_notTestSet+1;
    end
end

timeElapsed = toc/60 %min          



