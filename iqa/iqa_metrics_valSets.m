clear variables
close all
clc

tic

%% ========== Utils ==========%
gt_root = "D:/DOMI/University/Thesis/Coding/Visualization/derma_v1/lowResolution_512_try/";
sr_root = "D:/DOMI/University/Thesis/Coding/Visualization/derma_v1/visualization_reOrdered/";

testSet = "Validation/";

cont_notTestSet = 0;
subfolders = dir(sr_root);

num_iterations = 4;   % number of iterations to process
step_iterations = floor(length(subfolders)/num_iterations);

iterations = [];
for m = 1 : step_iterations : length(subfolders)
    iteration = subfolders(m).name;
    if ( iteration(1) ~= '.' )
        iterations = [iterations; str2double(iteration)];
    end
end
iterations = sort(iterations,"ascend");

stats_metrics = table();
parfor idx_subfolder = 1 : length(iterations)
    iteration = iterations(idx_subfolder);
    if ( iteration(1) ~= '.' )
        path = fullfile(subfolders(idx_subfolder).folder, num2str(iteration));
        images_filename = dir(path);
        fprintf(1, "\n================");
        fprintf(1, '\nIteration %d', iteration);

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
%                 array_mssim(idx_image-cont_notFilename) = mean(multissim(sr_image, gt_image));
                tmp_ch1 = multissim(sr_image(:,:,1), gt_image(:,:,1));
                tmp_ch2 = multissim(sr_image(:,:,2), gt_image(:,:,2));
                tmp_ch3 = multissim(sr_image(:,:,3), gt_image(:,:,3));
                array_mssim(idx_image-cont_notFilename) = (tmp_ch1+tmp_ch2+tmp_ch1)/3;
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
        
        % Metrics
        stats_metrics_tmp = table( iteration, mean(array_psnr), mean(array_ssim), mean(array_mssim), mean(array_fsim),...
        mean(array_niqe_gt), mean(array_niqe_sr), mean(array_piqe_gt), mean(array_piqe_sr),...
        mean(array_brisque_gt), mean(array_brisque_sr) );
        stats_metrics_tmp.Properties.VariableNames = ["iter","psnr","ssim","mssim","fsim","niqe_gt","niqe_sr","piqe_gt","piqe_sr","brisque_gt","brisque_sr"];
        
        stats_metrics = [stats_metrics_tmp; stats_metrics];
    else
        cont_notTestSet = cont_notTestSet+1;
    end
end

timeElapsed = toc/60 %min          


%% ========== Save files ==========%

%----- All metrics: -----%
root_metrics = "D:/DOMI/University/Thesis/Results/excel_metrics/";
path_metrics = (root_metrics + testSet);
mkdir(path_metrics);

% Best Iteration:
NRorder_stats_metrics = sortrows(stats_metrics, ["niqe_sr","piqe_sr"], "ascend");
FRorder_stats_metrics = sortrows(stats_metrics, ["ssim","psnr"], "descend");

row_limit = 3; %thresh where to search for best iteration

for w = 1 : size(FRorder_stats_metrics,1)
    if any(table2array(FRorder_stats_metrics(w,1)) == table2array(NRorder_stats_metrics(1:row_limit,1)))
        best_iteration_table = FRorder_stats_metrics(w,:);
        break
    end
end

if ( abs(best_iteration_table.("ssim")-table2array(NRorder_stats_metrics(1,"ssim"))) ...
       <= abs(best_iteration_table.("niqe_sr")-table2array(NRorder_stats_metrics(1,"niqe_sr"))) )
    best_iteration_table = NRorder_stats_metrics(1,:);
end

best_iteration = best_iteration_table.("iter");
fprintf(1, "\nBest Iteration NR: %d", best_iteration);

stats_metrics_xlsx =  path_metrics+"/stats_metrics.xlsx";
writetable([NRorder_stats_metrics; best_iteration_table], stats_metrics_xlsx);


