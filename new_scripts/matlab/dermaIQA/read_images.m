function [imagesList] = read_images(path, imageFormat)

% Function to read images from folder %


% Specify the folder where the files live.
% myFolder = 'D:/DOMI/University/Thesis/Inference/results';
myFolder = path;
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
    uiwait(warndlg(errorMessage));
    myFolder = uigetdir(); % Ask for a new one.
    if myFolder == 0
         % User clicked Cancel
         return;
    end
end

% Get a list of all files in the folder with the desired file name pattern.
% imageFormat = '*.jpg';
filePattern = fullfile(myFolder, imageFormat); % Change to whatever pattern you need.
theFiles = dir(filePattern);
imagesList = cell(1,length(theFiles));
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    % Now do whatever you want with this file name,
    % such as reading it in as an image array with imread()
    imageArray = imread(fullFileName);
    imagesList{k} = imageArray;
%     imshow(imageArray);  % Display image.
%     drawnow; % Force display to update immediately.
end