clear all
clc
close all
% current_dir = pwd;
% 
% preprocessingdir = current_dir(1:end-32);
% maindir = preprocessingdir(1:end-14);
% fs = filesep;
% %% Selezione cartella
% 
% answer = questdlg('Che lesioni vuoi processare?',...
%     'Check',...
%     'Nevo','Cheratosi','Melanoma','Nevo','Basalioma');
% 
% switch answer
%     case 'Nevo'
%           path_imm = convertStringsToChars([maindir fs 'Dataset' fs 'TEST DERMO' fs 'Nevo Benigno']);
%          
%     case 'Cheratosi'
%           path_imm = convertStringsToChars([maindir fs 'Dataset' fs 'TEST DERMO' fs 'Cheratosi Seborroica']);
%           
%     case 'Melanoma'
%           path_imm = convertStringsToChars([maindir fs 'Dataset' fs 'TEST DERMO' fs 'Melanoma Maligno']);
%        
%     case 'Basalioma'
%           path_imm = convertStringsToChars([maindir fs 'Dataset' fs 'TEST DERMO' fs 'Melanoma Maligno']);
%           
% end

% path1 = uigetdir();
% path1 = sprintf('%s%s',path1);

path_imm = 'D:\DOMI\University\Thesis\Coding\Dataset\TestSet_original\Novara dermoscopio trash';
elenco = dir([path_imm filesep '*.png']);
dim = length(elenco);

mkdir CROPPED

%% Processing

tic
for i=1:dim
     I = imread([path_imm filesep elenco(i).name]);
     
     [I_c] = cropping(I);
     
     imwrite(I_c,['CROPPED' filesep elenco(i).name]);
     i
end
toc