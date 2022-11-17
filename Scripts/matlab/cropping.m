function [I_cropped] = cropping(I)
%algoritmo di cropping basato su:
%1:calcolo della circolarità delle strutture binarie
%2:selezione degli elementi effettivamenti circolari
%3:selezione dell'elemento circolare maggiore
imageSizeX = size(I,2);
imageSizeY = size(I,1);
I_r = im2double(I(:,:,1));

%% binarization

th = 0.18;

bw = imbinarize(I_r,th);

row1 = size(I,1);
col1 = size(I,2);
n_pixel1 = row1*col1;
prop = 0.00103319; %fattore proporzionale necessario per rendere universale il crop ed indipendente dalla dimensione dell'immagine

dim = round((n_pixel1*prop)/100);

se = strel('disk',dim);
bw_p = imclose(bw,se); %chiusura per riempire la maschera
bw_p = imopen(bw_p,se); %apertura per smussare i contorni


%% calcolo circolarità

stats = regionprops(bw_p,'Centroid','MajorAxisLength','MinorAxisLength','Area','Perimeter');

dim = size(stats,1);
circolarita = [];

for i = 1:dim
    perimetro = stats(i).Perimeter;
    area = stats(i).Area;
    circolarita(i) = (perimetro^2)/(4*pi*area);
end

%% selezione elementi circolari

th_L = 0.6; %soglia inferiore
th_H = 1.8; %soglia superiore

indici = [];
assi = [];

circolarita = circolarita';
k = 1;

for i = 1:dim
    if circolarita(i) < th_H && circolarita(i) > th_L
        assi(k,1) = stats(i).MajorAxisLength;
        assi(k,2) = stats(i).MinorAxisLength;
        indici(k) = i; 
        k = k+1;
    end
end

%la variabile indici mi indica la posizione all'interno della struttura
%stats

diametri = mean(assi,2);
raggi = diametri./2;

%% selezione elemento circolare maggiore

[M,index] = max(diametri);
posizione = indici(index);

diametro = diametri(index);
raggio = diametro/2;

centroidi = cat(1,stats.Centroid);
centro = stats(posizione).Centroid;


%% cropping

[col row] = meshgrid(1:imageSizeX, 1:imageSizeY);

circlepixel = (row - centro(2)).^2 + ...
    + (col - centro(1)).^2 <= raggio.^2;

xmin = centro(1)-raggio;
ymin = centro(2)-raggio;
width = diametro;
height = diametro;

circlemask = cat(3, circlepixel, circlepixel,circlepixel);
I_circle = I.*(uint8(circlemask));

I_cropped = imcrop(I_circle,[xmin ymin width height]);

end
