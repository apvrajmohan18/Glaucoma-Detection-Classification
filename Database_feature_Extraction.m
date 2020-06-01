clc
clear all
close all

cd Database

kk = pwd;
filePattern = fullfile(kk, '*.jpg') %identify jpg files
jpegFiles = dir(filePattern) %use dir to list jpg files
size1 = length(jpegFiles); % length of the size of the file

TT = [];
wt = waitbar(0,'Please Wait....');
for i=1:size1    
b=imread(strcat(num2str(i),'.jpg'));
cd 
r=b(:,:,1);

g=b(:,:,2);

bb=b(:,:,3);

             

% % % disckel

% th=graythresh(g);

ne=g>130;

binaryImage=ne;
% Get rid of stuff touching the border
binaryImage = imclearborder(binaryImage);

fill=imfill(binaryImage,'holes');
 
se=strel('disk',6);
dil=imdilate(fill,se);
 
ne=g>140;

binaryImage=ne;
% Get rid of stuff touching the border
binaryImage = imclearborder(binaryImage);

cup=imfill(binaryImage,'holes');
 
se1=strel('disk',2);
di=imdilate(cup,se1);
 
cup=di;
 

BW=binaryImage ;
CC = bwconncomp(BW);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
BW(CC.PixelIdxList{idx}) = 0;

filteredForeground=BW;

% % Fill holes in the blobs to make them solid.
 binaryImage = imfill(binaryImage, 'holes');
% % Display the binary image.

dis(:,:,1)=immultiply(binaryImage,b(:,:,1));

dis(:,:,2)=immultiply(binaryImage,b(:,:,2));

dis(:,:,3)=immultiply(binaryImage,b(:,:,3));


a = dil;
stats = regionprops(double(a),'Centroid',...
    'MajorAxisLength','MinorAxisLength')


centers = stats.Centroid;
diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
radii = diameters/2;

c1=bwarea(dil);
c2=bwarea(di);

cdr=c2./(c1);


rim=(1-di)-(1-dil);

RDR=bwarea(rim)./(c2);


nn=sprintf('The CDR is  %2f ',cdr);


nn1=sprintf('The RDR is  %2f ',RDR/2);

%% HOG Feature Extraction

if ndims(b)==3
    img1 = imresize(rgb2gray(b),[256 256]);
else
    img1 = imresize(img,[256 256]);
end
[featureVector, hogVisualization] = extractHOGFeatures(img1);
%% PCA -- for Dimension Reduction
G = pca(double(img1));
whos G
g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)

Variance = mean2(var(double(G))); 
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement

 %%   Test Features  
 
TT(i,:) = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, RMS, Variance, Smoothness, Kurtosis, Skewness, cdr, RDR];





clear  b r g bb ne binaryImage fill dil se di cup se1 BW CC numPixels biggest idx filteredForeground 
clear centers diameters radii stats   rim  c1 c2 dis a dil ans o i nn1 nn ans  cdr RDR

clear img1 G g RMS Standard_Deviation Energy Homogeneity Contrast Correlation stats Variance 
clear a Smoothness Kurtosis Mean hogVisualization featureVector Mean
     

   waitbar(i/15,wt);

end
cd .. 
save TT


close(wt);
