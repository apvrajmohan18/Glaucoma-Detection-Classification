clc
clear all
close all
disp('If CDR (Cup to Disc Ratio value is 0.1 - 0.3 : NORMAL GLAUCOMA')
disp('If CDR (Cup to Disc Ratio value is 0.3 - 0.5 : NORMAL GLAUCOMA-MILD Level')
disp('If CDR (Cup to Disc Ratio value is 0.5 - 0.7 : MODERATE GLAUCOMA')
disp('If CDR (Cup to Disc Ratio value is  > 0.7    : SEVERE GLAUCOMA')
%% DataBase_Feature Extraction
% % % Database_feature_Extraction;
load TT;
Database_feature=TT;
Dfeature = Database_feature;
%% CHange Directory
cd Database
%% Input Image Browse from file/Folder
[fname,pname]=uigetfile('*.jpg','Select the Input Image');
img = imread(strcat(pname,fname));

cd ..

imwrite(img,'input_image.jpg')
I=img;
b=img;
r=b(:,:,1);

g=b(:,:,2);

bb=b(:,:,3);


%%             
ne=g>130;
binaryImage=ne;
% Get rid of stuff touching the border
binaryImage = imclearborder(binaryImage);

fill=imfill(binaryImage,'holes');
 
se=strel('disk',6)
dil=imdilate(fill,se)
 
figure,imshow(dil)
title('disk image ')
 
% % % disckel
% th=graythresh(g);

ne=g>140;

binaryImage=ne;
% Get rid of stuff touching the border
binaryImage = imclearborder(binaryImage);

cup=imfill(binaryImage,'holes');
 
se1=strel('disk',2);
di=imdilate(cup,se1);
 
cup=di;
 
 figure,;
imshow(cup)
 title('cup image ')
BW=binaryImage ;
CC = bwconncomp(BW);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
BW(CC.PixelIdxList{idx}) = 0;

filteredForeground=BW;


%  
% figure, imshow(BW);
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

% Plot the circles.
 figure,imshow(b)
hold on
viscircles(centers,radii);
hold off


figure
subplot(3,3,1)
imshow(b)
title('input image ')

subplot(3,3,2)
imshow(dil,[])
title('disk segment image ')

subplot(3,3,3)
imshow(b)
hold on
viscircles(centers,radii);
hold off
title('Disc boundary')

subplot(3,3,4)
imshow(di,[])
title('cup image ')

subplot(3,3,5)
imshow(b)
hold on
viscircles(centers,radii/2);
hold off
title('cup boundary')



c1=bwarea(dil);
c2=bwarea(di);

cdr=c2./(c1);


rim=(1-di)-(1-dil);

RDR=bwarea(rim)./(c2);


nn=sprintf('The CDR is  %2f ',cdr);


nn1=sprintf('The RDR is  %2f ',RDR/2);
%% HOG Feature Extraction

if ndims(img)==3
    img1 = imresize(rgb2gray(img),[256 256]);
else
    img1 = imresize(img,[256 256]);
end
[featureVector, hogVisualization] = extractHOGFeatures(img1);
figure; imshow(img1); hold on;
plot(hogVisualization);
%% PCA -- for Dimension Reduction
G = pca(double(img1));
whos G
g = graycomatrix(featureVector);
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
 
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, RMS, Variance, Smoothness, Kurtosis, Skewness, cdr, RDR];

QF = feat;   
%% Training FEatures
load label

%%
meas=Dfeature;% training feature
xdata = meas;
group = label;% category
species= label;
 
g1 = ['NORMAL GLAUCOMA'];
g2 = ['SEVERE GLAUCOMA'];
g3 = ['MODERATE GLAUCOMA'];

X=meas;Y=label;
Mdl = fitcdiscr(meas(:,12:13),species);
Mdl.Mu;

N = size(meas,1);
cvp = cvpartition(N,'Holdout',0.15);
idxTrn = training(cvp); % Training set indices
idxTest = test(cvp); 
tblTrn = array2table(meas(idxTrn,12:13));
tblTrn.Y = species(idxTrn);



labels = predict(Mdl,meas(idxTest,12:13));

% % confusionchart(species(idxTest),labels);
[C,err,P,logp,coeff] = classify(feat(1,12:13),Dfeature(:,12:13),...
                                group,'Quadratic');
   
% Caa = string(C{1}) ;                           
Caa =C;                            
% % disp(C)
% % % helpdlg(C)  
az1 = strcmp(Caa,g1) ;
az2 = strcmp(Caa,g2);
az3 = strcmp(Caa,g3);

 if  az1 ==1 | az2==1 | az3==1 
        if cdr<0.3
            disp('NORMAL GLAUCOMA');msgbox('NORMAL GLAUCOMA');
            cdr
        elseif cdr>0.3  & cdr<=0.5 
         
            disp('NORMAL GLAUCOMA-MILD Level');msgbox('NORMAL GLAUCOMA-MILD Level');
                cdr
                
         elseif cdr>0.5 &  cdr<0.7
           disp('MODERATE GLUCOMA');msgbox('MODERATE GLAUCOMA');
           cdr
       
        
         elseif cdr>0.7
         disp('SEVERE GLUCOMA');msgbox('SEVERE GLAUCOMA');
         cdr
        end
 else
      return;
   end



% Correct_Rate = ( 1 - err ) *100

 R = confusionmat(Mdl.Y,resubPredict(Mdl))

 
