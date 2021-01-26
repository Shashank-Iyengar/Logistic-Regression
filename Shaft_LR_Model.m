%% Homework 3 - Shaft Health Assessment
%% Group 1 - Shashank Iyengar, Johann Koshy, Ashwin Kumat, Ketan Shah

close all
clear all
clc
set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 15)
set(0,'defaultlinelinewidth',.5)
set(0,'DefaultLineMarkerSize', 5)
set(0,'defaultAxesFontWeight','bold') 

%% Data Acquisition
% Training - Healthy Data
testfiledir = 'C:\Users\johan\Desktop\UC_Spring2019\Big_data\HW3\Training\Healthy';
matfiles = dir(fullfile(testfiledir, '*.txt'));
nfiles = length(matfiles);
data  = cell(nfiles);
for i=1:nfiles
    data{i} = dlmread(fullfile(testfiledir, matfiles(i).name), ' ', 5, 0);
end

% Splitting the data array to parts
train_healthy=[];
for i=1:nfiles
    train_healthy(i,:)=cell2mat(data(i,1));
end

% Training - Faulty Data
testfiledir = 'C:\Users\johan\Desktop\UC_Spring2019\Big_data\HW3\Training\Faulty';
matfiles = dir(fullfile(testfiledir, '*.txt'));
nfiles = length(matfiles);
data  = cell(nfiles);
for i=1:nfiles
    data{i} = dlmread(fullfile(testfiledir, matfiles(i).name), ' ', 5, 0);
end

% Splitting the data array to parts
train_faulty=[];
for i=1:nfiles
    train_faulty(i,:)=cell2mat(data(i,1));
end

% Testing Data (30 Sets)
testfiledir = 'C:\Users\johan\Desktop\UC_Spring2019\Big_data\HW3\Testing\Testing';
matfiles = dir(fullfile(testfiledir, '*.txt'));
nfiles = length(matfiles);
data  = cell(nfiles);
for i = 1 : nfiles
    data{i} = dlmread(fullfile(testfiledir, matfiles(i).name), ' ', 5, 0);
end

% Splitting the data array to parts
test_data=[];
for i=1:nfiles
    test_data(i,:)=cell2mat(data(i,1));
end


%% Feature extraction

Fs = 2560;          % Sampling frequency
dt = 1/Fs;          % Time step
Ntime = 38400;      % Number of data points
Ttotal = 15;        % Total time
df = 1/Ttotal;      % Fundamental frequency

for i=1:20
    train_healthy_fft(i,:)=fft(train_healthy(i,:));
    train_faulty_fft(i,:)=fft(train_faulty(i,:));
    
    figure(i)
    plot((0:Ntime/2-1)/Ttotal,(2/Ntime)*abs(train_healthy_fft(i,1:Ntime/2)))
    xlabel(['$ Frequency \;\mathrm{[Hz]} $'],'interpreter','latex')
    ylabel(['$ Magnitude \;\mathrm{[V]} $'],'interpreter','latex')
    txt=['FFT Healthy', num2str(i)];
    title(txt)
    grid on
    xlim([0 50])
    figure(i+20)
    plot((0:Ntime/2-1)/Ttotal,(2/Ntime)*abs(train_faulty_fft(i,1:Ntime/2)))
    xlabel(['$ Frequency \;\mathrm{[Hz]} $'],'interpreter','latex')
    ylabel(['$ Magnitude \;\mathrm{[V]} $'],'interpreter','latex')
    txt=['FFT Faulty', num2str(i)];
    title(txt)
    grid on
    xlim([0 50])
    
    amplitude_healthy(i)=max(((2/Ntime)*abs(train_healthy_fft(i,1:750/2))));
    amplitude_faulty(i)=max(((2/Ntime)*abs(train_faulty_fft(i,1:750/2))));
end
for i=1:30
    testset(i,:) = fft(test_data(i,:));
    amplitude_testset(i) = max(((2/Ntime)*abs(testset(i,1:750/2))));
end

figure
plot(1:20,amplitude_healthy,'-ko')
hold on
plot(1:20,amplitude_faulty,'-r*')
xlabel(['$ Samples\;\mathrm{} $'],'interpreter','latex')
ylabel(['$ Amplitude\;\mathrm{[V]} $'],'interpreter','latex')
legend('Healthy Amplitudes','Faulty Amplitudes')
txt=['Feature extraction'];
title(txt)
grid on

%% Logistic Regression and Health Value Assessment
X = [sort(amplitude_faulty) sort(amplitude_healthy)]';
Xtest = [sort(amplitude_testset)]';
Xtest1 = [(amplitude_testset)]';
y1 = repmat(0.05,1,20);
y2 = repmat(0.95,1,20);
y = [y1 y2]';

b = glmfit(X,y,'binomial','link','logit')
yfit = glmval(b,Xtest1,'logit');

output_training=1./exp(-(b(1)+X*b(2)));
% figure
% plot(output_training,'-o')

% figure
% plot(X, y,'o',Xtest1,yfit,'-','LineWidth',2)
% grid on
% xlabel(['$ x\;\mathrm{} $'],'interpreter','latex')
% ylabel(['$ \pi(x)\;\mathrm{} $'],'interpreter','latex')
% title('LR Curve')
% % legend('Training Data','Logistic Binomial Fit')

figure
plot(1:30,yfit,'-o')
grid on
xlabel(['$ Sample Number\;\mathrm{} $'],'interpreter','latex')
ylabel(['$ CV\;\mathrm{} $'],'interpreter','latex')
title('Health Assessment Curve')

