
addpath 'C:\ProgramData\MATLAB\SupportPackages\R2021a\fmid-v40'
clear all
close all
%% Get data
%Choose original or raw data
%file = 1 for original data, anything else for raw data;
file = 0;
if file == 1 
    filename = 'preprocessed_original_data.csv';
else
    filename = 'preprocessed_raw_data.csv';
end

data = readtable(filename);
%% Preprocessing
if file == 1
    x = data(:,1:end-1);
    y = data(:,end);
else
    x = data(:,1:end-6);
    y = data(:,end-5:end);
end

% Feature Normalization
% Min-max for inputs, 0-1 for being malign

for i = 1:width(x)
    x_norm = x.(i);
    x_min = min(x_norm);
    x_max = max(x_norm);
    x_norm = ((x_norm - x_min)/(x_max - x_min));
    x.(i) = x_norm;
end

if file == 1
    y_norm = y.(1);
    y_min = min(y_norm);
    y_max = max(y_norm);
    y_norm = ((y_norm - y_min)/(y_max - y_min));
    y.(1) = y_norm;
end

x = table2array(x);
y = table2array(y);

% Test/train split
rng(82, 'twister');
test_ptg = 0.3;
test_idxs = randsample(1:size(x,1),round(size(x,1)*test_ptg));
train_idxs = setdiff(1:size(x,1), test_idxs);

x_train = x(train_idxs,:);
x_test = x(test_idxs,:);
y_train = y(train_idxs,:);
y_test = y(test_idxs,:);

train_data.U = x_train;
train_data.Y = y_train;
%% Parameters
par = struct();
par.c = 6;                     % Clusters
% par.m = 1.5;                     % Fuzziness
% par.tol = 0.01;                % Termination
par.ante = 2;                  % Antecedent:  1 - product-space MFS
                               %              2 - projected MFS
par.cons = 2;                  % Consequent:  1 - global LS
                               %              2 - weighted LS
% par.Ny = 1;                    % Lagged Outputs
% par.Nu = [1 1 1 1 1 1 1 1 1];  % Lagged inputs
% par.Nd = [1 1 1 1 1 1 1 1 1];  % Transport delays 

%% One Simulation
% Clustering e Simulação
% [FM,Part] = fmclust(train_data,par);
FM = fmclust(train_data,par);

ym = fmsim(x_test, y_test, FM);
ym2 = round(ym) == 1 + 0;
results = (ym2 == y_test);
accuracy = length(find(results))/size(y_test,1);
fprintf('              Estimated\n')
fprintf('            |Neg.|Pos.|\n')
fprintf('Actual Neg. | %u | %u |\n', sum(ym2 == 0 & y_test == 0), sum(ym2 == 1 & y_test == 0))
fprintf('       pos. | %u | %u |\n', sum(ym2 == 0 & y_test == 1), sum(ym2 == 1 & y_test == 1))
fprintf('Accuracy is %.2f%s\n', accuracy*100, '%')

%% Using genfis
options = genfisOptions('FCMClustering');
options.NumClusters = 2;
fis = genfis(x_train,y_train, options);
y_pred = evalfis(x_test, fis);
y_pred2 = round(y_pred) == 1 + 0;
results_genfis = (y_pred2  == y_test);
accuracy_genfis = length(find(results_genfis))/size(y_test,1);
fprintf('              Estimated\n')
fprintf('            |Neg.|Pos.|\n')
fprintf('Actual Neg. | %u | %u |\n', sum(y_pred2  == 0 & y_test == 0), sum(y_pred2  == 1 & y_test == 0))
fprintf('       pos. | %u | %u |\n', sum(y_pred2  == 0 & y_test == 1), sum(y_pred2  == 1 & y_test == 1))
fprintf('Accuracy is %.2f%s\n', accuracy_genfis*100, '%')