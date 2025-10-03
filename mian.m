clc;
close all;
clear;

%% A demo how to run

addpath(genpath('Requirement'))

% DataPath = {
%     'Datasets\ADNI\DATA_MRI.csv';
%     'Datasets\ADNI\DATA_PET.csv'
% };
% Group = {'AD','CN'};
% [X,Y] = getADData(DataPath,Group,15,false);

%% 生成数据 运行示例
c = 2;      % 类别数
n = 100;    % 样本数
d = 50;     % 每个特征维度
M = 3;      % 
%% 生成 Y (one-hot 编码)
Y = zeros(n,c);
labels = randi([1 c],n,1);   % 随机生成标签
for i = 1:n
    Y(i, labels(i)) = 1;
end
%% 生成 X (M个cell, 每个 n×d)
X = cell(1,M);
for m = 1:M
    X{m} = rand(n,d);  % 随机生成特征矩阵
end

%% run
method = 'My';
params = My().init_param([0.01,0.5,0.5,0.5,0.5,100]);

ins = feval(method);
param = params(1,:);
ins = ins.run(X,Y,param, 42);