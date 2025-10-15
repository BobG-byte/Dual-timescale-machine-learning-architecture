
clc;clear;close all
load NASAtrain.mat
%% Data prepare
% data = [train1.data];
% labels = [train1.label];

data = [train1.data; train2.data; train3.data; train4.data];
labels = [train1.label; train2.label; train3.label; train4.label];
for i =1:10
[RMSE(i), MAE(i), net{i}] = cnn_att_lstm(data, labels)
end

function [RMSE, MAE, net] = cnn_att_lstm(data, labels)
numObservations = length(data);
numfeatures = size(data{1},1);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations, [0.5 0.1 0.4]);

XTrain = data(idxTrain);
XValidation = data(idxValidation);
XTest = data(idxTest);

TTrain = labels(idxTrain);
TValidation = labels(idxValidation);
TTest = labels(idxTest);
numHiddenUnits = 128;

inputLayer = sequenceInputLayer(numfeatures, 'Name', 'input',Normalization="zscore");
convBlock = [
    convolution1dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn') 
    reluLayer('Name', 'relu111') 
    convolution1dLayer(5, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn1') 
    reluLayer('Name', 'relu11') 
    convolution1dLayer(7, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn2') 
    reluLayer('Name', 'relu1') 
    ];


attentionLayer = selfAttentionLayer(4,40,"Name","selfattention");

lstm = lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm');  

fcLayers = [
    fullyConnectedLayer(20, 'Name', 'fc1') 
    reluLayer('Name', 'relu2') 
    fullyConnectedLayer(1, 'Name', 'fc2') 
    regressionLayer('Name', 'regression')
    ];

lgraph = layerGraph(inputLayer);

lgraph = addLayers(lgraph, convBlock);
lgraph = addLayers(lgraph, attentionLayer);
lgraph = addLayers(lgraph, lstm);
lgraph = addLayers(lgraph, fcLayers);

concatLayer = concatenationLayer(1, 2, 'Name', 'concat');

lgraph = addLayers(lgraph, concatLayer);

lgraph = connectLayers(lgraph, 'input', 'conv1');
lgraph = connectLayers(lgraph, 'relu1', 'selfattention/in');
lgraph = connectLayers(lgraph, 'relu1', 'concat/in1');
lgraph = connectLayers(lgraph, 'selfattention', 'concat/in2');

lgraph = connectLayers(lgraph, 'concat', 'lstm');

lgraph = connectLayers(lgraph, 'lstm', 'fc1');
% numHiddenUnits = 100;

% layers = [ ...
%     sequenceInputLayer(numChannels, Normalization="zscore")
%     lstmLayer(numHiddenUnits, OutputMode="last")
%     fullyConnectedLayer(numResponses)
%     regressionLayer]

options = trainingOptions("adam", ...
    'MaxEpochs', 200, ...                
    'InitialLearnRate', 0.005, ...          
    'LearnRateSchedule', 'piecewise', ...  
    'LearnRateDropFactor', 0.1, ...        
    'LearnRateDropPeriod', 1200, ...       
    'Shuffle', 'every-epoch', ...          
    ValidationData={XValidation TValidation}, ...
    OutputNetwork="best-validation-loss", ...
    SequenceLength="shortest", ...
    Verbose= false);

%net = trainNetwork(XTrain, TTrain, layers, options);
net = trainNetwork(XTrain, TTrain, lgraph, options);

YTest = predict(net,XTest, SequenceLength="shortest");

RMSE = sqrt(sum((YTest - TTest).^2) ./ length(TTest))
MAE = sum(abs(YTest - TTest)) ./ length(TTest) 
end




function varargout = trainingPartitions(numObservations,splits)
%TRAININGPARTITONS Random indices for splitting training data
%   [idx1,...,idxN] = trainingPartitions(numObservations,splits) returns
%   random vectors of indices to help split a data set with the specified
%   number of observations, where SPLITS is a vector of length N of
%   partition sizes that sum to one.
%
%   Example: Get indices for 50%-50% training-test split of 500
%   observations.
%   [idxTrain,idxTest] = trainingPartitions(500,[0.5 0.5])
%
%   Example: Get indices for 80%-10%-10% training, validation, test split
%   of 500 observations. 
%   [idxTrain,idxValidation,idxTest] = trainingPartitions(500,[0.8 0.1 0.1])

arguments
    numObservations (1,1) {mustBePositive}
    splits {mustBeVector,mustBeInRange(splits,0,1,"exclusive"),mustSumToOne}
end

numPartitions = numel(splits);
varargout = cell(1,numPartitions);

idx = randperm(numObservations);

idxEnd = 0;

for i = 1:numPartitions-1
    idxStart = idxEnd + 1;
    idxEnd = idxStart + floor(splits(i)*numObservations) - 1;

    varargout{i} = idx(idxStart:idxEnd);
end

% Last partition.
varargout{end} = idx(idxEnd+1:end);

end

function mustSumToOne(v)
% Validate that value sums to one.

if sum(v,"all") ~= 1
    error("Value must sum to one.")
end

end
