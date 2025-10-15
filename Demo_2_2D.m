clc;clear;close all

load trainData2D.mat
res = [train1; train2; train3; train4];
%%Data downsample
res_downsampled = res(1:1000:end, :);
disp(size(res_downsampled));
res = res_downsampled;

%%  split train set and 
[rowCount, colCount] = size(res);   
RowNum = rowCount;                  
PTratio = 0.5;                        
FeatureNum = 4;                  

P_train = res((1: round(RowNum*PTratio)), 1: FeatureNum)';
T_train = res((1: round(RowNum*PTratio)), FeatureNum+1)';
M = size(P_train, 2);

P_test = res((round(RowNum*PTratio)+1: end), 1: FeatureNum)';
T_test = res((round(RowNum*PTratio)+1: end), FeatureNum+1)';
N = size(P_test, 2);

%% Normalisation
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

P_train =  double(reshape(P_train, FeatureNum, 1, 1, M));
P_test  =  double(reshape(P_test , FeatureNum, 1, 1, N));

t_train = t_train';
t_test  = t_test' ;

for i = 1 : M
    p_train{i, 1} = P_train(:, :, 1, i);
end

for i = 1 : N
    p_test{i, 1}  = P_test( :, :, 1, i);
end

%% CNN-SelfAttention-LSTM
inputLayer = sequenceInputLayer(FeatureNum, 'Name', 'input');

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

lstmLayer = lstmLayer(64, 'OutputMode', 'last', 'Name', 'lstm'); 

fcLayers = [
    fullyConnectedLayer(20, 'Name', 'fc1')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'fc2')
    regressionLayer('Name', 'regression')
    ];

lgraph = layerGraph(inputLayer);
lgraph = addLayers(lgraph, convBlock);
lgraph = addLayers(lgraph, attentionLayer);
lgraph = addLayers(lgraph, lstmLayer);
lgraph = addLayers(lgraph, fcLayers);
concatLayer = concatenationLayer(1, 2, 'Name', 'concat');
lgraph = addLayers(lgraph, concatLayer);
lgraph = connectLayers(lgraph, 'input', 'conv1');
lgraph = connectLayers(lgraph, 'relu1', 'selfattention/in');
lgraph = connectLayers(lgraph, 'relu1', 'concat/in1');
lgraph = connectLayers(lgraph, 'selfattention', 'concat/in2');
lgraph = connectLayers(lgraph, 'concat', 'lstm');
lgraph = connectLayers(lgraph, 'lstm', 'fc1');

%  Net train
options = trainingOptions('adam', ...      
    'MaxEpochs', 20, ...                 
    'InitialLearnRate', 0.05, ...          
    'LearnRateSchedule', 'piecewise', ...  
    'LearnRateDropFactor', 0.1, ...        
    'LearnRateDropPeriod', 1200, ...       
    'Shuffle', 'every-epoch', ...          
    'Plots', 'training-progress', ...      
    'Verbose', false);
net = trainNetwork(p_train, t_train, lgraph, options);
analyzeNetwork(net)

%%  Verification
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );
ResultTrain = mapminmax('reverse', t_sim1, ps_output);
ResultTest = mapminmax('reverse', t_sim2, ps_output);

%%  errors
err_train = T_train - ResultTrain';
err_test = T_test - ResultTest';

% R2
R1 = 1 - norm(T_train - ResultTrain')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - ResultTest')^2 / norm(T_test  - mean(T_test ))^2;

disp(['train set R2：', num2str(R1)])
disp(['test set R2：', num2str(R2)])

% MAE
mae1 = sum(abs(ResultTrain' - T_train)) ./ M ;
mae2 = sum(abs(ResultTest' - T_test )) ./ N ;

disp(['train set MAE：', num2str(mae1)])
disp(['test set MAE：', num2str(mae2)])

% MBE
mbe1 = sum(ResultTrain' - T_train) ./ M ;
mbe2 = sum(ResultTest' - T_test ) ./ N ;

disp(['train set MBE：', num2str(mbe1)])
disp(['test set MBE：', num2str(mbe2)])

%  RMSE
RMSE_train = sqrt(sum((err_train).^2) ./ M);
RMSE_test = sqrt(sum((err_test ).^2) ./ N);

disp(['train set RMSE：', num2str(RMSE_train)])
disp(['test set RMSE：', num2str(RMSE_test)])


%%  FIGs estimation-real
figure(1)
plot(1: M, T_train, 'r-*', 1: M, ResultTrain, 'b-o', 'LineWidth', 1)
legend('Real value', 'Prediction')
xlabel('time/min')
ylabel('SoH')
string = {'Train set'; ['RMSE=' num2str(RMSE_train)]};
title(string)
xlim([1, M])

figure(2)
plot(1: N, T_test, 'r-*', 1: N, ResultTest, 'b-o', 'LineWidth', 1)
legend('Real value', 'Prediction')
xlabel('time/min')
ylabel('SoH')
string = {'Test set'; ['RMSE=' num2str(RMSE_test)]};
title(string)
xlim([1, N])

%% FIGs error
figure(3)
plot(1:M, err_train, 'r-*', 'LineWidth', 1)
title('train set error')
xlabel('time/min')
ylabel('error')
xlim([1 M]) 
grid on

figure(4)
plot(1:N, err_test, 'b-o', 'LineWidth', 1)
title('test set error')
xlabel('time/min')
ylabel('error')
xlim([1 N]) 
grid on

