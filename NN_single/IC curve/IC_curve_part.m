clc,clear,close all
load('B0005.mat');
load('B0006.mat');
load("B0007.mat");
load("B0018.mat");
lower = 3.6;
higher = 4.0;
train1 = trainGenaration(IC_SoC_seg(B0005,'discharge',lower,higher))
train2 = trainGenaration(IC_SoC_seg(B0006,'discharge',lower,higher))
train3 = trainGenaration(IC_SoC_seg(B0007,'discharge',lower,higher))
train4 = trainGenaration(IC_SoC_seg(B0018,'discharge',lower,higher))
save NASAtrain_ICA_part train1 train2 train3 train4
function [train] = trainGenaration(cellin)
    for i = 1:length(cellin)
        cell = cellin(i);
        SoH = [cell.SoH];
        %train.data{i} = [cell.voltage', dQdV];
        train.data{i} = [cell.Voltage_sequence; cell.SoC_sequence; cell.ICA_Gaussian];
        %train.data{i} = [cell.Voltage_sequence; cell.ICA_Gaussian];
        %train.data{i} = [cell.Voltage_sequence; cell.SoC_sequence; cell.ICA_Gaussian;cell.Charge_Time_sequence;cell.Current_sequence];
        train.label{i} = cell.SoH;
    end
        train.data = train.data';
        train.label = cell2mat(train.label');
end
