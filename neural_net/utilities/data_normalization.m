%% 
% Normalizes data by each channel and each trial separately. Originally
% built off of the nan_removed_by_trial.zip dataset.

% Outputs to output/ directory.

clc;
clear all;

%% load the raw data

items = ['..\..\project_datasets\nan_removed_by_trial\A01T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A02T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A03T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A04T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A05T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A06T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A07T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A08T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A09T_slice.mat'];

for i = 1 : 9
    data(i) = load(items(i,:));
end

%% normalize by each channel by each trial

items = ['..\..\project_datasets\nan_removed_by_trial\A01T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A02T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A03T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A04T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A05T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A06T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A07T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A08T_slice.mat';'..\..\project_datasets\nan_removed_by_trial\A09T_slice.mat'];

for a = 1 : 9
    subject = load(items(a,:));
    num_channels = size(subject.image, 2);
    num_trials = size(subject.image, 3);

    for j = 1 : num_channels
        for k = 1 : num_trials
            trial = subject.image(:,j,k);
            trial = trial - mean(trial);
            trial = trial / std(trial);
            subject.image(:,j,k) = trial;
        end
    end
    
    data_normalize_by_trial_and_channel(a) = subject;
end

%% save

for a = 1: 9
    filename = strcat('output\A0', sprintf('%i', a), 'T_slice.mat');
    image = data_normalize_by_trial_and_channel(a).image;
    type = data_normalize_by_trial_and_channel(a).type;
    save(filename, 'image', 'type', '-v7.3');
end