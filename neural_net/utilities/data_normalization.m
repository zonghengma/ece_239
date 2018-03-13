%% 
% Normalizes data by each channel and each trial separately. Originally
% built off of the nan_removed_by_trial.zip dataset.

% Outputs to output/ directory.

clc;
clear all;

%% load the raw data

items = ['..\..\project_datasets\A01T_slice.mat';'..\..\project_datasets\A02T_slice.mat';'..\..\project_datasets\A03T_slice.mat';'..\..\project_datasets\A04T_slice.mat';'..\..\project_datasets\A05T_slice.mat';'..\..\project_datasets\A06T_slice.mat';'..\..\project_datasets\A07T_slice.mat';'..\..\project_datasets\A08T_slice.mat';'..\..\project_datasets\A09T_slice.mat'];

for i = 1 : 9
    data(i) = load(items(i,:));
end

%% nan remove

for a = 1 : 9
    data_removed(a) = data(a);
    for k = 288 : -1 : 1
        if any(any(isnan(data_removed(a).image(:,:,k))))
            % delete entire trial
            data_removed(a).image(:,:,k) = [];
            data_removed(a).type(k) = [];
%             fprintf('Removing subject %i, trial %i\n', a, k);
        end
    end 
end

%% test nans

% isequal(data(1).image(:,:,1:56), data_removed(1).image(:,:,1:56))
% isequal(data(1).type(1:56), data_removed(1).type(1:56))
% isequal(data(1).image(:,:,58:288), data_removed(1).image(:,:,57:287))
% isequal(data(1).type(58:288), data_removed(1).type(57:287))
% 
% isequal(data(6).image(:,:,1:97), data_removed(6).image(:,:,1:97))
% isequal(data(6).type(1:97), data_removed(6).type(1:97))
% isequal(data(6).image(:,:,99:115), data_removed(6).image(:,:,98:114))
% isequal(data(6).type(99:115), data_removed(6).type(98:114))
% isequal(data(6).image(:,:,117:140), data_removed(6).image(:,:,115:138))
% isequal(data(6).type(117:140), data_removed(6).type(115:138))

%% normalize by each channel by each trial

winsor_thresh = [5, 95];
for a = 1 : 9
    subject = data_removed(a);
    num_channels = size(subject.image, 2);
    num_trials = size(subject.image, 3);

    for j = 1 : num_channels
        for k = 1 : num_trials
            subject.image(:,j,k) = winsor(subject.image(:,j,k), winsor_thresh);
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