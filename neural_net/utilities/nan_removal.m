%% Only run the first four cells, including this one. 
% Inputs: Requires the raw AOnT_slice.mat files to be in project_datasets/
% directory.

% Outputs: the same mat files, but with NaN values removed. Goes through
% each subject and each trial, and if ANY channel in that trial has a
% NaN value ANYWHERE in it, it removes that entire trial from the 
% dataset for that subject. So if subject 1 has 1 NaN in trial 5, channel
% 2, then the entire trial 5 is deleted.
clc;
clear all;

%% load the raw data

items = ['..\..\project_datasets\A01T_slice.mat';'..\..\project_datasets\A02T_slice.mat';'..\..\project_datasets\A03T_slice.mat';'..\..\project_datasets\A04T_slice.mat';'..\..\project_datasets\A05T_slice.mat';'..\..\project_datasets\A06T_slice.mat';'..\..\project_datasets\A07T_slice.mat';'..\..\project_datasets\A08T_slice.mat';'..\..\project_datasets\A09T_slice.mat'];

for i = 1 : 9
    data(i) = load(items(i,:));
end

%% data remove

items = ['..\..\project_datasets\A01T_slice.mat';'..\..\project_datasets\A02T_slice.mat';'..\..\project_datasets\A03T_slice.mat';'..\..\project_datasets\A04T_slice.mat';'..\..\project_datasets\A05T_slice.mat';'..\..\project_datasets\A06T_slice.mat';'..\..\project_datasets\A07T_slice.mat';'..\..\project_datasets\A08T_slice.mat';'..\..\project_datasets\A09T_slice.mat'];

for a = 1 : 9
    data_removed(a) = load(items(a,:));
    for k = 288 : -1 : 1
        if any(any(isnan(data_removed(a).image(:,:,k))))
            % delete entire trial
            data_removed(a).image(:,:,k) = [];
            data_removed(a).type(1) = [];
        end
    end 
end

%% save

for a = 1: 9
    filename = strcat('output\A0', sprintf('%i', a), 'T_slice.mat');
    image = data_removed(a).image;
    type = data_removed(a).type;
    save(filename, 'image', 'type', '-v7.3');
end

%%

% %% trial nan count
% 
% count = 0;
% nan_count = 0;
% 
% for a = 1 : 9
%     for k = 1 : 288
%         if any(any(isnan(data(a).image(:,:,k))))
%             fprintf('subject: %i, trial: %i\n', a, k);
%             nan_count = nan_count + 1;
%         end
%         count = count + 1;
%     end
% end
% 
% pct_nan = nan_count / count * 100;
% 
% %% nan check
% min_nan_count = 10000;
% max_nan_count = -1;
% sum_nan_count = 0;
% sum_count = 0;
% 
% for a = 1 : 9
%     for k = 1 : 288
%         for j = 1 : 22
%             sum_count = sum_count + 1;
%             if any(isnan(data(a).image(:,j,k)))
%                 fprintf('subject: %i, trial: %i, channel: %i\n', a, k, j);
%                 nan_count = sum(sum(isnan(data(a).image(:,j,k))));
%                 sum_nan_count = sum_nan_count + 1;
%                 if nan_count > max_nan_count
%                     max_nan_count = nan_count;
%                     subject_max_nan = a;
%                     trial_max_nan = k;
%                     channel_max_nan = j;
%                 end
%                 if nan_count < min_nan_count
%                     min_nan_count = nan_count;
%                 end
%                 fprintf('num nan: %i\n', nan_count);
%             end
%         end
%     end
% end
% avg_nan_count = sum_nan_count / sum_count;
%                 
% 
% %% 
% num_lags = 800;
% res = zeros(num_lags + 1, 22);
% 
% for a = 1 : 9
%     for k = 1 : 288
% %         if sum(sum(isnan(data(a).image(:,:,k)))) == 0
%         if any(isnan(data(a).image(:,:,k)))
%             fprintf('subject: %i, trial: %i, channel: %i\n', a, k, j);
%         else
%             for j = 1 : 22
%                 res(:, j) = res(:, j) + autocorr(data(a).image(:, j, k), num_lags);
%             end
%         end
%     end
% end
% 
% times = linspace(0, 4, size(res,1));
% 
% plot(times, res)
% 
% %%
% 
% res = zeros(21, 22, 9);
% 
% for a = 1 : 9
%     for k = 1 : 288
%         if sum(sum(isnan(data(a).image(:,:,k)))) == 0
%                 for j = 1 : 22
%                     res(:, j, a) = res(:, j, a) + autocorr(data(a).image(:, j, k));
%                 end
%         end
%     end
% end
% 
% times = linspace(0, 4, size(res,1));
% 
% for a = 1 : 9
%     figure();
%     plot(times, res(:, :, a));
% end
% 
% 
% %%
% 
% Fs = 250;
% L = 1000;
% T = 1 / Fs;
% 
% Y = fft(data(1).image(:,1,1));
% 
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% f = Fs*(0:(L/2))/L;
% figure()
% plot(f, P1);
% ylim([0,9])
% figure();
% plot(data(1).image(:,1,1));
% 
% %%
% d = fdesign.lowpass('Fp,Fst', .05, .2);
% filt = design(d, 'butter');
% y = filter(filt, data(1).image(:,1,1));
% 
% 
% Fs = 250;
% L = 1000;
% T = 1 / Fs;
% 
% Y = fft(y);
% 
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% f = Fs*(0:(L/2))/L;
% figure();
% plot(f, P1);
% figure();
% plot(y);
