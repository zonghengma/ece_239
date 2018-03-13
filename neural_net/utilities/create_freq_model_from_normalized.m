
clc;
clear all;

%% load the raw data

items = ['..\..\project_datasets\nan_winsor_normalized\A01T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A02T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A03T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A04T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A05T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A06T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A07T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A08T_slice.mat';'..\..\project_datasets\nan_winsor_normalized\A09T_slice.mat'];

for i = 1 : 9
    data(i) = load(items(i,:));
end

%%

Fs = 250;
T = 1 / Fs;


num_splits = 8;
num_samples_per_split = 1000 / num_splits;
img_rows = 6;
img_cols = 7;
freq_channels = 3;

L = 1000 / num_splits;

theta_low = 4;
theta_high = 7;
alpha_low = 7;
alpha_high = 13;
beta_low = 13;
beta_high = 30;

clear data_out;
data_out(9) = struct();

for a = 1 : 9
    num_channels = size(data(a).image, 2);
    num_samples = size(data(a).image, 3);
    
    current_fft = zeros(num_samples, num_splits, num_channels, freq_channels);
    
    for j = 1 : num_channels
        for k = 1 : num_samples
            for n = 1 : num_splits
                n_low = (n - 1) * num_samples_per_split + 1;
                n_high = n * num_samples_per_split;
                Y = fft(data(a).image(n_low:n_high, j, k));
                P2 = abs(Y/L);
                P1 = P2(1:floor(L/2)+1);
                P1(2:end-1) = 2 * P1(2:end-1);
                f = Fs*(0:floor(L/2))/L;

                % find closest match in freq
                [low_d, low_theta_idx] = min( abs( f-theta_low ) );
                [high_d, high_theta_idx] = min( abs( f-theta_high ) );
                [low_d, low_alpha_idx] = min( abs( f-alpha_low ) );
                [high_d, high_alpha_idx] = min( abs( f-alpha_high ) );
                [low_d, low_beta_idx] = min( abs( f-beta_low ) );
                [high_d, high_beta_idx] = min( abs( f-beta_high ) );
                
                sum_theta = sum(P1(low_theta_idx:high_theta_idx).^2);
                sum_alpha = sum(P1(low_alpha_idx:high_alpha_idx).^2);
                sum_beta  = sum(P1(low_beta_idx :high_beta_idx ).^2);

                current_fft(k, n, j, 1) = sum_theta;
                current_fft(k, n, j, 2) = sum_alpha;
                current_fft(k, n, j, 3) = sum_beta;
            end
        end
    end
    
    data_out(a).image = current_fft;
end

%% double check

% % squeeze(data_reshaped(1).image(1, 1, :, :, 1))
% 
% Fs = 250;
% L = 125;
% T = 1 / Fs;
% 
% Y = fft(data(1).image(1:125,1,1));
% 
% P2 = abs(Y/L);
% P1 = P2(1:floor(L/2)+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% f = Fs*(0:floor(L/2))/L;
% plot(f, P1)



%% reshaper

img = [0,0,0,1,0,0,0 ; 0,2,3,4,5,6,0 ; 7,8,9,10,11,12,13 ; 0,14,15,16,17,18,0 ; 0,0,19,20,21,0,0 ; 0,0,0,22,0,0,0];

for a = 1 : 9
    data_reshaped(a) = data_out(a);
    img_size = size(data_out(a).image);
    data_reshaped(a).image = zeros(img_size(1), img_size(2), size(img, 1), size(img, 2), freq_channels);

    for ii = 1 : size(img, 1)
        for jj = 1 : size(img, 2)
            if img(ii, jj) ~= 0
                data_reshaped(a).image(:, :, ii, jj, :) = data_out(a).image(:, :, img(ii, jj), :);
            end
        end
    end
end

%%

for a = 1: 9
    filename = strcat('output\A0', sprintf('%i', a), 'T_slice.mat');
    % mismatched because labels never change
    image = permute(data_reshaped(a).image, [5, 4, 3, 2, 1]);
    type = data(a).type;
    save(filename, 'image', 'type', '-v7.3');
end

%%


%                 if low_theta_idx == high_theta_idx || low_alpha_idx == high_alpha_idx || low_beta_idx == high_beta_idx
%                     low_theta_idx, high_theta_idx, low_alpha_idx, high_alpha_idx, low_beta_idx, high_beta_idx
%                 end



% 
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% f = Fs*(0:(L/2))/L;
% figure()
% plot(f, P1);
% ylim([0, 2]);
% xlim([7,13]);
% figure();
% plot(data(1).image(:,1,1));