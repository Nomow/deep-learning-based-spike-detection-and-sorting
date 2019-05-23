path_to_data = "/home/vtpc/Documents/Alvils/spike-sorting/data/eeg_sess_1/data_1.npy";
path_to_gt= "/home/vtpc/Documents/Alvils/spike-sorting/data/eeg_sess_1/gdn_1.npy";

data = readNPY(path_to_data);
ground_truth = readNPY(path_to_gt);
ground_truth_matlab = ground_truth + 1;



Fb=100;
fs = 24000;
[b,a]=butter(1,[2*Fb/fs], 'high'); % femicro: sampling frequency (24kHz)
%bdata = data * - 1;
% filtrage du signal:
LFPh=filtfilt(b,a,data')';

opt_zscore = 0.6745 * (LFPh - median(LFPh)) ./ mad(LFPh, 1);

spikes = opt_zscore(ground_truth_matlab(1,:));
d = find(abs(spikes) < 4.5);
spikes(d)