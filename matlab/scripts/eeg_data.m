load('/home/vtpc/Documents/Alvils/spike-sorting/data/BOU_JO_Localizer6Hz_sessions','femicro','micros2');
path_to_save = "/home/vtpc/Documents/Alvils/spike-sorting/data/eeg_sess_1";

sess = 1;
fs = femicro;
new_fs = 24000 / fs;
recordings = cell(size(micros2, 1), 1);
tresholded_results = cell(size(micros2, 1), 1);
results_based_on_treshold = cell(size(micros2, 1), 1);

for i = 1:2
    recording = micros2(i, :, sess);
    recording_gpu = gpuArray(recording);
    index = gpuArray(0:size(recording, 2) - 1);
    new_index = gpuArray(0:new_fs:size(recording, 2) - 1);
    recordings{i} = gather(interp1(index,recording_gpu,new_index));
    result1 = detect1(recordings{i}, 24000);
    result2 = detect2(recordings{i}, 24000);
    spikes = [result1, result2];
    spikes = unique(spikes)
    Fb=100;
    [b,a]=butter(1,[2*Fb/24000], 'high'); % femicro: sampling frequency (24kHz)
    LFPh=filtfilt(b,a,recordings{i}')';
    opt_zscore = 0.6745 * (LFPh - median(LFPh)) ./ mad(LFPh, 1);
    results_based_on_treshold{i} = spikes;
    tresholded_results{i} = spikes(find(opt_zscore(spikes) <= -4.5 | opt_zscore(spikes) >= 4.5));
end

to_del = cell(size(recordings));
for i = 1:length(recordings)
    rec = recordings{i};
    spikes = tresholded_results{i};
    LFPh=filtfilt(b,a,rec')';
    opt_zscore = 0.6745 * (LFPh - median(LFPh)) ./ mad(LFPh, 1);
    d = opt_zscore(spikes);
    for j = 1:length(spikes)
        plot(opt_zscore(spikes(j) - 36:spikes(j) + 36));
        w = waitforbuttonpress;
        key = get(gcf,'CurrentKey');
        if(key == 'q')
            to_del{i}(end+1) = j;
        end
    end
end





    %find(abs(opt_zscore(spikes_to_use)) <= 4.5)
    spike_train1 = [spikes_to_use1 - 1; ones(size(spikes_to_use1))];
    save_path_gd1 = fullfile(path_to_save, "gdn_" + string(i) + ".npy");
    writeNPY(spike_train1, save_path_gd1)
    spike_train = [spikes_to_use - 1; ones(size(spikes_to_use))];
    save_path_gd = fullfile(path_to_save, "gd_" + string(i) + ".npy");
    writeNPY(spike_train, save_path_gd)
    save_path_data = fullfile(path_to_save, "data_" + string(i) + ".npy");
    writeNPY(new_recording, save_path_data)




