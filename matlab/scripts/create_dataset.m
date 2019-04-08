
ground_truth = load("~/Downloads/ground_truth.mat");
% creates recording
data = []
data_t = [];
for i = 1:5
    temp = load("~/Downloads/simulation_" + string(i) + ".mat");
    temp = temp.data;
    data = [data; temp];
    data_t = [data_t, temp];
end

writeNPY(data_t, "single_channel_data.npy")

data_test = []
data_t1 = [];
for i = 6:8
    temp = load("~/Downloads/simulation_" + string(i) + ".mat");
    temp = temp.data;
    data_test = [data_test; temp];
    data_t1 = [data_t1, temp];
end

writeNPY(data_t1, "single_channel_test_data.npy")
clear data_t1

% creates ground truth data [spike position in recording (index), which
% spike IT IS]
waveform_length = 73;
spike_index = [];
spike_position = [];
for i = 1:5
    max_spike = 0;
    curr_pos = 0;
    if(~isempty(spike_index))
        max_spike = max(spike_index);
        curr_pos = size(data, 2) * (i - 1);
    end
    gd_spike_classes = ground_truth.spike_classes{i}';
    gd_spike_first_sample = ground_truth.spike_first_sample{i}';
    to_remove = find(gd_spike_classes == 0);
    gd_spike_classes(to_remove) = [];
    gd_spike_first_sample(to_remove) = [];
    
    for j = 1:size(gd_spike_first_sample, 1)
        temp_waveform = data(i, gd_spike_first_sample(j):gd_spike_first_sample(j) + waveform_length);
        [val, ind] = max(temp_waveform); 
        gd_spike_first_sample(j) = gd_spike_first_sample(j) + ind - 1;
    end
    spike_position = [spike_position; curr_pos + gd_spike_first_sample - 1];        
    spike_index = [spike_index; max_spike + (gd_spike_classes -1)];

    
end


spike_train = [spike_position, spike_index];
writeNPY(spike_train', "single_channel_spike_train.npy")


clear data
for i = 1:10
    figure
    plot(data_t(spike_position(i) -32: spike_position(i) + 32))
end


% cTTTT
% spike IT IS]
waveform_length = 73;
spike_index = [];
spike_position = [];
for i = 1:3
    max_spike = 0;
    curr_pos = 0;
    if(~isempty(spike_index))
        max_spike = max(spike_index);
        curr_pos = size(data_test, 2) * (i - 1);
    end
    gd_spike_classes = ground_truth.spike_classes{5+i}';
    gd_spike_first_sample = ground_truth.spike_first_sample{5 + i}';
    to_remove = find(gd_spike_classes == 0);
    gd_spike_classes(to_remove) = [];
    gd_spike_first_sample(to_remove) = [];
    
    for j = 1:size(gd_spike_first_sample, 1)
        temp_waveform = data_test(i, gd_spike_first_sample(j):gd_spike_first_sample(j) + waveform_length);
        [val, ind] = max(temp_waveform); 
        gd_spike_first_sample(j) = gd_spike_first_sample(j) + ind - 1;
    end
    spike_position = [spike_position; curr_pos + gd_spike_first_sample - 1];        
    spike_index = [spike_index; max_spike + (gd_spike_classes -1)];

    
end


spike_train = [spike_position, spike_index];
writeNPY(spike_train', "single_channel_spike_train_test.npy")






