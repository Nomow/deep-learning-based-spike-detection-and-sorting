path_to_data = "/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/data";
path_to_ground_truth = "/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/ground_truth";

ground_truth = load("/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/ground_truth/ground_truth.mat");
% creates recording
data = []
data_t = [];
for i = 1:5
    temp = load("/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/data/simulation_" + string(i) + ".mat");
    temp = temp.data;
    data = [data; temp];
    data_t = [data_t, temp];
end


%%
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
