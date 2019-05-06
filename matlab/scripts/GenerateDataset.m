%% data loading
path_to_data = "/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/data";
path_to_ground_truth = "/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/ground_truth";
path_to_save = "/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets";
recordings = LoadData(path_to_data);
ground_truth = LoadData(path_to_ground_truth, "ground_truth.mat");
nb_of_recordings_per_dataset = 5;

%% dividers recordings and seves them to npy file
% concats recordings together and divides them by datasets
nb_of_recordings = size(recordings, 1);
concat_recordings = [];
datasets = {};
for i = 1:nb_of_recordings
    data = recordings{i}.data;
    concat_recordings = [concat_recordings, recordings{i}.data];
    if (mod(i, nb_of_recordings_per_dataset) == 0)
        datasets{end + 1} = concat_recordings;
        concat_recordings = [];
    end
end

% saves datasets
nb_of_datasets = size(datasets, 2)
for i = 1:nb_of_datasets
    file_name = "data_" + i + ".npy";
    save_path = fullfile(path_to_save, file_name);
    writeNPY(datasets{i}, save_path)
end

%% generates ground truth for each dataset
dataset_ground_truth = {}

concat_ground_truth.su_waveforms = {};
concat_ground_truth.spike_classes = {};
concat_ground_truth.spike_first_sample = {};        
for i =1:nb_of_recordings
    concat_ground_truth.su_waveforms(end + 1) = ground_truth{1}.su_waveforms(i)
    concat_ground_truth.spike_classes(end + 1) = ground_truth{1}.spike_classes(i)
    concat_ground_truth.spike_first_sample(end + 1) = ground_truth{1}.spike_first_sample(i)
    if (mod(i, nb_of_recordings_per_dataset) == 0)
        dataset_ground_truth{end + 1} = concat_ground_truth;
        concat_ground_truth.su_waveforms = {};
        concat_ground_truth.spike_classes = {};
        concat_ground_truth.spike_first_sample = {};        
    end
end

% saves ground truth
for i = 1:nb_of_datasets
    file_name = "groun_truth" + i + ".mat";
    save_path = fullfile(path_to_save, file_name);
    data = dataset_ground_truth{i}
    save(save_path, 'data');
end


%% generates ground truth data for each recording
waveform_length = 73;
spike_train_data = {};
remove_multinuit = 0;
for i = 1:nb_of_datasets
    max_amplitude = [];
    neuron = [];
    data = datasets{i};
    ground_truth_data = dataset_ground_truth{i};
    ground_truth_size = size(ground_truth_data.su_waveforms, 2);
    data_per_recording = size(data, 2) / ground_truth_size;
    for j = 1:ground_truth_size
        % removes 0th index
        gd_spike_classes = ground_truth_data.spike_classes{j}';
        gd_spike_first_sample = ground_truth_data.spike_first_sample{j}';
        if(remove_multinuit)
            to_remove = find(gd_spike_classes == 0);
            gd_spike_classes(to_remove) = [];
            gd_spike_first_sample(to_remove) = [];
        end
        gd_spike_first_sample = (j - 1) * data_per_recording + gd_spike_first_sample;
        
        % removes indices that are higher than recording
        indices_higher_than_recording = gd_spike_first_sample + waveform_length  < size(data, 2);
        to_remove = find(indices_higher_than_recording == 0);
        if(~isempty(to_remove))
            gd_spike_classes(to_remove) = [];
            gd_spike_first_sample(to_remove) = [];
        else
            gd_spike_classes = gd_spike_classes + 1;
        end
        
        for k = 1:size(gd_spike_first_sample, 1)
            temp_waveform = data(gd_spike_first_sample(k):gd_spike_first_sample(k) + waveform_length);
            [val, ind] = max(abs(temp_waveform)); 
            gd_spike_first_sample(k) = gd_spike_first_sample(k) + ind - 1;
        end
        max_amplitude = [max_amplitude, (gd_spike_first_sample - 1)'];  
        if(~isempty(neuron))
            max_neuron = max(neuron);
            neuron = [neuron, max_neuron + (gd_spike_classes)'];
        else
            neuron = [neuron, gd_spike_classes'];
        end
    end
    spike_train = [max_amplitude; neuron];
    if(remove_multinuit)
        file_name = "ground_truth_data_" + i + ".npy";
    else
        file_name = "ground_truth_data_multiunit_" + i + ".npy";
    end
    save_path = fullfile(path_to_save, file_name);
    writeNPY(spike_train, save_path)
    spike_train_data{end + 1} = spike_train;
end


dataset = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/datasets_11.npy");
gd = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/ground_truth_data_multiunit_11.npy");

for i=1:size(gd, 2)
    plot(dataset(gd(1, 44885) +1 -32:gd(1, 44885) +1 + 32));
    pause
end
