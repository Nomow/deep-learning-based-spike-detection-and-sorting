%% data loading
path_to_data = "/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/data";
path_to_ground_truth = "/home/vtpc/Documents/Alvils/spike-sorting/simulation_data/ground_truth";
path_to_save = "/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets";
recordings = LoadData(path_to_data);
ground_truth = LoadData(path_to_ground_truth);
nb_of_recordings_per_dataset = 5;

%% dividers recordings and seves them to npy file
% concats recordings together and divides them by datasets
nb_of_recordings = size(recordings, 2);
concat_recordings = [];
concat_ground_truth = [];
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
    file_name = "datasets_" + i + ".npy";
    save_path = fullfile(path_to_save, file_name);
    writeNPY(datasets{i}, save_path)
end

%% 