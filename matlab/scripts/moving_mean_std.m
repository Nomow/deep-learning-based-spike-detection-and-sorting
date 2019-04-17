recording_synthesized = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/datasets_1.npy");
load('/home/vtpc/Documents/Alvils/spike-sorting/data/BOU_JO_Localizer6Hz_sessions','femicro','micros2');
recording_eeg = reshape(micros2(1:32,:, 1), [], 1)';



mean_sub_synth = (recording_synthesized - mean(recording_synthesized)) / std(recording_synthesized);
mean_sub_eeg = (recording_eeg - mean(recording_eeg)) / std(recording_eeg);

Fb=300;
Fh=6000;
[b,a]=butter(1,[2*Fb/30000], 'high'); % femicro: sampling frequency (24kHz)

% filtrage du signal:
LFPh=filtfilt(b,a,recording_eeg')';


window_size = 1000;
from = 1;
to = window_size;
nb_of_steps = ceil(size(recording_synthesized, 2) / window_size) 
recording_size = size(recording_synthesized, 2);
current_mean = zeros(nb_of_steps, 1);
current_std = zeros(nb_of_steps, 1);
temp_recording = recording_synthesized;
current_median = zeros(nb_of_steps, 1);
for i = 1:nb_of_steps
    weight = 0.001 ;
    %avg i = 0
    if (i == 1)
        current_mean(1) = mean(temp_recording(from:to));
        current_std(i) = std(temp_recording(from:to));
        temp_recording(from:to) = (temp_recording(from:to) - current_mean(i)) / current_std(i);

   
    elseif (to < recording_size)
        temp = mean(temp_recording(from:to)) * weight;
        temp_std = std(temp_recording(from:to)) * weight; 
        current_mean(i) = current_mean(i - 1) * (1 - weight) + temp;
        current_std(i) = current_std(i - 1) * (1 - weight) + temp_std;
        temp_recording(from:to) = (temp_recording(from:to) - current_mean(i)) / current_std(i);

    else
        temp = mean(temp_recording(from:end)) * weight;
        temp_std = std(temp_recording(from:end)) * weight; 
        current_mean(i) = current_mean(i - 1) * (1 - weight) + temp;
        current_std(i) = current_std(i - 1) * (1 - weight) + temp_std;
        temp_recording(from:end) = (temp_recording(from:end) - current_mean(i)) / current_std(i);

    end
    from = to + 1;
    to = to + window_size;
end

mean_sub_rec = (temp_recording - mean(temp_recording));
std(mean_sub_rec)