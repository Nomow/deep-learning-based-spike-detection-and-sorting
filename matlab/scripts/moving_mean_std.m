recording_synthesized = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/datasets_1.npy");
spikes = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/ground_truth_data_multiunit_1.npy");
spikes = spikes + 1;
load('/home/vtpc/Documents/Alvils/spike-sorting/data/BOU_JO_Localizer6Hz_sessions','femicro','micros2');
recording =  micros2(16,:, 1); fs = 30000;
%recording = recording_synthesized; fs = 24000;


Fb=100;
[b,a]=butter(1,[2*Fb/fs], 'high'); % femicro: sampling frequency (24kHz)
%bdata = data * - 1;
% filtrage du signal:
LFPh=filtfilt(b,a,recording')';


d = 1;
s = size(LFPh, 2);

mov_mean = movmean(LFPh,1000);
mov_std = movstd(LFPh,1000);
mov_med = movmad(LFPh, 72);
normalized = (LFPh - mov_mean) ./ mov_std;
normalized_med = (LFPh - mov_med) ./ 0.6745;
norm1 = (LFPh - mean(LFPh)) ./ std(LFPh);
figure 
n = normalized ./ max(abs(normalized(d:s))) ;
plot(normalized(d:s))
hold
plot( index(find(index < s & index > d)) -d  - 1, normalized(1, index(find(index < s & index > d))), '*')
t = n(1, index(find(index < s & index > d)));

figure 
nm = normalized_med ./ max(abs(normalized_med(d:s)));

plot(normalized_med(d:s))
hold
plot( index(find(index < s & index > d)) -d  - 1, normalized_med(1, index(find(index < s & index > d))), '*')
t1 = nm(1, index(find(index < s & index > d)));


figure 
hold
plot(n(d:s))
plot(nm(d:s))



index_multi(find(index_multi < s & index_multi > d))








start = spike_first_sample{1};
index = zeros(length(start), 1);
for i = 1:length(index)
    [~, argmax] = max(abs(data(1, start(i):start(i) + 72)));
    index(i) = argmax - 1 + start(i);
end


start_multi = spike_first_sample{1}(find((spike_classes{1} == 0)));
index_multi = zeros(length(start_multi), 1);
for i = 1:length(index_multi)
    [~, argmax] = max(abs(data(1, start_multi(i):start_multi(i) + 72)));
    index_multi(i) = argmax - 1 + start_multi(i);
end