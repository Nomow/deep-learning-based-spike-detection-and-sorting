load('/home/vtpc/Documents/Alvils/spike-sorting/data/BOU_JO_Localizer6Hz_sessions','femicro','micros2');
recording =  micros2(16,:, 1); fs = 30000;
%recording = recording_synthesized; fs = 24000;
result1 = detect1(recording, fs);
result2 = detect2(recording, fs);
spikes_in_both_datasets = find(ismember(result1, result2) == 1);
spikes = result1(spikes_in_both_datasets)
spikes1 = result1(find(ismember(result1, result2) == 0));
spikes2 = result2(find(ismember(result2, result1) == 0));
spikes3 = [spikes1, spikes2];
rnd = randi([1 size(spikes3, 2)],1, floor(size(spikes3, 2) * 0.25));
spikes = [spikes, spikes3(rnd)];
spikes = spikes;

Fb=100;
[b,a]=butter(1,[2*Fb/fs], 'high'); % femicro: sampling frequency (24kHz)
%bdata = data * - 1;
% filtrage du signal:
LFPh=filtfilt(b,a,recording')';


d = 1;
s = size(LFPh, 2);

mov_mean = movmean(LFPh,1000);
mov_std = movstd(LFPh,1000);
mov_med = movmad(LFPh, 120);
normalized = (LFPh - mov_mean) ./ mov_std;
normalized_med = (LFPh - mov_med) ./ 0.6745;
norm1 = (LFPh - mean(LFPh)) ./ std(LFPh);
figure 
n = normalized ./ max(abs(normalized(d:s))) ;
plot(normalized(d:s))
hold
plot( spikes(find(spikes < s & spikes > d)) -d  - 1, normalized(1, spikes(find(spikes < s & spikes > d))), '*')
t = n(1, spikes(find(spikes < s & spikes > d)));

figure 
nm = normalized_med ./ max(abs(normalized_med(d:s)));

plot(normalized_med(d:s))
hold
plot( spikes(find(spikes < s & spikes > d)) -d  - 1, normalized_med(1, spikes(find(spikes < s & spikes > d))), '*')
t1 = nm(1, spikes(find(spikes < s & spikes > d)));


mov_mean90 = movmean(LFPh,90);
mov_mad90 = movmad(LFPh,30);
temp = mov_mad90 ./ 0.6745;
figure
hold
plot(data1)
plot( spikes(find(spikes < s & spikes > d)) -d  - 1, data1(1, spikes(find(spikes < s & spikes > d))), '*')

mov_mean1000 = movmean(LFPh,1000);
mov_std1000 = movstd(LFPh,1000);


data1 =( LFPh - mov_mean90 ) ./ mov_std1000;
data2 = ( LFPh - mov_mean90 ) ./ std(LFPh - mov_mean90);
