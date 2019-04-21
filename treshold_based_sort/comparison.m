load('BOU_JO_Localizer6Hz_sessions','femicro','micros2');
sess=1; % choose one of the two sessions

dataset = [];
for i =1 :size(micros2, 1)
   dataset = [dataset, micros2(i, :, sess)];
end
dataset_ind = gpuArray(0:size(dataset,2)-1);
dataset_val = gpuArray(dataset);
interp_val = gpuArray(0:1.25:size(dataset,2)-1);
interp_dataset = interp1(dataset_ind, dataset_val, interp_val, 'spline');
cpu_interp = gather(interp_dataset);

result1 = detect1(micros2(:, :, sess), 24000);
result2 = detect2(dataset, 24000);
spikes_in_both_datasets = find(ismember(result1, result2) == 1);
spikes = result1(spikes_in_both_datasets)
spikes1 = result1(find(ismember(result1, result2) == 0));
spikes2 = result2(find(ismember(result2, result1) == 0));
spikes3 = [spikes1, spikes2];
rnd = randi([1 size(spikes3, 2)],1, floor(size(spikes3, 2) * 0.25));
spikes = [spikes, spikes3(rnd)];
spikes = spikes-1;
gd = ones(size(spikes));
concat_spikes = [spikes; gd];
writeNPY(concat_spikes, "ground_truth_rec1.npy");
writeNPY(cpu_interp, "datasets_rec1.npy");