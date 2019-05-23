tp = [];
fp = [];
total = [];
for i = 46:94
    gt_file = "/home/vtpc/Documents/Alvils/spike-sorting/data/synthesized/gt_" + string(i) +".npy";
    data_file = "/home/vtpc/Documents/Alvils/spike-sorting/data/synthesized/data_" + string(i) +".npy";
    data = readNPY(data_file);
    gt = readNPY(gt_file);
    spikes = gt(1,:) +1;
    [result1] = detect3(data,24000);
    tp = [tp; size(find(ismember(result1, spikes) == 1), 2)];
    fp = [fp; size(find(ismember(result1, spikes) == 0), 2)];
    total = [total; size(spikes,2)];
    i
end