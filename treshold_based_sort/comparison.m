load("datasets_3_result4.5_sc2.mat");
original_result = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/ground_truth_data_multiunit_3.npy");
cnn_result = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/models/resnet_multi/results/datasets_3_results.npy");
original_result = original_result(1,:)';
cnn_result = cnn_result + 1;
original_result = original_result + 1;

cnn_original_tp = find(ismember(original_result,cnn_result) == 1);
cnn_original_fp = find(ismember(cnn_result,original_result) == 0);
static_original_tp = find(ismember(original_result,result') == 1);
static_original_fp = find(ismember(result',original_result) == 0);

acc_static = size(static_original_tp, 1) / size(original_result, 1)
acc_cnn = size(cnn_original_tp, 1) / size(original_result, 1)