

fs = 24000;
Fb=300;
Fh=6000;

[b,a] = butter(1,[2*Fb/fs 2*Fh/fs], 'high'); 
LFPh=filtfilt(b,a,data')';

step = 1000;
moving_mean = movmean(LFPh, step);
moving_std = movstd(LFPh, step);
moving_normal_dist = (LFPh - moving_mean) ./ moving_std;

moving_mad = movmad(LFPh, step);
moving_median = movmedian(LFPh, step);
moving_opt_zscore = 0.6745 * (LFPh - moving_median) ./ moving_mad;

opt_zscore = 0.6745 * (LFPh - median(LFPh)) ./ mad(LFPh, 1);
zscore = (LFPh - mean(LFPh)) ./ std(LFPh);
%%
figure 
data_pts = zscore(gd);
plot(zscore)
hold
plot(gd, data_pts, ".");
%%
figure 
data_pts1 = opt_zscore(gd);
plot(opt_zscore)
hold
plot(gd, data_pts1, ".");
