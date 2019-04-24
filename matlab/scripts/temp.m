load('/home/vtpc/Documents/Alvils/spike-sorting/data/BOU_JO_Localizer6Hz_sessions','femicro','micros2');
sess = 1;
fs = 30000;
step = 1000;
nb_of_channels = size(micros2, 1);

for i = 1:nb_of_channels
    data = micros2(i, :, sess);
    result1 = detect1(data, fs);
    result2 = detect2(data, fs);
    spikes = unique([result1, result2]);
    Fb=100;
    Fh = 6000
   % [b,a]=butter(1,[2*Fb/fs], 'high');
   [b,a]=butter(1,[2*Fb/fs 2*Fh/fs]); % femicro: sampling frequency (24kHz)

    LFPh=filtfilt(b,a,data')';

    %%
    mov_mean = movmean(LFPh, step);
    mov_std = movstd(LFPh, step);
    moving_z_score = (LFPh - mov_mean) ./ mov_std;
    moving_z_score_pts = moving_z_score(spikes);
    %%
    mov_mad = movmad(LFPh, step);
    mov_med = movmedian(LFPh, step);
    moving_opt_z_score = 0.6745 * (LFPh - mov_med) ./ mov_mad;
    moving_opt_z_score_pts = moving_opt_z_score(spikes);

    %%
    opt_zscore = 0.6745 * (LFPh - median(LFPh)) ./ mad(LFPh, 1);
    opt_zscore_pts = opt_zscore(spikes);
    %%
    zscore = (LFPh - mean(LFPh)) ./ std(LFPh);
    zscore_pts = zscore(spikes);

    figure
    title(string(i) + " channel")
    subplot(4,1,1);
    plot(moving_z_score)
    hold
    plot(spikes, moving_z_score_pts, '*');
    
    subplot(4,1,2);
    plot(moving_opt_z_score)
    hold
    plot(spikes, moving_opt_z_score_pts, '*'); 
    
    subplot(4,1,3);
    plot(opt_zscore)
    hold
    plot(spikes, opt_zscore_pts, '*'); 
    
    subplot(4,1,4);
    plot(zscore)
    hold
    plot(spikes, zscore_pts, '*'); 

end

