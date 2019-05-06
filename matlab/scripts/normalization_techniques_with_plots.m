load('/home/vtpc/Documents/Alvils/spike-sorting/data/BOU_JO_Localizer6Hz_sessions','femicro','micros2');
sess = 1;
fs = 30000;
step = 1000;
nb_of_channels = size(micros2, 1);

for i = 1:1
    recording_synthesized = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/datasets_1.npy");
    spikes = readNPY("/home/vtpc/Documents/Alvils/spike-sorting/data/recording_datasets/ground_truth_data_multiunit_1.npy");
    spikes = spikes+1;
    Fb=100;
    Fh = 6000
    [b,a]=butter(1,[2*Fb/24000], 'high');
   %[b,a]=butter(1,[2*Fb/fs 2*Fh/fs]); % femicro: sampling frequency (24kHz)

    LFPh=filtfilt(b,a,recording_synthesized(1:1400000)')';
    spikes = spikes(find(spikes < 1400000));
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
    ax1 = subplot(4,1,1);
    plot(moving_z_score)
    hold
    plot(spikes, moving_z_score_pts, '*');
    
    ax2 = subplot(4,1,2);
    plot(moving_opt_z_score)
    hold
    plot(spikes, moving_opt_z_score_pts, '*'); 
    
    ax3 = subplot(4,1,3);
    plot(opt_zscore)
    hold
    plot(spikes, opt_zscore_pts, '*'); 
    
    ax4 = subplot(4,1,4);
    plot(zscore)
    hold
    plot(spikes, zscore_pts, '*'); 
    
    linkaxes([ax1 ,ax2, ax3, ax4],'xy');

end

