%%
recording_nb = 45;
gt_file = "/home/vtpc/Documents/Alvils/spike-sorting/data/synthesized/gt_" + string(recording_nb) +".npy";
data_file = "/home/vtpc/Documents/Alvils/spike-sorting/data/synthesized/data_" + string(recording_nb) +".npy";
gt = readNPY(gt_file);
gt(1, :) = gt(1, :) + 1;
data = readNPY(data_file);
%%
spike_position = gt(1,:);
neuron_index = gt(2,:);
%%
waveform_length = 72;
waveforms = zeros(size(gt, 2), 72);
for i =1:size(gt, 2)
    waveforms(i, :) = data(gt(1, i) - waveform_length / 2:gt(1, i) + waveform_length / 2 - 1); 
end
%%
multiunit_index = (find(neuron_index == 0));
singleunit_index_1 = (find(neuron_index == 1));
singleunit_index_2 = (find(neuron_index == 2));
singleunit_index_3 = (find(neuron_index == 5));

sampling_rate = 24000;
seconds = 10;
spikes_in_60_seconds = spike_position(find(sampling_rate*seconds > spike_position))
amplitude_of_spikes = data(spikes_in_60_seconds);
%% raw signal plot
figure1 = figure
plot(data(1:sampling_rate*seconds));
title('60 Seconds of raw signal')
ylabel('amplitude')
xlabel('samples')
hold on
plot(spikes_in_60_seconds - 1, amplitude_of_spikes, 'V' ,'MarkerFaceColor','r', 'MarkerSize',3);
saveas(figure1,'raw_signal.jpg')  %

%% spikes plot
figure2 = figure
sgtitle('Single unit activity spikes')

ax1 = subplot(1,3,1);
ylabel('amplitude')
xlabel('samples')
for i = 1:size(singleunit_index_1, 2)
    hold on
    plot(waveforms(singleunit_index_1(i), :))
    hold on
end
hold on
plot(mean(waveforms(singleunit_index_1, :), 1), 'linewidth', 4, 'color', 'r')

ax2 = subplot(1,3,2);
ylabel('amplitude')
xlabel('samples')
for i = 1:size(singleunit_index_2, 2)
    hold on
    plot(waveforms(singleunit_index_2(i), :))
    hold on
end
hold on
plot(mean(waveforms(singleunit_index_2, :), 1), 'linewidth', 4, 'color', 'r')


ax3 = subplot(1,3,3);
ylabel('amplitude')
xlabel('samples')
for i = 1:size(singleunit_index_3, 2)
    hold on
    plot(waveforms(singleunit_index_3(i), :))
    hold on
end
hold on
plot(mean(waveforms(singleunit_index_3, :), 1), 'linewidth', 4, 'color', 'r')

linkaxes([ax3,ax2,ax1],'xy'); 
saveas(figure2,'single_unit_spikes.jpg')  %


%% shift
index = spike_position(2);
shifted_index = index - 30:15:index + 30;
figure3 = figure

for i =1:size(shifted_index, 2)
    wf =   data(shifted_index(1, i) - waveform_length / 2:shifted_index(1, i) + waveform_length / 2 - 1); 

    if(i == 3)
       plot(wf, 'linewidth', 3, 'color', 'r')
    else
       plot(wf)
    end
    hold on
end
title('Spike shifting')
ylabel('amplitude')
xlabel('samples')
saveas(figure3,'spike_shift.jpg')  %


%% horizontal flip
figure4 = figure
sgtitle('Spike horizontal flip')

ax1 = subplot(1,2,1);
plot(waveforms(25, :))

ylabel('amplitude')
xlabel('samples')

ax2 = subplot(1,2,2);
plot(fliplr(waveforms(25, :)))

ylabel('amplitude')
xlabel('samples')
saveas(figure4,'horizontal_flip.jpg')  %
%% vertical flip
figure5 = figure
sgtitle('Spike vertical flip')

ax1 = subplot(1,2,1);
plot(waveforms(25, :))

ylabel('amplitude')
xlabel('samples')

ax2 = subplot(1,2,2);
plot(-1 * (waveforms(25, :)))

ylabel('amplitude')
xlabel('samples')
linkaxes([ax2,ax1],'xy'); 

saveas(figure5,'vertical_flip.jpg')  %

%% awgn
data_awgn = awgn(data,10,'measured');
figure5 = figure
sgtitle('Added additive white gaussian noise')

ax1 = subplot(1,2,1);
wf =   data(index- waveform_length / 2:index + waveform_length / 2 - 1); 

plot( data(index- waveform_length / 2:index + waveform_length / 2 - 1))

ylabel('amplitude')
xlabel('samples')

ax2 = subplot(1,2,2);
plot( data_awgn(index- waveform_length / 2:index + waveform_length / 2 - 1))

ylabel('amplitude')
xlabel('samples')
linkaxes([ax2,ax1],'xy'); 

saveas(figure5,'awgn.jpg')  %

