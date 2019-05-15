
from pathlib import Path
from os.path import expanduser
from os import path
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from scipy import signal

def conv1x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv1x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=63, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(0.5),
                                      torch.nn.Linear(256 * block.expansion, num_classes))
        

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


class Recording(Dataset):
    """Recording data class
    path_to_data - path to recordings
    device - device to use '('cpu or gpu')'
    transform - composed tranformations
    """
    
    def __init__(self, path_to_data, device="cpu", transform=None):
        data = np.load(path_to_data);
        assert data.size > 0
        self.data = torch.tensor(data).float().to(device);
        self.transform = transform
        if self.transform:
          self.data = self.transform(self.data)
    def __len__(self):
        return self.data.size()[1]

    def __getitem__(self, idx):
        sample = self.data[idx];
        return sample     
      
class GroundTruth(Dataset):
    """GroundTruth data 
    path_to_data - path to ground truth dataset 2 x n array '['0, : ']' = position ,  '['1, : ']' = neuron index
    device - device to use '('cpu or gpu')'
    transform - composed tranformations
    """
    
    def __init__(self, path_to_data, device="cpu", transform=None):
        data = np.load(path_to_data);
        assert data.size > 0
        self.data = torch.tensor(data).int().to(device);
        self.transform = transform
        if self.transform:
          self.data = self.transform(self.data)
    def __len__(self):
        return self.data.size()[1]

    def __getitem__(self, idx):
        sample = self.data[:, idx];
        return sample     
      
      
      
class NormalizeDataset(object):
    """Normalizes recording from ... to range
    range_from - recording range to start from
    range_to - recording range to end to
    """
    def __init__(self, range_from, range_to):
      assert (range_from < range_to)
      self.range_from = range_from;
      self.range_to = range_to;
      
    def __call__(self, recording):
      recording_min = recording.min();
      recording_max = recording.max();
      unit_range = recording_max - recording_min;
      unit_recording = (recording - recording_min) / unit_range;
      recording_range = self.range_to - self.range_from;
      normalized_recording = (unit_recording * recording_range) + self.range_from;
      return normalized_recording;

    
    
class FlipData(object):
    """Flips tensor along given axis in dims
    axis_to_flio - tensor of axis to flip
    """
    def __init__(self, axis_to_flip):
      self.axis_to_flip = axis_to_flip;
      
    def __call__(self, data):
      return torch.flip(data, self.axis_to_flip)


class ShiftData(object):
    """Shifts position of neuron positions by shift_from to shift_to
    shift_indices - shift data indices 
    recording_length - recording size of 1 channel
    waveform_length - waveform length
    """
    def __init__(self, shift_indices, recording_length, waveform_length):
      assert (waveform_length % 2 == 0)
      self.shift_indices = shift_indices;
      self.recording_length = recording_length;
      self.waveform_length = waveform_length;
      
    def __call__(self, data):
      waveform_div = self.waveform_length // 2;
      spike_max_amplitude_index_size = data[0, :].size()[0];
      neuron =  data[1, :];
      print(neuron.max())
      shift_size = self.shift_indices.size()[0];
      new_size = spike_max_amplitude_index_size * shift_size;
      shifted_data = torch.zeros([2, new_size], dtype=torch.int)
      iter_from = 0;
      iter_to = 0;
      # copies data and shifts by j positions
      for i in range(0, shift_size):
        temp = torch.FloatTensor();
        temp = temp.new_tensor(data);
        temp[0, :] = temp[0, :] + self.shift_indices[i];
        iter_to = iter_from + temp[0, :].size()[0];
        shifted_data[:, iter_from:iter_to] = temp;

        iter_from = iter_to;
        #print(temp[1, :].max())  
      # out of bound check
      non_out_of_bound_index = np.where((shifted_data[0, :] - waveform_div >= 0) & (shifted_data[0, :] + waveform_div < self.recording_length));
      shifted_data = shifted_data[:, non_out_of_bound_index[0]];
      return shifted_data;

class ExtractWaveforms(object):
    """Extracts waveforms from recordings
    spike_data - spike positions and neuron indexes
    waveform_length - waveform length
    """
    def __init__(self, spike_data, waveform_length):
      assert (waveform_length % 2 == 0)
      self.spike_data = spike_data;
      self.waveform_length = waveform_length;
    """ recording """  
    def __call__(self, data):
      waveform_div = self.waveform_length // 2;
      spike_data_size = self.spike_data[0, :].size()[0];
      waveforms = torch.zeros([spike_data_size, 1, self.waveform_length], dtype=torch.float)
      for i in range(0, spike_data_size):
        index_from = self.spike_data[0, i] - waveform_div;
        index_to = self.spike_data[0, i] + waveform_div;
        waveforms[i, :] = data[0, index_from:index_to]
      return waveforms;
    
class Awgn(object):
    """Add white Gaussian noise to signal
    data - recording
    snr - signal to noise ratio in db
    """
    def __init__(self, snr):
      self.snr = snr;
    """ recording """  
    def __call__(self, data):
      data_length = data[0, :].size()[0];
      signal_to_noise_ratio = 10**(self.snr / 10);
      energy = (abs(data[0, :]) ** 2).sum() / (data_length); #Calculate  energy
      noise_spectral_density = energy / signal_to_noise_ratio; #ind the noise spectral density
      std = (noise_spectral_density).sqrt(); #Standard deviation for AWGN Noise
      noise = std * torch.randn(data_length); #computes noise
      noised_data = data + noise
      return noised_data;
      
class SpikeTrainDataset(Dataset):
    """Spike train dataset
    data - waveforms
    labels - neuron class
    """
    def __init__(self, data = None, labels=None):
        if (torch.is_tensor(data)):
          self.data = data;
        else:
          self.data = torch.FloatTensor();

        if (torch.is_tensor(labels)):
          self.labels = labels;
        else:
          self.labels = torch.torch.IntTensor();

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spike = self.data[idx, :];
        label = self.labels[idx];
        return spike, label

"""Augments data from original recording
    path_to_recording - recording path
    path_to_ground_truth - ground_truth_path 1st row - position of spike 2nd row neuron class
    normalization_range - data range to normalize from ... to
    waveform_length - waveform length to be extracted
    snr_ratio_db - signal to noise ratio in decibels
    shift_indexes - int tensor of indices to shift original position of spike
    flip_data_horizontal - flip data horizontally
    flip_data_vertical - flip_data vertically by changing value sign
    transform_list - transforms for recording
"""
def AugmentData(path_to_recording, path_to_ground_truth, waveform_length,
                snr_ratio_db=None, shift_indexes = torch.IntTensor(),
                flip_data_horizontal = False,
                flip_data_vertical = False,
                transform_list = []):
  
  transform_list_recording = [];
  # adds noise
  if (snr_ratio_db != None):
    transform_list_recording.append(Awgn(snr_ratio_db));

  transform_list_recording.extend(transform_list);
  transform = transforms.Compose(transform_list_recording);
  recording = Recording(path_to_recording, transform = transform);
  print(transform_list_recording)
  # shifts data
  if (shift_indexes.nelement() != 0):
    transform = transforms.Compose([ShiftData(shift_indexes, recording.__len__(), waveform_length)]);
    ground_truth = GroundTruth(path_to_ground_truth, transform = transform);
  else:    
    ground_truth = GroundTruth(path_to_ground_truth);

  #recording = MovingMeanAndTsdNormalizationOnSpikesOnly(recording, ground_truth, waveform_length, 1000)    
      
  transform = transforms.Compose([ExtractWaveforms(ground_truth.data, waveform_length)]);
  waveforms = transform(recording.data);
  if (flip_data_horizontal == True):
    transform = transforms.Compose([FlipData([2])]);
    waveforms = transform(waveforms);

  if (flip_data_vertical == True):
    waveforms = -1 * wavefroms;
  dataset = SpikeTrainDataset(waveforms, ground_truth.data[1, :])
  return dataset;

"""Generates dataset for train/test from recording and ground truth
    path_to_recording - recording path
    path_to_ground_truth - ground_truth_path 1st row - position of spike 2nd row neuron class
    normalization_range - data range to normalize from ... to
    waveform_length - waveform length to be extracted
    max_dataset_size - dataset size of recording, can be little bit bigger than max
    snr_from - snr ratio to augment from (rnd init)
    snr_to - snr ratio to augment 
    max_shift - +- indices to shift spike
    use_horizontal_flip - use horizontal flip for augmentation
    use_vertical_flip - use vertical flip for augmentation
    transform_list - transforms list for recording
"""

def GenerateDataset(path_to_recording, path_to_ground_truth, waveform_length, max_dataset_size,
                    snr_from, snr_to, max_shift, use_horizontal_flip, use_vertical_flip, transform_list = []):
  i = 0;
  #snr_from = 20;
  #snr_to = 100;
  dataset = SpikeTrainDataset();

  while(dataset.__len__() < max_dataset_size):
    print("=" * 10, i + 1, "generation", "=" * 10)
    temp_dataset = SpikeTrainDataset();
    print("temp_dataset_len: ", temp_dataset.__len__());


    # first iteration awgn is not used
    snr_ratio_db = np.random.randint(snr_from, snr_to + 1) + np.random.uniform();
 
    # calculates shift from __ to
    shift_from = - 1 * max_shift;
    shift_to = max_shift + 1;
    shift_step = np.random.randint(1, 2);
    shift_indexes = torch.tensor(np.arange(shift_from, shift_to, shift_step)).int();



    print("shift_from: ", shift_from)
    print("shift_to: ", shift_to)
    print("shift_step: ", shift_step)
    print("shift_indexes: ", shift_indexes)
    print("snr_ratio: ", snr_ratio_db)
    flip_data_horizontal = 0;
    if (use_horizontal_flip):
      flip_data_horizontal = np.random.randint(0, 2);

    flip_data_vertical = 0;
    if (use_vertical_flip):
      flip_data_vertical = np.random.randint(0, 2);

    print("flip_data_horz: ", flip_data_horizontal)
    # generates flipped data
    temp = AugmentData(path_to_recording, path_to_ground_truth, waveform_length, snr_ratio_db = snr_ratio_db, shift_indexes = shift_indexes, flip_data_horizontal = flip_data_horizontal, flip_data_vertical = flip_data_vertical, transform_list = transform_list);

    dataset = torch.utils.data.ConcatDataset((dataset, temp));
    print("dataset len: ", dataset.__len__());
    i = i + 1;
  return dataset;

"""trains model
    model - pytorch model
    device - device type cpu or gpu
    train_loader - torch.DataLoader object of data and classes
    optimizer  - torch optimizer 
    criterion - torch criterion
    scheduler - lr scheduler
    epoch - current epoch
    logging_interval - logging interval after how many batches
"""
def Train(model, device, train_loader, optimizer, criterion, scheduler, epoch, logging_interval):
    scheduler.step() 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % logging_interval  == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.20f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss;
    
    
"""test model
    model - pytorch model
    device - device type cpu or gpu
    criterion - torch criterion
    test_loader - torch.DataLoader object of data and classes
"""
def Test(model, device, criterion, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for (data, target) in test_loader:
      data, target = data.to(device), target.to(device=device, dtype=torch.int64)
      output = model(data)
      test_loss += criterion(output, target)
      pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.20f}, Accuracy: {}/{} ({:.20f}%)\n'.format(
  test_loss, correct, len(test_loader.dataset),
  100. * correct / len(test_loader.dataset)))
  
  
  
"""Generates train and test dataset
    data - train data
    labels - train labels
    divider - divier of train and test 
"""

def GenerateTrainAndTestDataset(data, labels, divider):
  labels = labels.int();
  permuted_indices = torch.randperm(labels.nelement());
  data_permuted = data[permuted_indices, :, :];
  labels_permuted = labels[permuted_indices];
  # calculates train and test data size for each class
  unique_classes, nb_of_occourences = np.unique(labels_permuted, return_counts=True);
  size_of_each_test_class = np.round(nb_of_occourences * divider);
  test_data_size = int(np.sum(size_of_each_test_class))
  train_data_size = labels_permuted.nelement() - test_data_size;
  #init
  train_data = torch.zeros([train_data_size, data_permuted.size()[1], data_permuted.size()[2]], dtype=torch.float)
  train_labels = torch.zeros(train_data_size, dtype=torch.int)
  test_data = torch.zeros([test_data_size, data_permuted.size()[1], data_permuted.size()[2]], dtype=torch.float)
  test_labels = torch.zeros(test_data_size, dtype=torch.int)
  
  train_counter = 0;
  test_counter = 0;
  class_size_counter = torch.zeros(unique_classes.size);

  for i in range(labels_permuted.nelement()):
    spike = data_permuted[i, :]
    label = int(labels_permuted[i])
    # test data
    if(class_size_counter[label] < size_of_each_test_class[label]):
      test_data[test_counter, :] = spike;
      test_labels[test_counter] = label;
      test_counter = test_counter + 1;
      class_size_counter[label] = class_size_counter[label] + 1
    # train data
    else:
      train_data[train_counter, :] = spike;
      train_labels[train_counter] = label;
      train_counter = train_counter + 1;
  train = SpikeTrainDataset(train_data, train_labels);
  test = SpikeTrainDataset(test_data, test_labels);
  return train, test;
  
  
  
"""
  Adds padding at the end of recording 0's
  recording - recording data object
  step_size - step size of inference
  waveform_length - waveform length to be extracted
"""
def AddPaddingToRecording(recording, step_size, waveform_length):
  assert(recording.__len__() > waveform_length)
  recording_size = recording.__len__(); 
  nb_of_steps = (recording_size - waveform_length) // step_size
  #adds +1 nb_of_steps for padding purposes if nb of steps is not int
  mod = (recording_size - waveform_length) % step_size
  if(mod > 0):
    nb_of_steps = nb_of_steps + 1;
    padding_size = (nb_of_steps * step_size + waveform_length) - recording_size
    padding = torch.zeros((1, padding_size), dtype=torch.float);
    recording.data = torch.cat((recording.data, padding), 1)
  return recording;

import subprocess
"""Get the current gpu usage.
   Keys are device ids as integers.
   Values are memory usage as integers in MB.
"""
def get_gpu_memory_map():

    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map
  
"""dataset for inference.
   data - waveforms to infer
   transform - any transformations to dataset
"""
class InferenceDataset(Dataset):
    # spike dataset.
    def __init__(self, data, transform=None):
        self.data = data;
        print(self.data.type())
        self.transform = transform
        torch.cuda.synchronize()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spike = self.data[idx, :];
        sample = spike
        if self.transform:
          sample = self.transform(sample)
        return sample     
      
"""dataset for inference.
   model - nn model
   data_loader - data loader object
   num_classes - number of classes for output
"""
def Inference(model, data_loader, num_classes):
  model.eval()
  gpu_result = torch.cuda.FloatTensor(len(data_loader.dataset), num_classes).fill_(0);
  idx_from = 0;
  idx_to = 0;
  with torch.no_grad():
    for i, data in enumerate(data_loader):
      output = model(data)
      idx_to = idx_from + len(output);
      gpu_result[idx_from:idx_to] = output;
      idx_from = idx_to;
      print("done: ", i, " batch")
    torch.cuda.synchronize() 
    result = gpu_result.cpu();
    return result;


"""
  Extracts waveform indices from recording using step size
  0...waveform_length, step_size ... waveform_length + step 
  recording - recording dataset object
  step_size - step size of inference
  waveoform_length - waveform length to be extracted
"""
def GetWaveformIndices(recording, step_size, waveform_length):
  nb_of_steps = (recording.__len__() - waveform_length) // step_size;
  waveform_indices = torch.zeros((1, nb_of_steps), dtype=torch.int);
  for i in range(0, nb_of_steps):
    waveform_indices[0, i] = i * step_size + waveform_length // 2;
  return waveform_indices
  
  
  
"""
  Treshold prediction based on std of results
  result - result of inference
  treshold - coefficent to mutiply std
"""
def TresholdPredictBasedOnStd(result, treshold):
  mean_result = result.mean();
  std_result = result.std();
  predictions = (result > mean_result + std_result * treshold)
  return predictions;
  
  
  
  
  
"""
  Gets sequence of spike train indices
  if max amplitude is in Nth position then it gets indices from np.arange(N - waveform_length // 2 - shift_size:N + waveform_length // 2 + shift_size)
  spike_train - GroundTruth data object
  waveform_length - waveform length of extractable spike
  shift_size = shift of each side of waveform_length Waveform_length - shift_size to Waveform_length + shift_size

"""
def GetSpiketrainWaveformSequenceIndices(spike_train, waveform_length, recording, shift_size=0):
  spike_positions = spike_train.data[0, :];
  spike_from = spike_positions - waveform_length // 2 - shift_size;
  spike_to = spike_positions + waveform_length // 2 + shift_size;
  spike_waveform_sequence = torch.zeros(spike_positions.nelement() * (waveform_length +  2 * shift_size), dtype = torch.long);
  for i in range(spike_positions.nelement()):
    sequence = torch.arange(spike_from[i], spike_to[i]);
    waveform_from = i * (waveform_length + 2 * shift_size);
    waveform_to = waveform_from + waveform_length + 2 * shift_size;
    spike_waveform_sequence[waveform_from:waveform_to] = sequence;
  return spike_waveform_sequence;




"""
  Gets randomly noise indices
  path_to_recording - path to the original recording
  path_to_ground-truth - path_to_ground_truth_data
  nb_of_elements - number of elements to choose
"""
def GetNoiseIndices(path_to_recording, path_to_ground_truth, waveform_length, nb_of_elements):
  recording = Recording(path_to_recording);
  spike_train = GroundTruth(path_to_ground_truth);
  shift_size = waveform_length // 2 + waveform_length // 4;
  spike_train_sequence = GetSpiketrainWaveformSequenceIndices(spike_train, waveform_length, recording, shift_size=shift_size);
  non_out_of_bound_index = np.where((spike_train_sequence >= 0) & (spike_train_sequence < recording.__len__()));
  spike_train_sequence = spike_train_sequence[non_out_of_bound_index];
  noise_indexes_in_recording = torch.ones(recording.__len__());
  noise_indexes_in_recording[spike_train_sequence] = 0;
  noise_positions = torch.tensor(np.where(noise_indexes_in_recording == 1)[0]).int();
  noise_random_indices =  torch.randint(0, noise_positions.size()[0], (nb_of_elements,));
  noise_indexes = noise_positions[noise_random_indices];
  return torch.unsqueeze(noise_indexes, 0);
  
  
class StandartNormalization(object):
    """StandartNormalization using mean and std
    meam - mean of spikes
    std -  std of spikes
    """
    def __init__(self, mean, std):
      self.mean = mean;
      self.std = std;
    """ recording """  
    def __call__(self, waveforms):
      spike_mean = self.mean.repeat(waveforms.__len__());
      spike_std = self.std.repeat(waveforms.__len__());
      transform = transforms.Compose([transforms.Normalize(spike_mean, spike_std)]);
      normalized_wavefroms = transform(waveforms)
      return normalized_wavefroms;


    
def MovingMeanAndTsdNormalizationOnSpikesOnly(recording, ground_truth, waveform_length, window_size):
  
    """Normalizes data by using moving mean and std
    window_size - window size of nb of elements
    recoridng - recording of dataset
    ground_truth - ground truth data
    waveform_length - wavefroms length of spike
    """
    # extracts unique waveform indices
    waveform_div = waveform_length // 2;
    indices = ground_truth.data[0, :];
    spike_indices = torch.zeros((indices.nelement(), waveform_length), dtype=torch.int);
    for i in range(indices.nelement()):
      curr_max_amplitude_index = indices[i];
      print("curr_index:", curr_max_amplitude_index);
      waveform_from = curr_max_amplitude_index - waveform_div;
      waveform_to = curr_max_amplitude_index + waveform_div;
      spike_indices[i, :] = torch.arange(waveform_from, waveform_to);
    unique_spike_indices = spike_indices.view(-1).unique();

    data = recording.data.clone();

    # moving mean std
    window_div = window_size // 2;
    for i in range(unique_spike_indices.nelement()):
      curr_max_amplitude_index = unique_spike_indices[i];
      ind_from = np.max([curr_max_amplitude_index - window_div, 0]);
      ind_to = np.min([curr_max_amplitude_index + window_div, data.nelement()]);
      mean = recording.data[0, ind_from:ind_to].mean();
      std = recording.data[0, ind_from:ind_to].std();
      data[0, curr_max_amplitude_index]  = (data[0, curr_max_amplitude_index] - mean) / std;
      print("curr_max_amplitude_index: ", curr_max_amplitude_index, "from: ", ind_from, ind_to, "mean std: ", mean, std, "data: ", data[0, curr_max_amplitude_index])

    recording.data = data;
    return recording;
    
class MovingMeanAndStdNormalization(object):
  
    """Normalizes data by using moving mean and std
    window_size - window size of nb of elements
    """
    def __init__(self, window_size):
      self.window_size = window_size;
    """ recording """  
    def __call__(self, data):
      print("started movingmeanandstd")
      mean = torch.zeros((1, data.nelement()));
      std = torch.zeros((1, data.nelement()));
      window_div = self.window_size // 2;
      for i in range(data.nelement()):
        ind_from = np.max([-1 * window_div + i, 0]);
        ind_to = np.min([i + window_div, data.nelement()]);
        mean[0, i] = data[0, ind_from:ind_to].mean();
        std[0, i] = data[0, ind_from:ind_to].std();
      return (data - mean) / std;
      print("done movingmeanandstd")
class FilterSignalUsingButtersWorth(object):
    """filters signal using buttersworth filter type
        filter_type - type of filter - lowpass highpass, bandpass
        sampling_frequency - sampling frequency of signal
        freq_range - passband frequency and/or stopband frequency range '('array')'
        order - order of filter
    """
    def __init__(self, filter_type, sampling_frequency, freq_range, order):
        self.filter_type = filter_type;
        self.sampling_frequency = sampling_frequency;
        self.freq_range = freq_range;
        self.order = order;
    """ recording """  
    def __call__(self, data):
        normalized_freq =  2 * self.freq_range / self.sampling_frequency
        b, a = signal.butter(self.order, normalized_freq, btype=self.filter_type);
        print(b, a)
        filtered_data = signal.filtfilt(b, a,  data.numpy(),  padlen  = 3*(max(len(b),len(a))-1));
        return torch.FloatTensor(filtered_data.copy());

from scipy import stats
class OptimizedZScoreNormalizaton(object):
    """optimized z-score normalization """
    """ recording """  
    def __call__(self, data):
      median = data.median();
      median_abs_dev = torch.median((data - data.median()).abs());
      normalized_data = 0.6745 * (data - median) / median_abs_dev;
      return torch.FloatTensor(normalized_data);

from torch.nn import init

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=128, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

# Define the Resnet18-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.2):
        super(ft_net, self).__init__()
        model_ft = resnet18();
        self.model = model_ft
        self.model.fc = ClassBlock(256, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        #x = self.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.model.fc(x)
        return x


"""
  softmax and treshold based prediction
  result - cnn result
  treshold - treshold to detect if its spike or not
"""
def PredictionByTreshold(result, treshold):
  soft_max = nn.Softmax(1)
  probability = soft_max(result)
  prediction = torch.argmax(probability, 1);
  for i  in range(prediction.nelement()):
    if prediction[i] > 0:
      score = probability[i, prediction[i]];
      if(score < treshold):
        prediction[i] = 0;
  return prediction;

  
"""
  Gets max amplitude of spikes so it doesn't overlap with other spikes
  overlap is dfefined based on waveform_length, no 2 max amplitudes will be in +- waveform_length // 2 
  recording - dataset recording
  predictions - predicted indexes to be spikes
  waveform_lengh - spike waveform length
  waveform_indices - waveform indices of each detection window
"""
def GetNonOverlappingSpikesMaxAmplitude(recording, predictions, waveform_length, waveforms, waveform_indices):
    pred_ind = np.where(predictions == 1)[0];
    waveform_start = waveform_indices[0, pred_ind] - waveform_length // 2;
    waveform_argmax = np.argmax(abs(waveforms[pred_ind, :, :,]), axis=2);
    predicted_index = np.unique(waveform_start + waveform_argmax.view(-1).to(torch.int32));
    predicted_index1 = np.unique(waveform_start + waveform_argmax.view(-1).to(torch.int32));
    difference = np.diff(predicted_index);
    arg_min = np.argmin(difference);
    val_min = np.min(difference)
    while (val_min < waveform_length):
        isuppr = np.argmin(abs(recording.data[0,predicted_index[arg_min:arg_min+2]]));
        predicted_index = np.concatenate( ( predicted_index[:arg_min+isuppr], predicted_index[arg_min+isuppr+1:] ) )
        difference = np.diff(predicted_index);
        arg_min = np.argmin(difference);
        val_min = np.min(difference)
    return predicted_index;