import os
import json
from tqdm import tqdm

import IPython
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

args = json.load(open("config.json"))
mytransform=transforms.Compose([transforms.Resize((args['DataSet']['n_mels'], args['length']))])

class Get_Mfcc:
    def __init__(self, audio_dir, args):
        self.args = args['DataSet']
        self.sample_wave, self.sr = librosa.load(audio_dir)
        self.input_nfft = int(round(self.sr * self.args['frame_length']))
        self.input_stride = int(round(self.sr * self.args['frame_stride']))
        self.init_mfcc = librosa.feature.melspectrogram(self.sample_wave,
                                                        n_mels=self.args['n_mels'],
                                                        n_fft=self.input_nfft,
                                                        hop_length=self.input_stride)
        self.init_wave = self.sample_wave.copy()
        
    def resize(self, mfcc):
        size = (args['DataSet']['n_mels'], args['length'])
        if mfcc.shape[1] < size[1]:
            pad = np.zeros((size[0], size[1]-mfcc.shape[1]))
            return np.hstack((mfcc, pad))
        else:
            return mfcc[:, :size[1]]


    def noising(self):
        noise = np.random.randn(len(self.sample_wave))
        augmented_data = self.sample_wave + np.random.uniform(0, self.args['noise_factor']) * noise
        # Cast back to same data type
        self.sample_wave = augmented_data
        return augmented_data

    def shifting(self, shift_direction='both'):
        shift = np.random.randint(self.sr * self.args['shift_max'] + 1)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(self.sample_wave, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        self.sample_wave = augmented_data
        return augmented_data

    def time_mask(self):
        start_point = np.random.randint(0, len(self.sample_wave) - int(self.args['mask_ratio'] * len(self.sample_wave)))
        end_point = start_point + int(self.args['mask_ratio'] * len(self.sample_wave))

        augmented_data = np.hstack((self.sample_wave[:start_point],
                                    np.zeros(end_point - start_point),
                                    self.sample_wave[end_point:]))
        self.sample_wave = augmented_data
        return augmented_data

    def change_pitch(self):
        augmented_data = librosa.effects.pitch_shift(self.sample_wave, self.sr, np.random.randint(-self.args['pitch_factor'], self.args['pitch_factor']))
        self.sample_wave = augmented_data
        return augmented_data

    def get_audio(self):
        return IPython.display.Audio(data=self.sample_wave, rate=self.sr)

    def plot_mfcc(self):
        self.sample_mfcc = self.get_mfcc()
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(self.sample_mfcc, ref=np.max),
                                 y_axis='mel', sr=self.sr, hop_length=self.input_stride, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        plt.show()

    def get_mfcc(self):
        self.mfcc = librosa.feature.melspectrogram(self.sample_wave,
                                                   n_mels=self.args['n_mels'],
                                                   n_fft=self.input_nfft,
                                                   hop_length=self.input_stride)
        self.mfcc = self.resize(self.mfcc)
        return self.mfcc

    def get_random_aug_mfcc(self):
        self.noising()
        self.shifting()
        self.time_mask()
        self.change_pitch()
        self.mfcc = librosa.feature.melspectrogram(self.sample_wave,
                                                   n_mels=self.args['n_mels'],
                                                   n_fft=self.input_nfft,
                                                   hop_length=self.input_stride)
        self.mfcc = self.resize(self.mfcc)
        return self.mfcc


class AugDset(Dataset):
    def __init__(self, args):
        root = 'mfcc_data/'
        self.all_waves = [root + genre + '/' + wav for genre in os.listdir(root) for wav in os.listdir(root + genre)]
        self.all_genres = os.listdir(root)
        self.label_num = len(self.all_genres)
        self.args = args
        self.transform = mytransform

    def __len__(self):
        return len(self.all_waves)

    def __getitem__(self, idx):
        wave = self.all_waves[idx]
        genre = wave.split('/')[1]
        genre_index = self.all_genres.index(genre)
        x = torch.load(wave)
        y = torch.tensor(genre_index)

        return x, y


class DSet(Dataset):
    def __init__(self, args):
        root = 'data/genres/'
        self.all_waves = [root + genre + '/' + wav for genre in os.listdir(root) for wav in os.listdir(root + genre)]
        self.all_genres = os.listdir(root)
        self.label_num = len(self.all_genres)
        self.args = args
        self.transform = mytransform

    def __len__(self):
        return len(self.all_waves)

    def __getitem__(self, idx):
        wave = self.all_waves[idx]
        genre = wave.split('/')[1]
        genre_index = self.all_genres.index(genre)
        x = torch.from_numpy(Get_Mfcc(wave, self.args).get_mfcc())
        x = self.transform(x.view(1,x.size(0),x.size(1)))
        y = torch.tensor(genre_index)

        return x, y

def save_aug_tensor():
    try:os.mkdir('mfcc_data')
    except FileExistsError:return print('All ready data')
    for genre in os.listdir('data/genres'):
        os.mkdir('mfcc_data/'+genre)
    root = 'data/genres/'
    all_waves = [root + genre + '/' + wav for genre in os.listdir(root) for wav in os.listdir(root + genre)]
    all_genres = os.listdir(root)
    for idx in tqdm(range(len(all_waves))):
        wave = all_waves[idx]
        genre = wave.split('/')[2]
        x = torch.from_numpy(Get_Mfcc(wave, args).get_random_aug_mfcc())
        x = x.view(1, x.size(0), x.size(1))
        torch.save(x, f"mfcc_data/{genre}/{wave.split('.')[1]}.pt")
    return print('All ready data')

