import os

import IPython
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

class Get_Mfcc:
    def __init__(self, audio_dir, frame_length=0.25, frame_stride=0.01):
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.sample_wave, self.sr = librosa.load(audio_dir)
        self.input_nfft = int(round(self.sr * self.frame_length))
        self.input_stride = int(round(self.sr * self.frame_stride))
        self.init_mfcc = librosa.feature.melspectrogram(self.sample_wave,
                                                        n_mels=128,
                                                        n_fft=self.input_nfft,
                                                        hop_length=self.input_stride)
        self.init_wave = self.sample_wave.copy()

    def noising(self, noise_factor=np.random.uniform(0, 0.2)):
        noise = np.random.randn(len(self.sample_wave))
        augmented_data = self.sample_wave + noise_factor * noise
        # Cast back to same data type
        self.sample_wave = augmented_data
        return augmented_data

    def shifting(self, shift_max=10, shift_direction='both'):
        shift = np.random.randint(self.sr * shift_max + 1)
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

    def time_mask(self, mask_ratio=0.1):
        start_point = np.random.randint(0, len(self.sample_wave) - int(mask_ratio * len(self.sample_wave)))
        end_point = start_point + int(mask_ratio * len(self.sample_wave))

        augmented_data = np.hstack((self.sample_wave[:start_point],
                                    np.zeros(end_point - start_point),
                                    self.sample_wave[end_point:]))
        self.sample_wave = augmented_data
        return augmented_data

    def change_pitch(self, pitch_factor=np.random.randint(-5, 5)):
        augmented_data = librosa.effects.pitch_shift(self.sample_wave, self.sr, pitch_factor)
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
                                                   n_mels=128,
                                                   n_fft=self.input_nfft,
                                                   hop_length=self.input_stride)
        return self.mfcc

    def get_random_aug_mfcc(self):
        self.noising()
        self.shifting()
        self.time_mask()
        self.change_pitch()
        self.mfcc = librosa.feature.melspectrogram(self.sample_wave,
                                                   n_mels=128,
                                                   n_fft=self.input_nfft,
                                                   hop_length=self.input_stride)
        return self.mfcc


class AugDset(Dataset):
    def __init__(self):
        root = 'data/genres/'
        self.all_waves = [root + genre + '/' + wav for genre in os.listdir(root) for wav in os.listdir(root + genre)]
        self.all_genres = os.listdir(root)
        self.label_num = len(self.all_genres)

    def __len__(self):
        return len(self.all_waves)

    def __getitem__(self, idx):
        wave = self.all_waves[idx]
        genre = wave.split('/')[2]
        genre_index = self.all_genres.index(genre)
        x = torch.from_numpy(Get_Mfcc(wave).get_random_aug_mfcc())
        y = torch.zeros(self.label_num)
        y[genre_index] = 1

        return x, y


class DSet(Dataset):
    def __init__(self) -> object:
        """

        :rtype: object
        """
        root = 'data/genres/'
        self.all_waves = [root + genre + '/' + wav for genre in os.listdir(root) for wav in os.listdir(root + genre)]
        self.all_genres = os.listdir(root)
        self.label_num = len(self.all_genres)

    def __len__(self):
        return len(self.all_waves)

    def __getitem__(self, idx):
        wave = self.all_waves[idx]
        genre = wave.split('/')[2]
        genre_index = self.all_genres.index(genre)
        x = torch.from_numpy(Get_Mfcc(wave).get_mfcc())
        y = torch.zeros(self.label_num)
        y[genre_index] = 1

        return x, y

