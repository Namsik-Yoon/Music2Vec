import torch
import torch.nn as nn

import Dataset


class FeatureExtractor:
    def __init__(self, model):
        self.forward_model = nn.Sequential(*list(model.children())[:-1]).cpu()
        self.embedding_layer = nn.Sequential(*list(model.children())[-1][:-2]).cpu()

    def feature(self, music_dir, args):
        tensor = torch.from_numpy(Dataset.Get_Mfcc(music_dir, args).get_mfcc())
        tensor = tensor.view(1, tensor.size(0), tensor.size(1)).float()
        genre = music_dir.split('/')[2]
        emb_vec = self.embedding_layer(self.forward_model(tensor.unsqueeze(0)).view(1, self.embedding_layer[0].in_features))
        return emb_vec, genre
