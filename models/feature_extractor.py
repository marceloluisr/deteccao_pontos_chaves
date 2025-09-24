import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class ExtractFeaturesKeypoints(nn.Module):
    """
    Extrai um vetor de características com as distâncias euclidianas entre pares únicos de keypoints.

    Para N keypoints, o vetor de saída terá N(N-1)/2 valores, correspondentes às distâncias entre
    todos os pares distintos (sem repetição e sem pares consigo mesmo).

    Parâmetros:
        keypoints (dict): Dicionário com keypoints nomeados. Exemplo:
            {
                'top_head': (x1, y1),
                'neck': (x2, y2),
                ...
            }

    Retorno:
        torch.Tensor: Vetor 1D com shape (N(N-1)/2), contendo as distâncias euclidianas entre pares únicos.
    """
    def __init__(self):
        super(ExtractFeaturesKeypoints, self).__init__()

    def forward(self, keypoints: dict):
        coords = torch.tensor(list(keypoints.values()), dtype=torch.float32)  # shape: [N, 2]
        pairs = list(combinations(range(len(coords)), 2))  # isso gera pares únicos de indices da lista coords (keypoint)

        distances = []
        for i, j in pairs:
            dist = torch.norm(coords[i] - coords[j])
            distances.append(dist)

        return torch.stack(distances)  # total: [N(N-1)/2]
