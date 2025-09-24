
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os 
import torchvision.transforms as T

class U2NetToTensor(object):
    """
    Versão levemente modificada de
    https://github.com/xuebinqin/U-2-Net/blob/ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29/data_loader.py#L103

    Converte uma imagem ndarray para tensor.
    Esta versão ignora 'label' e 'imidx', retornando apenas a imagem normalizada como tensor.
    """

    def __call__(self, image):
        # Normaliza a imagem para [0, 1]
        image = image / np.max(image)

        # Inicializa imagem com 3 canais
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

        # Aplica normalização tipo ImageNet
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        # Transforma para formato [C, H, W]
        tmpImg = tmpImg.transpose((2, 0, 1))

        # Retorna apenas o tensor da imagem
        return torch.from_numpy(tmpImg).float()

class RegressaoDataset(Dataset):
    """
    Dataset para nossa tarefa de regressão

    Este dataset associa imagens a vetores de rótulos numéricos extraídos dos DataFrames dos splits originado do aqruivo de medidas.
    Os rótulos podem ser opcionalmente normalizados com um objeto scaler (ex: StandardScaler).

    As imagens são transformadas com redimensionamento, conversão para tensor e normalizadas no intervalo 0,1.

    Parameters
    ----------
    image_paths : list of str
        Lista de caminhos absolutos para os arquivos de imagem.
    dataframe : pandas.DataFrame
        DataFrame contendo os rótulos associados a cada imagem. Deve conter uma coluna 'id'
        que corresponde ao nome da pasta onde a imagem está localizada.
    scaler : object
        Objeto de normalização (ex: StandardScaler) com método `.transform()` para aplicar nos rótulos.
        Se not None, aplica o scaler aos rótulos. Caso contrário, os rótulos são mantidos em seu formato original.
        Default é False.
    dataset_path: str
        O diretório onde as imagens estão armazenadas
    exclude_cols : list of str, optional
        Lista de colunas a serem excluídas do conjunto de rótulos. Default é ['id', 'height', 'split'].
    u2net : bool, optional
        Se True, aplica o preprocessamento da U2Net. Caso contrário, o prepocessamento com ToTensor é aplicado para normalizar a imagem converter 
        de PIL para Tensor com valores intervalo [0,1]. [Atenção]: esse argumento deve ser True se você está usando U2Net durante o treinamento
        ou inferência.
    
    """

    def __init__(self, image_paths, dataframe, scaler, dataset_path, exclude_cols=['id', 'height', 'split'], u2net=False):
        self.image_paths = image_paths
        self.dataset_path = dataset_path
        self.df = dataframe

        if not u2net:
            self.transform = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(), # normaliza [0,1]
               # T.Normalize(mean=[0.485, 0.456, 0.406],
               #             std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((512, 512)),
                U2NetToTensor(), # normaliza [0,1]
            ])
            
    
        self.label_cols = [col for col in self.df.columns if col not in exclude_cols]

        if scaler is not None:
            self.id_to_label = {
                row['id']: scaler.transform(row[self.label_cols].values.astype('float32').reshape(1, -1))
                for _, row in self.df.iterrows()
            }
        else:
            self.id_to_label = {
                row['id']: row[self.label_cols].values.astype('float32').reshape(1, -1)
                for _, row in self.df.iterrows()
            }

    def __len__(self):
        """
        Retorna o número total de amostras no dataset.

        Returns
        -------
        int
            Número de imagens no dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Recupera a imagem e o vetor de rótulos correspondente ao índice fornecido.

        Parameters
        ----------
        idx : int
            Índice da amostra desejada.

        Returns
        -------
        image : torch.Tensor
            Imagem transformada como tensor normalizado.
        label : torch.Tensor
            Vetor de rótulos associado à imagem, convertido para tensor float32.

        Raises
        ------
        ValueError
            Se o ID da imagem não for encontrado no mapeamento de rótulos.
        """
        img_path = self.image_paths[idx]
        img_path = os.path.join(self.dataset_path, img_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        folder_id = os.path.basename(os.path.dirname(img_path))
        label = self.id_to_label.get(folder_id)

        if label is None:
            raise ValueError(f"ID '{folder_id}' não encontrado no split.")

        return image, torch.tensor(label, dtype=torch.float32)
