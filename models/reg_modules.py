import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

class RegressionModule:
    """
    Converte keypoints em medidas corporais aproximadas.
    Para protótipo, gera valores simulados baseados em proporções.
    """
    def predict_measures(self, keypoints):
        measures = {}
        if not keypoints:
            return measures

        # Exemplo: usar distância entre pontos como proxy
        def distance(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        measures['pescoço'] = distance(keypoints['top_head'], keypoints['neck']) * 1.2
        measures['peito'] = distance(keypoints['shoulders'], keypoints['waist']) * 2
        measures['cintura'] = distance(keypoints['waist'], keypoints['hips']) * 1.5
        measures['quadril'] = distance(keypoints['hips'], keypoints['knees']) * 1.3
        measures['coxa'] = distance(keypoints['hips'], keypoints['knees'])
        measures['joelho'] = distance(keypoints['knees'], keypoints['ankles'])
        measures['panturrilha'] = distance(keypoints['knees'], keypoints['ankles']) * 0.7
        measures['abdomen'] = measures['cintura'] * 0.9
        measures['biceps'] = measures['peito'] * 0.3

        return measures


class DepthwiseSeparableConv1DRegressor(nn.Module):
    """
    Esse módulo usa dois componentes pricipais: 
    a) Depthwise convolution: aplica um único filtro convolucional por canal de feature de entrada (sem agregação de informação)
    b) Pointwise convolution: usa convolução 1×1 para agregar os canais.

    
    # código baseado de https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch/tree/master
    """
    def __init__(self, input_channels=1, input_length=21, output_size=9):
        super(DepthwiseSeparableConv1DRegressor, self).__init__()
        #https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.depthwise = nn.Conv1d(in_channels=input_channels, out_channels=input_channels,
                                   kernel_size=3, padding=1, groups=input_channels)
        self.pointwise = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1, groups=16),  # depthwise
            nn.Conv1d(32, 32, kernel_size=1),  # pointwise
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(32 * input_length, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.relu(self.pointwise(self.depthwise(x)))
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class DepthwiseSeparableConv1DRegressorModule:
    """
    Converte features extraídas dos keypoints em medidas corporais aproximadas usando DepthwiseSeparableConv1DRegressor
    """

    def __init__(self, model_dir,  input_channels=1, input_length=21, output_size=9, scaler_dir=None, device=None):
        """
        model_dir: caminho para o modelo PyTorch (.pt ou .pth)
        scaler_dir: caminho para o scaler sklearn (.pkl)
        device: 'cuda' ou 'cpu'
        """
        self.device = device 

        # Carrega modelo PyTorch
        self.model = DepthwiseSeparableConv1DRegressor(input_channels,input_length,output_size) #torch.load(model_dir, map_location=self.device)
        self.model.load_state_dict(torch.load(model_dir, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Carrega scaler se fornecido
        if scaler_dir is not None:
            self.scaler_regressor = joblib.load(scaler_dir)
        else:
            self.scaler_regressor = None

    @staticmethod
    def preprocess_feature(x):
        """
        Aplica max-normalization para o vetor de característica de entrada.
        """
        row_max = np.max(x, axis=1, keepdims=True)
      
        return x / row_max + 1e-10 # normalizando entrada

    def predict_measures(self, keypoints_features):
        """
        Recebe um vetor de features de pontos chance [1, 21] e retorna medidas corporais aproximadas em dicionário
        """
        measures = {}
        if keypoints_features is None:
            return measures

        # Normaliza
        keypoints_features_norm = self.preprocess_feature(keypoints_features)

        # Converte para tensor [1, 1, 21]
        input_tensor = torch.tensor(keypoints_features_norm, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Predição
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            measures_vector = output_tensor.cpu().detach().numpy()

        # Inversão do scaler para converter predição para medidas aproximadas reais
        if self.scaler_regressor is not None:
            measures_vector = self.scaler_regressor.inverse_transform(measures_vector)

        measures_vector = measures_vector.squeeze()  #  shape [1,9] -> shape [9]

        measures['peito'] = measures_vector[0]
        measures['cintura'] = measures_vector[1]
        measures['quadril'] = measures_vector[2]
        measures['coxa'] = measures_vector[3]
        measures['joelho'] = measures_vector[4]
        measures['panturrilha'] = measures_vector[5]
        measures['abdomen'] = measures_vector[6]
        measures['pescoço'] = measures_vector[7]
        measures['biceps'] = measures_vector[8]

        return measures
        
class XGBoostRegressionModule:
    """
    Converte features extraídas dos keypoints em medidas corporais aproximadas usando XGBoost
    """

    def __init__(self, model_dir, scaler_dir=None):
        """
        model_dir: diretório do modelo XGBoost em formato .pkl. Exemplo: './multioutput_xgb_model.pkl'
        scaler_dir: diretório do sklearn Scaler em formato .pkl. Exemplo: './treino_scaler.pkl'
        """
        # Carrega modelo
        self.model = joblib.load(model_dir)
        if scaler_dir is not None:
          self.scaler_regressor = joblib.load(scaler_dir)
        else:
          self.scaler_regressor = None
    def preprocess_feature(x):
      """
      Aplica row-wise normalization para cada feature vector
      """
      row_max = np.max(x, axis=1, keepdims=True)
      x = x / row_max + 1e-10
      return x

    def predict_measures(self, keypoints_features):
        """
        TODO
        """
        measures = {}
        if keypoints_features is None:
            return measures

        keypoints_features_norm  = XGBoostRegressionModule.preprocess_feature(keypoints_features)
        measures_vector = self.model.predict(keypoints_features_norm)

        if self.scaler_regressor is not None:
          measures_vector = self.scaler_regressor.inverse_transform(measures_vector)

        measures_vector = measures_vector.squeeze() # N

        measures['peito'] = measures_vector[0]
        measures['cintura'] = measures_vector[1]
        measures['quadril'] = measures_vector[2]
        measures['coxa'] = measures_vector[3]
        measures['joelho'] = measures_vector[4]
        measures['panturrilha'] = measures_vector[5]
        measures['abdomen'] = measures_vector[6]
        measures['pescoço'] = measures_vector[7]
        measures['biceps'] = measures_vector[8]


        return measures
