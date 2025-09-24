import os
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def salvar_lista_em_txt(lista_caminhos, caminho_arquivo):
    """
    Salva uma lista de strings (caminhos de arquivos) em um arquivo .txt, 
    onde cada linha representa um item da lista.

    Args:
    lista_caminhos : list[str]
        Lista contendo os caminhos completos ou relativos dos arquivos.

    caminho_arquivo : str
        Caminho do arquivo .txt onde a lista será salva.

    """
    with open(caminho_arquivo, "w") as f:
        for caminho in lista_caminhos:
            f.write(caminho + "\n")

def carregar_lista_de_txt(caminho_arquivo):
    """
    Lê um arquivo .txt contendo uma lista de caminhos (um por linha) 
    e retorna uma lista de strings.

    Args:
    caminho_arquivo : str
        Caminho do arquivo .txt que contém os caminhos a serem carregados.

    Return:

    list[str]
        Lista contendo os caminhos lidos do arquivo.
    """
    with open(caminho_arquivo, "r") as f:
        lista_caminhos = [linha.strip() for linha in f.readlines()]
    return lista_caminhos

def salvar_dataframe(df, caminho_arquivo):
    """
    Salva um DataFrame do pandas em um arquivo CSV.

    Args:
    df : pandas.DataFrame
        O DataFrame que será salvo.
    caminho_arquivo : str
        Caminho completo (incluindo nome do arquivo) onde o CSV será salvo.

   
    """
    df.to_csv(caminho_arquivo, index=False)

def carregar_dataframe(caminho_arquivo):
    """
    Carrega um arquivo CSV em um DataFrame do pandas.

    Args:
    ----------
    caminho_arquivo : str
        Caminho completo do arquivo CSV a ser carregado.

    Returns:
    -------
    pandas.DataFrame
        O DataFrame carregado a partir do arquivo.
    """
    return pd.read_csv(caminho_arquivo)
    
def list_image_files(root_dir, extensions={'.jpg', '.jpeg', '.png', '.bmp'}): # utils junior novo
    """
    Lista todas as imagens recursivamente.
    Args:
      - root_dir: image root directory
      - extensions: the extension
    """
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                image_paths.append(os.path.join(dirpath, fname))
    return image_paths


def criar_scaler_para_split(dataframe, dir_arquivo_saida):
  """
  Criar um scaler baseado em Z-Score normalization a partir dos feature vector reais
  associados com cada ID de amostra. O scaler é uma funçao que permite inversão para 
  permitir a volta para o valor original

  Z-Score Nomralization: um tipo de padronização que substrai a média e divide pelo desvio padrão.

  args:
    dataframe: datataframe originado de medidas_dados_sinteticos.csv e associado com um split
    nome_arquivo_saida: nome do arquivo no formato .pkl que contém objeto StandarScaler do sklearn.

  """
  if ".pkl" not in dir_arquivo_saida:
    raise ValueError("Nome do arquivo deve conter a extensão .pkl")

  # Define colunas de rótulo
  exclude_cols = ['id', 'height', 'split']
  label_cols = [col for col in dataframe.columns if col not in exclude_cols]

  # Extrai rótulos como matriz
  labels_original = dataframe[label_cols].values.astype('float32')

  # Fit e transformação
  scaler = StandardScaler()
  scaler = scaler.fit(labels_original)

  # Salva o scaler para uso na inferência
  joblib.dump(scaler, dir_arquivo_saida)


def salvar_silhuetas(img, mask, output_folder, nome_imagem):
  """
  Salva a máscara binária e a imagem com contornos da silhueta em arquivos PNG.

  Esta função realiza os seguintes passos:
  1. Salva a máscara binária da silhueta.
  2. Detecta contornos na máscara limpa.
  3. Desenha os contornos sobre a imagem original.
  4. Salva a imagem com os contornos desenhados.

  Args:
  ----------
  img : torch.Tensor
      Tensor da imagem original com formato (1, C, H, W), normalizada entre 0 e 1.

  mask : numpy.ndarray
      Máscara binária da silhueta com valores entre 0 e 1.

  output_folder : str
      Caminho da pasta onde os arquivos serão salvos. A pasta será criada se não existir.

  nome_imagem : str
      Nome base para os arquivos gerados (sem extensão). Será usado para nomear os arquivos
      de saída como "<nome_imagem>_mask.png" e "<nome_imagem>_boundary.png".

  Returns:
  -------
  None
      Os arquivos são salvos diretamente no sistema de arquivos. A função não retorna valores.
  """
  # Garantir que a pasta de saída existe
  os.makedirs(output_folder, exist_ok=True)
  # Salvar a máscara binária
  mask_path = os.path.join(output_folder, f"{nome_imagem}_mask.png")
  cv2.imwrite(mask_path, mask * 255)
  # Preparar imagem original
  img_arr = img.cpu().squeeze().permute((1, 2, 0)).detach().numpy().copy()
  img_arr = (img_arr * 255).astype(np.uint8)
  
  #mask_clean = (mask > 0.5).astype(np.uint8) * 255
  #mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  #mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
  mask = (mask > 0.5).astype(np.uint8) * 255
  # Encontrar contornos
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Desenhar contornos sobre a imagem original
  overlay = img_arr.copy()
  cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)
  # Salvar imagem com contornos
  boundary_path = os.path.join(output_folder, f"{nome_imagem}_boundary.png")
  cv2.imwrite(boundary_path, overlay)


def visualizar_keypoints(image_path, keypoints_dict, figure_title="Visualização de Keypoints", save_to_file=None):
    """
    Função utilitária para carregar uma imagem e plotar os pontos chaves sobrepostos
    na imagem
    
    Args:
      image_path (string): diretório da imagem
      keypoints_dict (dict): chave é o local do ponto no corpo humano e valor corresponde às coordenadas
      figure_title (string): título da figura
      save_to_file (string): se fornecida, salva a figura no arquivo especificado
    """
    colours = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta",
        "yellow", "lime", "pink", "brown", "gray", "olive", "teal", "navy", "gold"
    ]

    # Carregando e redimensiona a imagem
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((512, 512), resample=Image.BILINEAR)
    img_array = np.asarray(img_resized)

   
    fig, (ax_img, ax_legend) = plt.subplots(1, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [3, 1]})

    # Exibe a imagem com os keypoints
    ax_img.imshow(img_array)
    ax_img.axis("off")

    legend_entries = []
    for idx, (name, (x, y)) in enumerate(keypoints_dict.items()):
        color = colours[idx % len(colours)]
        ax_img.scatter(x, y, c=color, marker='x', s=100, linewidths=2)
        legend_entries.append((color, name))

    
    ax_legend.axis("off")
    for i, (color, name) in enumerate(legend_entries):
        ax_legend.scatter([], [], c=color, marker='x', s=100, label=name)
    ax_legend.legend(loc='center left', fontsize=12)

    fig.suptitle(figure_title, fontsize=16)
    plt.tight_layout()

    if save_to_file:
        plt.savefig(save_to_file)
    plt.show()

def unnormalize_tensor(tensor):
    """
    Reverte a normalização tipo ImageNet e transforma o tensor [C, H, W] de volta para imagem [H, W, 3].

    Args:
        tensor (torch.Tensor): Tensor normalizado com shape [3, H, W] onde H é a altura e W é a largura.

    Returns:
        np.ndarray: Imagem RGB não normalizada com shape [H, W, 3] e valores em [0, 1]
    """
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to numpy and transpose to [H, W, C]
    img = tensor.numpy().transpose((1, 2, 0))

    # Unnormalize
    img = (img * std) + mean

    # Clip to [0, 1] range
    img = np.clip(img, 0, 1)

    return img
