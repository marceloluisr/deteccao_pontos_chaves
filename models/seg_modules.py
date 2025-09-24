import torchvision
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .u2net_components import *
from skimage.filters import threshold_otsu

class SegmentationModule:
    """
    Segmentação semântica da pessoa na imagem.
    - Usa modelo DeepLabV3 pré-treinado do torchvision.
    - Retorna a máscara binária da pessoa (silhueta) e a imagem original.
    """

    def __init__(self):
        # Carrega modelo DeepLabV3 com backbone ResNet50, pré-treinado no COCO
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.eval()  # modo avaliação, desativa dropout etc.

        # Pré-processamento: redimensionar para 512x512 e converter para tensor
        self.preprocess = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])
    
    def extract_silhouette(self, image_path):
        # Abrir imagem com PIL e converter para RGB
        img = Image.open(image_path).convert("RGB")

        # Pré-processar e adicionar dimensão batch (1, C, H, W)
        input_tensor = self.preprocess(img).unsqueeze(0)

        # Forward pass do modelo para segmentação
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]  # saída do modelo

        # COCO class 15 = pessoa
        mask = (output.argmax(0) == 15).byte().cpu().numpy()  # 1=pessoa, 0=fundo
        mask_img = mask * 255  # escala para 0-255 para visualização

        # Mostrar imagem original e silhueta lado a lado
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        axs[0].imshow(img)
        axs[0].set_title("Imagem Original")
        axs[1].imshow(mask_img, cmap="gray")
        axs[1].set_title("Silhueta Extraída")
        for ax in axs: ax.axis("off")
        plt.show()

        # Retorna máscara binária e imagem original como array numpy
        return mask_img, np.array(img)
  

class DeepLabSegmentationModule:
    """
    Segmentação semântica da pessoa na imagem.
    - Usa modelo DeepLabV3 pré-treinado do torchvision.
    - Retorna a máscara binária da pessoa (silhueta).
    """

    def __init__(self, device):
        # Carrega modelo DeepLabV3 com backbone ResNet50, pré-treinado no COCO
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
        self.model = self.model.eval()  # modo avaliação, desativa dropout etc.

    def visualizar_silhueta(self, img, mask):
        # Mostrar imagem original e silhueta lado a lado
        mask_img = mask * 255  # escala para 0-255 para visualização
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        axs[0].imshow(img.cpu().squeeze().permute((1,2,0)).detach().numpy())
        axs[0].set_title("Imagem Original")
        axs[1].imshow(mask_img, cmap="gray")
        axs[1].set_title("Silhueta Extraída")
        for ax in axs: ax.axis("off")
        plt.show()

    def extract_silhouette(self, input_tensor):
        # Forward pass do modelo para segmentação
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]  # saída do modelo

        # COCO class 15 = pessoa
        mask = (output.argmax(0) == 15).byte().detach().cpu().numpy()  # 1=pessoa, 0=fundo

        # Retorna máscara binária 
        return mask
       



class U2NetSegmentationModule:
    """
    Detecção de Objeto Saliente da pessoa na imagem.
    - Usa modelo U2Net pré-treinado para segmentar pessoas.
    - Retorna a máscara binária da pessoa (silhueta) após limiarização da saída da U2Net.
    """

    def __init__(self, model_dir, device):
        """
        model_dir: diretório do modelo U2Net
        """
        # Carrega modelo U2Net pretreinada
        self.model = U2NET(3,1)


        self.model.load_state_dict(torch.load(model_dir))
        self.model = self.model.to(device)
        self.model = self.model.eval()  # modo avaliação

    def normPRED(d):
      # função copiada de https://github.com/xuebinqin/U-2-Net/tree/master
      """
      Realiza normalização MIN-MAX na entrada.
      Args:
        - d: Mapa de Saliência de entrada. Numpy array de tamanho HxWxC
      """
      ma = torch.max(d)
      mi = torch.min(d)

      dn = (d-mi)/(ma-mi)

      return dn

    def otsu_binarization(inp):
      """
        Realiza a binarização adaptativa de OTSU em uma imagem de entrada

        Args:
          - inp: Mapa de Saliência de entrada. Numpy array de tamanho HxWxC

        Return:
          - mask: Máscara binária. Numpy array of shape HxWx1
      """

      # limiarização adaptativa com OTSU
      otsu_thresh = threshold_otsu(inp)
      mask = (inp > otsu_thresh).astype(np.uint8) # 1: objeto, 0: fundo
      return mask



    def visualizar_silhueta(self, img, mask):
        # Mostrar imagem original e silhueta lado a lado
        mask_img = mask * 255  # escala para 0-255 para visualização
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        axs[0].imshow(img.cpu().squeeze().permute((1,2,0)).detach().numpy())
        axs[0].set_title("Imagem Original")
        axs[1].imshow(mask_img, cmap="gray")
        axs[1].set_title("Silhueta Extraída")
        for ax in axs: ax.axis("off")
        plt.show()

    def extract_silhouette(self, input_tensor):
        # Forward pass do modelo para segmentação
        with torch.no_grad():
             saliencia,_,_,_,_,_,_ =  self.model(input_tensor)  # saída do modelo

        # prepara saliencia com valores no intervalo [0,1] para máscara binária
        pred = saliencia[:,0,:,:]
        pred = U2NetSegmentationModule.normPRED(pred)

        pred = pred.squeeze()
        pred = pred.cpu().data.numpy()

        # limiarização adaptativa com OTSU
        mask = U2NetSegmentationModule.otsu_binarization(pred) # 1: objeto, 0: fundo

        return mask

