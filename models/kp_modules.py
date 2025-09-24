import cv2
import mediapipe as mp
from rt_pose import PoseEstimationPipeline
import numpy as np
import torch

class KeypointsModule:
    """
    Detecta pontos-chave aproximados da silhueta da pessoa.
    Abordagem simplificada sem usar MediaPipe:
    - Usa o contorno da silhueta para localizar o corpo.
    - Calcula uma bounding box (retângulo) ao redor do contorno.
    - Estima keypoints com base em proporções dentro da bounding box.
    """

    def extract_keypoints(self, mask):
        # Encontrar todos os contornos na máscara
        # cv2.RETR_EXTERNAL -> pega apenas os contornos externos
        # cv2.CHAIN_APPROX_SIMPLE -> reduz número de pontos do contorno
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # Nenhum contorno encontrado → retorna dicionário vazio
            return {}

        # Seleciona o maior contorno, que corresponde à pessoa
        c = max(contours, key=cv2.contourArea)

        # Calcula a bounding box do contorno
        # x, y -> canto superior esquerdo da caixa
        # w, h -> largura e altura da caixa
        x, y, w, h = cv2.boundingRect(c)

        # Estima keypoints com base em proporções da bounding box
        keypoints = {
            'top_head': (x + w//2, y),                # topo da cabeça
            'neck': (x + w//2, y + h//10),           # pescoço (10% da altura)
            'shoulders': (x + w//2, y + h//5),       # ombros (20% da altura)
            'waist': (x + w//2, y + h//2),           # cintura (50% da altura)
            'hips': (x + w//2, y + int(h*0.6)),      # quadril (60% da altura)
            'knees': (x + w//2, y + int(h*0.8)),     # joelhos (80% da altura)
            'ankles': (x + w//2, y + h),             # tornozelos (base da bounding box)
        }

        # Retorna dicionário com keypoints aproximados
        return keypoints
  
class MediaPipeKeypointDetector:
    """
    Classe responsável por implementar o uso da biblioteca MediaPipe para detecção de pontos chaves
    do corpo.
    """
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def detect(self, image_np, mask):
        results = self.pose.process(image_np) #cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        if not results.pose_landmarks: # se nenhum ponto chave foi detectado
            return None

        landmarks = results.pose_landmarks.landmark
        

        def get_point(index):
            """
            Obtém coordenadas de pontos chaves dectados dado indice
            """
            lm = landmarks[index]
            return int(lm.x * w), int(lm.y * h)

        def get_top_head_using_centroid_mask(mask, image_height, image_width):
            """
            Estima o topo da cabeça lançando (casting) um raio vertical para cima a partir do nariz (alinhado com o centroide X)
            e parando quando a máscara terminar (máscara == 0).

            Bascicamente, essa função está:
            1) Alinhando o nariz verticalmente com o centroide da máscara.
            3) Projeta um raio vertical para cima a partir dessa posição alinhada.
            4) Para o lançamento raio quando sai da máscara — ou seja, atinge a borda superior da cabeça.
            Args:
                mask (np.ndarray): Máscara binária ou silhueta do corpo (0 or 1)
                image_height, image_width: dimensões da imagem usadas para obter as coordenadas do nariz por MediaPipe
            
              Returns:
                (int, int): Coordenada estimada do topo da cabeça
            """
            nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
            nose_y = int(nose.y * image_height)
            
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0:
                return (0, nose_y)
            # alinhando coordenada horizontal do nariz com o centroid
            centroid_x = int(np.mean(x_indices))
            top_of_head = (centroid_x, nose_y)
            
            # estimando coordenada do top da cabeça
            for y in range(nose_y, -1, -1):
                if mask[y, centroid_x] == 0: # se parar na borda da máscara
                    top_of_head = (centroid_x, y + 1)
                    break

            return top_of_head

        h, w, _ = image_np.shape
        # Construindo pontos chaves
        keypoints = {
            'top_of_head': get_top_head_using_centroid_mask(mask, h, w),
            'neck': tuple(np.mean([get_point(11), get_point(12)], axis=0).astype(int)),
            'shoulder_left': get_point(11),
            'shoulder_right': get_point(12),
            'waist': tuple(np.mean([get_point(23), get_point(24)], axis=0).astype(int)),
            'hip_left': get_point(23),
            'hip_right': get_point(24),
            'knee_left': get_point(25),
            'knee_right': get_point(26),
            'ankle_left': get_point(27),
            'ankle_right': get_point(28),
        }

        return keypoints

class MediaPipeKeypointsModule:
    """
    Detecta pontos-chave usando a silhueta da pessoa e imagem RGB da pessoa.
    Usa o módulo MediaPipeKeypointDetector para detecção de pontos 11 pontos chaves:
    top_head (estimado geometricamente), neck, shoulders (left e right), waist, hips (left e right),
    knees(left e right), ankles (left e right) 
    """
    
    def __init__(self):
        self.detector = MediaPipeKeypointDetector()


    def extract_keypoints(self, image, mask):
        """
        Extrai os pontos-chave da imagem e da máscara usando MediaPipe.
        Retorna um dicionário com os pontos estimados.
        
        Args:
          image (np.ndarray): Imagem RGB da pessoa, com shape (H, W, 3),
                              onde H é a altura e W é a largura da imagem.
          mask (np.ndarray): Máscara binária da silhueta da pessoa, com shape (H, W),
                            contendo valores 0 (fundo) e 1 (corpo).

        Returns:
          dict: Dicionário contendo 11 pontos-chave estimados:
                - top_of_head (estimado geometricamente)
                - neck
                - shoulder_left, shoulder_right
                - waist
                - hip_left, hip_right
                - knee_left, knee_right
                - ankle_left, ankle_right
    
        """
        
      
        # detectando pontos chaves
        return self.detector.detect(image, mask)



class RTPoseKeypointDetector:
    """
    Classe responsável por implementar o uso da biblioteca RT Pose para detecção de pontos chaves
    do corpo.
    """
    def __init__(self, device="cuda", dtype=torch.float32, compile=False):
        self.pipeline = PoseEstimationPipeline(
            object_detection_checkpoint="PekingU/rtdetr_r50vd_coco_o365",
            pose_estimation_checkpoint="usyd-community/vitpose-plus-small",
            device=device,
            dtype=dtype,
            compile=compile
        )

    def detect(self, image_np, mask):
        # Converte para PIL
        #image_pil = Image.fromarray(image_np.astype(np.uint8))

        # Executa o pipeline
        try:
           output = self.pipeline(image_np)
        except RuntimeError as e:
           #print("[INFO] Erro interno no pipeline:", e)
           return None

        if output.keypoints_xy is None or len(output.keypoints_xy) == 0: # se não detectou nenhum ponto chave
            return None

        keypoints = output.keypoints_xy[0]  
        keypoints_px = [(int(x), int(y)) for x, y in keypoints]

        def get_point(index):
            """
            Obtém coordenadas de pontos chaves dectados dado indice
            """
            return keypoints_px[index]

        def get_top_head_using_centroid_mask(mask, image_height, image_width):
            """
            Estima o topo da cabeça lançando (casting) um raio vertical para cima a partir do nariz (alinhado com o centroide X)
            e parando quando a máscara terminar (máscara == 0).

            Bascicamente, essa função está:
            1) Alinhando o nariz verticalmente com o centroide da máscara.
            3) Projeta um raio vertical para cima a partir dessa posição alinhada.
            4) Para o lançamento raio quando sai da máscara — ou seja, atinge a borda superior da cabeça.
            Args:
                mask (np.ndarray): Máscara binária ou silhueta do corpo (0 or 1)
                image_height, image_width: dimensões da imagem usadas para obter as coordenadas do nariz por MediaPipe
            
              Returns:
                (int, int): Coordenada estimada do topo da cabeça
            """
            
            nose_y = get_point(0)[1]
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0:
                return (0, nose_y)
            centroid_x = int(np.mean(x_indices))
            top_of_head = (centroid_x, nose_y)
            for y in range(nose_y, -1, -1):
                if mask[y, centroid_x] == 0:
                    top_of_head = (centroid_x, y + 1)
                    break
            return top_of_head


        # Construindo pontos chaves
        h, w = image_np.shape[:2]
        keypoints_dict = {
            'top_of_head': get_top_head_using_centroid_mask(mask, h, w),
            'neck': tuple(np.mean([get_point(5), get_point(6)], axis=0).astype(int)),
            'shoulder_left': get_point(5),
            'shoulder_right': get_point(6),
            'waist': tuple(np.mean([get_point(11), get_point(12)], axis=0).astype(int)),
            'hip_left': get_point(11),
            'hip_right': get_point(12),
            'knee_left': get_point(13),
            'knee_right': get_point(14),
            'ankle_left': get_point(15),
            'ankle_right': get_point(16),
        }

        return keypoints_dict

class RTPoseKeypointDetectorModule:
    """
    Detecta pontos-chave usando a silhueta da pessoa e imagem RGB da pessoa.
    Usa o módulo RTPoseKeypointDetector para detecção de pontos 11 pontos chaves:
    top_head (estimado geometricamente), neck, shoulders (left e right), waist, hips (left e right),
    knees(left e right), ankles (left e right)
    """

    def __init__(self):
        self.detector = RTPoseKeypointDetector()



    def extract_keypoints(self, image, mask):
        """
        Extrai os pontos-chave da imagem e da máscara usando RTPose.
        Retorna um dicionário com os pontos estimados.

        Args:
          image (np.ndarray): Imagem RGB da pessoa, com shape (H, W, 3),
                              onde H é a altura e W é a largura da imagem.
          mask (np.ndarray): Máscara binária da silhueta da pessoa, com shape (H, W),
                            contendo valores 0 (fundo) e 1 (corpo).

        Returns:
          dict: Dicionário contendo 11 pontos-chave estimados:
                - top_of_head (estimado geometricamente)
                - neck
                - shoulder_left, shoulder_right
                - waist
                - hip_left, hip_right
                - knee_left, knee_right
                - ankle_left, ankle_right

        """


        # detectando pontos chaves
        return self.detector.detect(image, mask)