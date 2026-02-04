
# Esse arquivo tem como objetivo capturar dados espaciais para classificação dos dados usando Computação Visual (CV)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
import time


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),      # indicador
    (0, 9), (9, 10), (10, 11), (11, 12), # médio
    (0, 13), (13, 14), (14, 15), (15, 16), # anelar
    (0, 17), (17, 18), (18, 19), (19, 20), # mindinho
    (5, 9), (9, 13), (13, 17)            # palma base
]

# Configurações do HandLandmarker
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# MUDANÇA: Trocado de VIDEO para IMAGE para evitar problemas com timestamps em webcam virtual
options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_hands=1
)

dataset = []
modo_gravacao = None
frames_restantes = 0
contador = {"ROOM":0, "SHAMBLES":0, "CANCEL":0, "SCAN":0, "APONTAR":0}

with HandLandmarker.create_from_options(options) as detector:
    camera = cv2.VideoCapture(0)  
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # MUDANÇA: Limita buffer para webcam virtual

    # MUDANÇA: Cria janela antes do loop
    cv2.namedWindow("Câmera com Mãos")

    fps_limit = 30   # Limita processamento a 30 FPS
    prev_time = time.time()
    frame_timestamp = 0

    while True:
        sucesso, imagem = camera.read()
        if not sucesso:
            print("Falha na câmera")
            break

        # Limita FPS
        now = time.time()
        if now - prev_time < 1/fps_limit:
            time.sleep(0.001)  # MUDANÇA: Libera CPU
            continue
        prev_time = now

        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        mp_imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        frame_timestamp += 33

        # MUDANÇA: Usando detect() em vez de detect_for_video() por conta do modo IMAGE
        resultados = detector.detect(mp_imagem)

        coords = None
        if resultados.hand_landmarks:
            for hand_landmarks in resultados.hand_landmarks:
                coords = []
                # Desenha landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * imagem.shape[1])
                    y = int(landmark.y * imagem.shape[0])
                    cv2.circle(imagem, (x, y), 5, (0, 0, 255), -1)
                    coords.extend([landmark.x, landmark.y, landmark.z])
                # Desenha conexões
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    x1, y1 = int(start.x*imagem.shape[1]), int(start.y*imagem.shape[0])
                    x2, y2 = int(end.x*imagem.shape[1]), int(end.y*imagem.shape[0])
                    cv2.line(imagem, (x1,y1),(x2,y2), (0,255,0), 2)

        # Grava frames do gesto
        if modo_gravacao and coords is not None:
            dataset.append(coords + [modo_gravacao])
            contador[modo_gravacao] += 1
            frames_restantes -= 1
            if frames_restantes <= 0:
                modo_gravacao = None

        # Mostra contagem na tela
        for i, (gesto, total) in enumerate(contador.items()):
            cv2.putText(imagem, f"{gesto}: {total}", (10, 30 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Câmera com Mãos", imagem)
        tecla = cv2.waitKey(1) & 0xFF  # MUDANÇA: Adicionado máscara & 0xFF

        # Começa gravação de 100 frames por gesto
        if tecla == ord("r"): modo_gravacao, frames_restantes = "ROOM", 100
        if tecla == ord("s"): modo_gravacao, frames_restantes = "SHAMBLES", 100
        if tecla == ord("c"): modo_gravacao, frames_restantes = "CANCEL", 100
        if tecla == ord("a"): modo_gravacao, frames_restantes = "APONTAR", 100
        if tecla == ord("n"): modo_gravacao, frames_restantes = "SCAN", 100
        if tecla == ord("q"): break

# Salva dataset
if dataset:
    df = pd.DataFrame(dataset)
    df.to_csv("data/dados.csv", index=False)
    print("Salvo:", len(dataset), "amostras")
else:
    print("Nenhuma amostra coletada")

camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1) 