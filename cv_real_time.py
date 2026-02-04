import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import joblib
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Função pra criar o efeito ROOM
def desenhar_room(imagem, centro_mao):
    """Cria esfera azul brilhante estilo One Piece"""
    overlay = imagem.copy()
    h, w = imagem.shape[:2]
    
    cx, cy = centro_mao
    
    raios = [120, 100, 80, 60, 40]
    alphas = [0.1, 0.15, 0.2, 0.3, 0.5]
    
    for raio, alpha in zip(raios, alphas):
        cv2.circle(overlay, (cx, cy), raio, (255, 200, 100), -1)
        imagem = cv2.addWeighted(imagem, 1-alpha, overlay, alpha, 0)
    
    cv2.circle(imagem, (cx, cy), 30, (255, 255, 200), -1)
    
    num_particulas = 20
    for i in range(num_particulas):
        angulo = (time.time() * 2 + i * (360/num_particulas)) % 360
        rad = np.radians(angulo)
        dist = 130
        px = int(cx + dist * np.cos(rad))
        py = int(cy + dist * np.sin(rad))
        cv2.circle(imagem, (px, py), 3, (255, 200, 100), -1)
    
    return imagem

def aplicar_shambles(imagem):
    """Distorção visual estilo teleporte"""
    h, w = imagem.shape[:2]
    
    centro_x, centro_y = w // 2, h // 2
    intensidade = 50
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    dx = x - centro_x
    dy = y - centro_y
    dist = np.sqrt(dx**2 + dy**2)
    
    fator = np.sin(time.time() * 10) * 0.1
    x_novo = x + (dx / (dist + 1)) * intensidade * fator
    y_novo = y + (dy / (dist + 1)) * intensidade * fator
    
    x_novo = np.clip(x_novo, 0, w-1).astype(np.float32)
    y_novo = np.clip(y_novo, 0, h-1).astype(np.float32)
    
    distorcido = cv2.remap(imagem, x_novo, y_novo, cv2.INTER_LINEAR)

    return distorcido

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode = VisionRunningMode.IMAGE,
    num_hands=2
)

model = joblib.load("models/MLP.pkl")
scaler = joblib.load("models/scaler.pkl")
yolo = YOLO("yolov8n.pt")

# Estado do jogo
room_ativo = False
mao_room = None 
gestos_maos = ["Nenhum", "Nenhum"]
probs_maos = [0.0, 0.0]
alpha_room = 0.0
objetos_cache = []
frame_count = 0
objetos_selecionados = []  # MUDA: Lista de 2 objetos
swap_ativo = False  # ADICIONA: Flag pro efeito de swap
swap_timer = 0  # ADICIONA: Timer do efeito

# ADICIONA: Cooldowns pra evitar spam
ultimo_gesto = {"ROOM": 0, "SHAMBLES": 0, "CANCEL": 0, "APONTAR": 0}
COOLDOWN = 1.0  # 1 segundo entre gestos
max_sel = 2  # número máximo de objetos selecionados


with HandLandmarker.create_from_options(options) as detector:
    camera = cv2.VideoCapture(0)  
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow("Trafalgar Law")
    
    fps_limit = 30
    prev_time = time.time()
    frame_timestamp = 0

    while True:
        sucesso, imagem = camera.read()
        if not sucesso:
            print("Falha na câmera")
            break

        now = time.time()
        if now - prev_time < 1/fps_limit:
            time.sleep(0.001)
            continue
        prev_time = now

        h, w = imagem.shape[:2]
        frame_count += 1

        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        mp_imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        frame_timestamp += 33

        resultados = detector.detect(mp_imagem)

        gestos_maos = ["Nenhum", "Nenhum"]
        probs_maos = [0.0, 0.0]
        
        # Roda YOLO a cada 10 frames
        if frame_count % 10 == 0:
            resultados_yolo = yolo(imagem, verbose=False)
            objetos_cache = []
            for box in resultados_yolo[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = yolo.names[cls]
                objetos_cache.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'label': label,
                    'centro': ((x1+x2)//2, (y1+y2)//2)
                })
        
        # Desenha objetos
        for i, obj in enumerate(objetos_cache):
            x1, y1, x2, y2 = obj['bbox']
            
            # Cor: Verde se selecionado, Azul normal
            if obj in objetos_selecionados:
                cor = (0, 255, 0)
                espessura = 4
                idx = objetos_selecionados.index(obj) + 1
                cv2.putText(imagem, f"ALVO {idx}", (x1, y1-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cor = (255, 0, 0)
                espessura = 2
            
            cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, espessura)
            cv2.putText(imagem, f"{obj['label']}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        if resultados.hand_landmarks:
            for idx, hand_landmarks in enumerate(resultados.hand_landmarks):
                if idx >= 2:
                    break
                    
                coords = []
                
                for landmark in hand_landmarks:
                    x = int(landmark.x * imagem.shape[1])
                    y = int(landmark.y * imagem.shape[0])
                    cv2.circle(imagem, (x, y), 5, (0, 0, 255), -1)
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    x1, y1 = int(start.x*imagem.shape[1]), int(start.y*imagem.shape[0])
                    x2, y2 = int(end.x*imagem.shape[1]), int(end.y*imagem.shape[0])
                    cv2.line(imagem, (x1,y1),(x2,y2), (0,255,0), 2)

                if len(coords) == 63:
                    coords_array = np.array(coords).reshape(1, -1)
                    coords_scaled = scaler.transform(coords_array)
                    predicao = model.predict(coords_scaled)[0]
                    probabilidades = model.predict_proba(coords_scaled)[0]
                    prob_maxima = np.max(probabilidades)
                    
                    gestos_maos[idx] = predicao
                    probs_maos[idx] = prob_maxima

        # ROOM (com cooldown)
        if "ROOM" in gestos_maos and probs_maos[gestos_maos.index("ROOM")] > 0.85:
            if now - ultimo_gesto["ROOM"] > COOLDOWN:
                room_ativo = True
                mao_room = gestos_maos.index("ROOM")
                ultimo_gesto["ROOM"] = now

        # CANCEL (com cooldown)
        if "CANCEL" in gestos_maos and probs_maos[gestos_maos.index("CANCEL")] > 0.85:
            if now - ultimo_gesto["CANCEL"] > COOLDOWN:
                room_ativo = False
                mao_room = None
                objetos_selecionados = []
                swap_ativo = False
                ultimo_gesto["CANCEL"] = now

        # APONTAR (seleciona objetos, com cooldown)
        if "APONTAR" in gestos_maos and room_ativo and probs_maos[gestos_maos.index("APONTAR")] > 0.85:
            if now - ultimo_gesto["APONTAR"] > COOLDOWN and len(objetos_selecionados) < max_sel:
                idx_apontar = gestos_maos.index("APONTAR")
                if len(resultados.hand_landmarks) > idx_apontar:
                    hand = resultados.hand_landmarks[idx_apontar]
                    ponta_dedo = hand[8]
                    x_dedo = int(ponta_dedo.x * imagem.shape[1])
                    y_dedo = int(ponta_dedo.y * imagem.shape[0])
                    
                    cv2.circle(imagem, (x_dedo, y_dedo), 15, (0, 255, 255), 3)
                    
                    for obj in objetos_cache:
                        if obj in objetos_selecionados:
                            continue
                        x1, y1, x2, y2 = obj['bbox']
                        if x1 <= x_dedo <= x2 and y1 <= y_dedo <= y2:
                            objetos_selecionados.append(obj)
                            ultimo_gesto["APONTAR"] = now
                            break

                    
                    # Verifica se tá apontando pra algum objeto
                    for obj in objetos_cache:
                        x1, y1, x2, y2 = obj['bbox']

                        # 🚫 Evita selecionar a si mesmo
                        if obj['label'].lower() == "person":
                            # se o dedo está dentro da bounding box da pessoa → é você
                            if x1 <= x_dedo <= x2 and y1 <= y_dedo <= y2:
                                # ignora essa seleção
                                continue

                        if obj in objetos_selecionados:
                            cor = (0, 255, 0)
                            espessura = 4
                            idx = objetos_selecionados.index(obj) + 1
                            cv2.putText(imagem, f"ALVO {idx}", (x1, y1-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            # círculo pulsante no centro
                            pulsar = int(8 + 4*np.sin(time.time()*5))
                            cv2.circle(imagem, obj['centro'], pulsar, (0,255,0), 2)
                        else:
                            cor = (255, 0, 0)
                            espessura = 2

                        cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, espessura)
                        cv2.putText(imagem, f"{obj['label']}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

                        # ✅ Seleção normal de qualquer outro objeto
                        if x1 <= x_dedo <= x2 and y1 <= y_dedo <= y2 and obj not in objetos_selecionados:
                            objetos_selecionados.append(obj)
                            ultimo_gesto["APONTAR"] = now
                            break



        # SHAMBLES
        if "SHAMBLES" in gestos_maos and room_ativo and probs_maos[gestos_maos.index("SHAMBLES")] > 0.85:
            if now - ultimo_gesto["SHAMBLES"] > COOLDOWN and len(objetos_selecionados) > 0:
                swap_ativo = True
                swap_timer = now
                ultimo_gesto["SHAMBLES"] = now

        # efeito visual
        if swap_ativo:
            if now - swap_timer < 5.0:
                imagem = aplicar_shambles(imagem)
                cv2.putText(imagem, "SHAMBLES!!!", (w//2 - 200, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10)
                cv2.putText(imagem, "SHAMBLES!!!", (w//2 - 200, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)

                if len(objetos_selecionados) == 2:
                    c1 = objetos_selecionados[0]['centro']
                    c2 = objetos_selecionados[1]['centro']
                    cv2.arrowedLine(imagem, c1, c2, (0, 0, 255), 4, tipLength=0.3)
                    cv2.arrowedLine(imagem, c2, c1, (0, 0, 255), 4, tipLength=0.3)
                elif len(objetos_selecionados) == 1:
                    x1, y1, x2, y2 = objetos_selecionados[0]['bbox']
                    
                    # cria overlay transparente do mesmo tamanho da imagem
                    overlay = imagem.copy()
                    
                    # calcula progresso do fade (0 → 1 ao longo de 2s)
                    progresso = (now - swap_timer) / 2.0
                    progresso = min(max(progresso, 0.0), 1.0)
                    
                    # desenha retângulo branco sobre a bounding box
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
                    
                    # alpha aumenta com o tempo → objeto vai sumindo
                    alpha = progresso
                    imagem = cv2.addWeighted(overlay, alpha, imagem, 1 - alpha, 0)
                    
                    # feedback visual
                    cv2.putText(imagem, "DESAPARECEU!", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)


            else:
                swap_ativo = False
                objetos_selecionados = []


        # ROOM ativo com efeito
        if room_ativo:
            alpha_room = min(1.0, alpha_room + 0.05)
            if mao_room is not None and len(resultados.hand_landmarks) > mao_room:
                hand = resultados.hand_landmarks[mao_room]
                centro = hand[9]
                cx = int(centro.x * imagem.shape[1])
                cy = int(centro.y * imagem.shape[0])
                imagem = desenhar_room(imagem, (cx, cy))
            
            cv2.putText(imagem, "ROOM", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
            cv2.putText(imagem, "ROOM", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200, 100), 3)
            
            # Mostra quantos objetos selecionados
            cv2.putText(imagem, f"Alvos: {len(objetos_selecionados)}/2", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        else: 
            alpha_room = max(0.0, alpha_room - 0.1)
            mao_room = None

        # SHAMBLES com efeito (dura 3 segundos)
        if swap_ativo:
            if now - swap_timer < 3.0:  # Efeito dura 3 segundos
                imagem = aplicar_shambles(imagem)
                cv2.putText(imagem, "SHAMBLES!!!", (w//2 - 200, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10)
                cv2.putText(imagem, "SHAMBLES!!!", (w//2 - 200, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)

                if len(objetos_selecionados) == 2:
                    obj1, obj2 = objetos_selecionados

                    # troca as bounding boxes
                    bbox1, bbox2 = obj1['bbox'], obj2['bbox']
                    obj1['bbox'], obj2['bbox'] = bbox2, bbox1

                    # troca os centros
                    c1, c2 = obj1['centro'], obj2['centro']
                    obj1['centro'], obj2['centro'] = c2, c1

                    # feedback visual
                    cv2.arrowedLine(imagem, c1, c2, (0, 0, 255), 4, tipLength=0.3)
                    cv2.arrowedLine(imagem, c2, c1, (0, 0, 255), 4, tipLength=0.3)
                    cv2.putText(imagem, "SWAP!", (w//2 - 100, h//2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)



                elif len(objetos_selecionados) == 1:
                    x1, y1, x2, y2 = objetos_selecionados[0]['bbox']
                    overlay = imagem.copy()

                    tempo_passado = now - swap_timer

                    if tempo_passado <= 3.0:
                        # fase de fade animado
                        progresso = tempo_passado / 3.0
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
                        alpha = progresso
                        imagem = cv2.addWeighted(overlay, alpha, imagem, 1 - alpha, 0)
                    else:
                        # fase branca fixa (2 segundos extras)
                        cv2.rectangle(imagem, (x1, y1), (x2, y2), (255, 255, 255), -1)

                    cv2.putText(imagem, "DESAPARECEU!", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            else:
                swap_ativo = False
                objetos_selecionados = []
                


        cv2.imshow("Trafalgar Law", imagem)
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord("r"):
            room_ativo = False
            mao_room = None
            objetos_selecionados = []
            swap_ativo = False
            
        if tecla == ord("q"): 
            break

camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)