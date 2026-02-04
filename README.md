# ⚔️ Trafalgar Law — ROOM & SHAMBLES com YOLO + MediaPipe

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?logo=google)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-red?logo=roboflow)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

---

## 📌 Visão Geral

Este projeto implementa um sistema interativo inspirado nas técnicas do personagem **Trafalgar Law (One Piece)**:

- 🌀 **ROOM** — criação de uma esfera energética ao redor da mão  
- ⚡ **SHAMBLES** — troca ou desaparecimento de objetos detectados  
- ✋ **Reconhecimento de gestos** — ativação dos poderes com sinais manuais  

O objetivo principal foi **combinar visão computacional e aprendizado de máquina** para criar uma experiência visual interativa em tempo real, utilizando:

- Detecção de mãos com **MediaPipe HandLandmarker**  
- Classificação de gestos com **MLP treinado manualmente**  
- Detecção de objetos com **YOLOv8**  
- Efeitos visuais com **OpenCV**  

---

## 🏆 Funcionalidades

- ✅ **ROOM**: esfera translúcida com partículas orbitando a mão  
- ✅ **SHAMBLES**:  
  - Com 2 objetos → troca de posições  
  - Com 1 objeto → desaparecimento com fade animado  
- ✅ **Seleção de objetos**: apontar com o dedo para marcar até 2 alvos  
- ✅ **Cancelamento**: gesto de CANCEL limpa seleções e desativa poderes  
- ✅ **Feedback visual**: caixas pulsantes, labels e animações dinâmicas  


---

## 🧩 Implementação Técnica

### 🔎 Detecção de Gestos
- Modelo **MLP** treinado com features das landmarks da mão (63 coordenadas).  
- Gestos suportados: `ROOM`, `SHAMBLES`, `APONTAR`, `CANCEL`.  

### 🎯 Seleção de Objetos
- Objetos detectados pelo **YOLOv8n**.  
- Seleção feita ao apontar para dentro da bounding box.  
- Limite de seleção: `max_sel = 2`.  

### ⚡ SHAMBLES
- **Dois objetos**: troca real das bounding boxes e centros.  
- **Um objeto**: desaparecimento com fade animado (3s de transição + 2s branco fixo).  

### 🌀 ROOM
- Overlay circular translúcido com partículas orbitando.  
- Intensidade e cor variam dinamicamente com o tempo.  

---

## ⏱️ Performance

Rodando em tempo real (~30 FPS) em:

- CPU: Intel i5 (9ª geração)  
- RAM: 8 GB  
- GPU: opcional (YOLO roda em CPU, mas pode ser acelerado em CUDA)  

---

## 🧠 Conceitos Implementados

- Detecção de mãos com MediaPipe  
- Extração de features e classificação com MLP  
- Detecção de objetos com YOLOv8  
- Efeitos visuais com OpenCV (overlay, partículas, fade, distorção)  
- Lógica de seleção persistente e cooldowns para gestos  

---

## ⚙️ Tecnologias Utilizadas

- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- YOLOv8 (Ultralytics)  
- Joblib (para salvar modelo MLP)  

---

## 🎯 Conclusões

- O sistema consegue **combinar gestos e detecção de objetos** em tempo real.  
- O efeito **ROOM** cria uma esfera energética convincente.  
- O **SHAMBLES** agora funciona tanto para troca quanto para desaparecimento animado.  
- Melhorias futuras incluem:  
  - Efeitos visuais mais complexos (partículas mágicas, cores dinâmicas)  
  - Otimização da seleção para evitar falsos positivos  
  - Suporte a múltiplas pessoas sem confundir com o usuário  

---

## 👤 Autor

**Kauã Dias**  
Estudante de Estatística e entusiasta de Ciência de Dados / Visão Computacional

- 🐙 GitHub: [https://github.com/Kauadp](https://github.com/Kauadp)  
- 🔗 LinkedIn: [https://www.linkedin.com/in/kauad/](https://www.linkedin.com/in/kauad/)
