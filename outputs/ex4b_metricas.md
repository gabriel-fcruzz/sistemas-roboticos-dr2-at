# Métricas consolidadas da disciplina

Todos os valores foram medidos nesta máquina (CPU), pelos scripts indicados na coluna fonte.

| etapa | técnica | tempo (ms) | memória/disco (MB) | qualidade medida | complexidade | fonte |
|---|---|---|---|---|---|---|
| Pré-processamento | Segmentação HSV + morfologia | 2.28 | - | 81% das regiões candidatas descartadas por filtros geométricos | O(n) por pixel, sem treino | AT ex2b.ipynb (média de 30 frames) |
| Pré-processamento | Segmentação HSV da via (asfalto) | 3.40 | - | 30,8% da imagem marcada como navegável; não separa via de calçada | O(n) por pixel, sem treino | AT ex4a.py (cena campus_via) |
| Calibração | calibrateCamera (24 imagens) | - | - | erro de reprojeção 0,0202 px; erro de fx 0,078% | offline, uma vez por câmera | AT ex1a.py |
| Calibração | undistort por frame | 6.45 | - | corrige deslocamento radial de até 27 px na borda | O(n) por pixel, mapas pré-calculáveis | AT ex2b.ipynb |
| Pose | solvePnP + projectPoints (tabuleiro) | 0.27 | - | erro de translação 0,31 mm; rotação 0,028° | iterativo sobre 42 pontos | AT ex1b.py |
| Características | ORB (1000 keypoints) | 5.75 | - | descritor binário de 32 bytes | FAST + BRIEF, sem treino | TP2 ex3a.py |
| Características | AKAZE | 29.40 | - | 2666 keypoints, descritor de 61 bytes | difusão não linear | TP2 ex3a.py |
| Características | SIFT | 52.40 | - | 2419 keypoints, descritor de 128 floats | pirâmide DoG | TP2 ex3a.py |
| Detecção clássica | HOG+SVM pré-treinado (rápido) | 24.90 | - | 1,39 detecção/frame | janela deslizante, 1 escala grossa | TP3 ex1a.py |
| Detecção clássica | HOG+SVM pré-treinado (preciso) | 202.30 | - | 4,22 detecções/frame | janela deslizante, pirâmide fina | TP3 ex1a.py |
| Detecção clássica | HOG+SVM treinado por nós | 5800.00 | - | precisão 0,89 e F1 0,64 com hard negatives | treino próprio + janela deslizante | TP3 ex1b.py |
| Detecção clássica | Haar Cascade (rostos) | - | - | 0 rostos em cena de vigilância a 40 m | cascata de classificadores fracos | AT ex2b.ipynb |
| Subtração de fundo | MOG2 | 4.48 | - | 4,28 objetos/frame (223 FPS) | modelo de mistura por pixel | TP3 ex2a.py |
| Subtração de fundo | KNN | 3.66 | - | 4,39 objetos/frame (273 FPS) | vizinhos por pixel | TP3 ex2a.py |
| Classificação | MLP no MNIST | - | - | 98,00% de acurácia, 535.818 parâmetros | 2 camadas densas | TP3 ex3a.py |
| Classificação | CNN no MNIST | - | - | 99,41% de acurácia, 421.642 parâmetros | 2 blocos convolucionais | TP3 ex3a.py |
| Classificação | MobileNetV2 fine-tuned (gênero) | 14.63 | 11.6 | 90,67% de acurácia em 150 rostos | transfer learning, 10 épocas | TP3 ex4b.py |
| Classificação | Caffe gender_net | 8.19 | 45.7 | 88,00% de acurácia | rede pronta, sem treino | TP3 ex4b.py |
| Classificação | MobileNetV2 via OpenCV DNN (TFLite) | 7.06 | 81.8 | 91,7% top-1 e 100% top-3 em 12 imagens | 3.538.984 parâmetros | AT ex2a.py |
| Classificação | MobileNetV2 via Keras/TensorFlow | 13.87 | 437.0 | mesma acurácia do OpenCV DNN | mesma rede, outro runtime | AT ex2a.py |
| Detecção profunda | YOLOv4-tiny (416x416) | 40.60 | 23.1 | 24,6 FPS; confiança média 80,4%; 6,98 objetos/frame | 6.062.826 parâmetros | AT ex3a.py (300 frames) |
| Detecção profunda | YOLOv4-tiny (320x320) | 31.30 | 23.1 | 31,9 FPS; 5,22 objetos/frame | mesma rede, entrada menor | AT ex3a.py (varredura) |
| Detecção profunda | SSD MobileNetV2 (300x300) | 32.20 | 66.6 | 31,1 FPS; confiança média 72,3%; 7,74 objetos/frame | 16.876.287 parâmetros | AT ex3a.py (300 frames) |
| Rastreamento | IoU + ID persistente (sobre YOLO) | 36.30 | - | 27,6 FPS; 98 IDs; 10,19 ID switches/min | associação gulosa O(t*d) | AT ex3b.py (795 frames) |
| Rastreamento | CamShift | - | - | perdeu o alvo no frame 275 | deslocamento de média sobre histograma | TP3 ex2a.py |
| Segmentação | DeepLabV3-ResNet50 (VOC) | 951.00 | 538.0 | 93,9% dos pixels caem em 'fundo' nas cenas externas | ResNet50 + ASPP | AT ex4a.py (7 cenas) |
| Segmentação | FCN-ResNet50 (VOC) | 777.00 | 538.0 | 99,4% de concordância com o DeepLabV3 | ResNet50 + cabeça FCN | AT ex4a.py (7 cenas) |
