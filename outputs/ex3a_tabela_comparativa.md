# Exercício 3A — YOLOv4-tiny vs. SSD MobileNetV2

| modelo | entrada | FPS médio | FPS no p95 | inferência (ms) | frame completo (ms) | parâmetros | arquivo (MB) |
|---|---|---|---|---|---|---|---|
| YOLOv4-tiny | 416x416 | 23.7 | 21.5 | 36.3 | 42.2 | 6.062.826 | 23.1 |
| SSD MobileNetV2 | 300x300 | 28.4 | 25.8 | 33.4 | 35.2 | 16.876.287 | 66.6 |

## Sensibilidade do YOLO à resolução de entrada

| entrada | inferência (ms) | FPS | objetos por frame | pessoas por frame |
|---|---|---|---|---|
| 256x256 | 15.8 | 63.4 | 0.72 | 0.72 |
| 320x320 | 22.7 | 44.1 | 5.22 | 3.23 |
| 416x416 | 39.1 | 25.6 | 7.05 | 5.05 |
| 512x512 | 53.9 | 18.5 | 7.55 | 5.55 |

Vídeo: vtest.avi, 300 frames, confiança >= 0.4, NMS = 0.4, backend cv2.dnn/CPU.

Concordância entre modelos (IoU >= 0.5): 1823 pares.
