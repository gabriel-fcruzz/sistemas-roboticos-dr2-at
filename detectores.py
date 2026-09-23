"""
DETECTORES YOLO E SSD SOBRE O MÓDULO DNN DO OPENCV

Módulo compartilhado pelos scripts ex3a.py (comparação) e ex3b.py
(rastreamento). Mantém os dois detectores atrás da mesma interface:

    det = DetectorYOLO()         # ou DetectorSSD()
    caixas, scores, classes, ms = det.detectar(frame)

com caixas no formato (x, y, w, h) em pixels do frame original.

POR QUE OS DOIS NO MESMO BACKEND
Rodar YOLO pelo Ultralytics (PyTorch) e SSD pelo OpenCV mediria, além da
arquitetura, a diferença entre dois runtimes e dois níveis de otimização. Com
ambos em cv2.dnn/CPU, o runtime é constante e a comparação isola o que
interessa: o custo e a qualidade de cada arquitetura.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from common import caminho_ascii

ROOT = Path(__file__).resolve().parent
MODELOS = ROOT / "modelos"

# Limiares exigidos/adotados no enunciado do item A.
CONF_MIN = 0.40      # confiança mínima para considerar uma detecção
NMS_LIMIAR = 0.40    # Non-Maximum Suppression, conforme pede o enunciado

# Classes de interesse em percepção de trânsito. Usadas para filtrar as saídas
# e para a contagem por classe nas tabelas.
CLASSES_TRANSITO = {"person", "bicycle", "car", "motorbike", "motorcycle",
                    "bus", "truck", "traffic light"}


def carregar_nomes_coco80():
    """As 80 classes do COCO na ordem do Darknet (índice direto do YOLO)."""
    return (MODELOS / "coco.names").read_text(encoding="utf-8").splitlines()


def carregar_nomes_coco90():
    """
    Mapa id -> nome do COCO com 90 ids, usado pela TensorFlow Object Detection
    API. Os ids do COCO original vão de 1 a 90 com lacunas (categorias que
    foram removidas do conjunto final), enquanto o Darknet compacta tudo em 80.
    Sem este mapa, as classes do SSD apareceriam deslocadas.
    """
    texto = (MODELOS / "mscoco_label_map.pbtxt").read_text(encoding="utf-8")
    nomes, id_atual = {}, None
    for linha in texto.splitlines():
        linha = linha.strip()
        if linha.startswith("id:"):
            id_atual = int(linha.split(":")[1])
        elif linha.startswith("display_name:") and id_atual is not None:
            nomes[id_atual] = linha.split(":", 1)[1].strip().strip('"')
            id_atual = None
    return nomes


def contar_parametros(net):
    """
    Soma os elementos de todos os tensores de peso da rede.

    cv2.dnn não expõe um contador de parâmetros; getParam(camada, i) devolve o
    i-ésimo blob aprendido de cada camada (pesos e viés). Percorremos todas as
    camadas até o importador acusar que não há mais blobs.
    """
    total = 0
    for i in range(len(net.getLayerNames()) + 1):
        for j in range(4):
            try:
                p = net.getParam(i, j)
            except cv2.error:
                break
            if p is None:
                break
            total += int(np.prod(p.shape))
    return total


class DetectorBase:
    nome = "base"
    arquivos = ()

    def tamanho_disco_mb(self):
        return sum((MODELOS / a).stat().st_size for a in self.arquivos) / (1024 ** 2)

    def parametros(self):
        return contar_parametros(self.net)


class DetectorYOLO(DetectorBase):
    """
    YOLOv4-tiny (Darknet) via cv2.dnn.readNetFromDarknet.

    Detector de estágio único: a imagem é dividida em uma grade e cada célula
    prediz caixas, confiança de objeto e probabilidades de classe em uma única
    passagem. A variante "tiny" reduz a profundidade do backbone e usa duas
    escalas de saída, em vez de três, para caber em hardware modesto.
    """

    nome = "YOLOv4-tiny"
    arquivos = ("yolov4-tiny.cfg", "yolov4-tiny.weights")

    def __init__(self, entrada=416):
        self.entrada = entrada
        self.net = cv2.dnn.readNetFromDarknet(
            caminho_ascii(MODELOS / "yolov4-tiny.cfg"),
            caminho_ascii(MODELOS / "yolov4-tiny.weights"))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.saidas = self.net.getUnconnectedOutLayersNames()
        self.classes = carregar_nomes_coco80()

    def detectar(self, frame, conf_min=CONF_MIN, nms=NMS_LIMIAR):
        h, w = frame.shape[:2]
        # scalefactor 1/255: o Darknet treina com pixels normalizados em [0,1].
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
                                     (self.entrada, self.entrada),
                                     swapRB=True, crop=False)
        t0 = time.perf_counter()
        self.net.setInput(blob)
        saidas = self.net.forward(self.saidas)
        ms = (time.perf_counter() - t0) * 1000

        caixas, scores, classes = [], [], []
        for saida in saidas:
            for det in saida:
                # det = [cx, cy, w, h, objectness, p_classe_0 ... p_classe_79]
                pontuacoes = det[5:]
                idc = int(np.argmax(pontuacoes))
                conf = float(pontuacoes[idc]) * float(det[4])
                if conf < conf_min:
                    continue
                cx, cy, bw, bh = det[0] * w, det[1] * h, det[2] * w, det[3] * h
                caixas.append([int(cx - bw / 2), int(cy - bh / 2),
                               int(bw), int(bh)])
                scores.append(conf)
                classes.append(self.classes[idc])

        # O YOLO emite várias caixas por objeto (uma por célula/âncora); a NMS
        # mantém a de maior confiança e descarta as que se sobrepõem acima do
        # limiar de IoU.
        indices = cv2.dnn.NMSBoxes(caixas, scores, conf_min, nms)
        indices = np.array(indices).ravel() if len(indices) else []
        return ([caixas[i] for i in indices],
                [scores[i] for i in indices],
                [classes[i] for i in indices], ms)


class DetectorSSD(DetectorBase):
    """
    SSD MobileNetV2 (TensorFlow Object Detection API) via
    cv2.dnn.readNetFromTensorflow.

    Também é detector de estágio único, mas prediz a partir de caixas-âncora
    definidas em vários níveis do backbone MobileNetV2. A entrada é fixa em
    300x300, o que o torna rápido e, ao mesmo tempo, limita a detecção de
    objetos pequenos: um pedestre distante pode ocupar menos de 10 px depois do
    redimensionamento.
    """

    nome = "SSD MobileNetV2"
    arquivos = ("ssd_mobilenet_v2_coco.pb", "ssd_mobilenet_v2_coco.pbtxt")

    def __init__(self, entrada=300):
        self.entrada = entrada
        self.net = cv2.dnn.readNetFromTensorflow(
            caminho_ascii(MODELOS / "ssd_mobilenet_v2_coco.pb"),
            caminho_ascii(MODELOS / "ssd_mobilenet_v2_coco.pbtxt"))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.classes = carregar_nomes_coco90()

    def detectar(self, frame, conf_min=CONF_MIN, nms=NMS_LIMIAR):
        h, w = frame.shape[:2]
        # ENTRADA CRUA, em 0-255, sem subtrair média e sem escalar.
        # O grafo congelado da TensorFlow Object Detection API carrega o próprio
        # sub-grafo Preprocessor, que já normaliza os pixels para [-1, 1]. Se
        # aplicarmos a normalização também no blobFromImage, como fazem os
        # tutoriais genéricos de OpenCV DNN, a imagem é normalizada duas vezes:
        # medimos confiança máxima de 0.058 (nenhuma detecção acima de 0.4)
        # contra 0.934 com a entrada crua, no mesmo frame. O erro é silencioso,
        # porque a rede continua respondendo, apenas sem detectar nada.
        blob = cv2.dnn.blobFromImage(frame, 1.0,
                                     (self.entrada, self.entrada),
                                     (0, 0, 0), swapRB=True, crop=False)
        t0 = time.perf_counter()
        self.net.setInput(blob)
        saida = self.net.forward()
        ms = (time.perf_counter() - t0) * 1000

        caixas, scores, classes = [], [], []
        for det in saida[0, 0]:
            conf = float(det[2])
            if conf < conf_min:
                continue
            idc = int(det[1])
            x1, y1 = int(det[3] * w), int(det[4] * h)
            x2, y2 = int(det[5] * w), int(det[6] * h)
            caixas.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            scores.append(conf)
            classes.append(self.classes.get(idc, f"classe_{idc}"))

        # O grafo do SSD já executa NMS internamente (nó NonMaxSuppression).
        # Aplicamos assim mesmo, com o mesmo limiar 0.4 do YOLO, porque o
        # enunciado exige a etapa e porque manter o pós-processamento idêntico
        # elimina mais uma diferença entre os dois lados da comparação.
        indices = cv2.dnn.NMSBoxes(caixas, scores, conf_min, nms)
        indices = np.array(indices).ravel() if len(indices) else []
        return ([caixas[i] for i in indices],
                [scores[i] for i in indices],
                [classes[i] for i in indices], ms)


# ---------------------------------------------------------------------------
# Desenho
# ---------------------------------------------------------------------------

def cor_da_classe(nome):
    """Cor estável por classe, derivada do hash do nome."""
    h = abs(hash(nome))
    return (60 + h % 180, 60 + (h // 180) % 180, 60 + (h // 32400) % 180)


def desenhar_deteccoes(frame, caixas, scores, classes, titulo=None,
                       fps=None, so_transito=False):
    out = frame.copy()
    for (x, y, w, h), s, c in zip(caixas, scores, classes):
        if so_transito and c not in CLASSES_TRANSITO:
            continue
        cor = cor_da_classe(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), cor, 2)
        etiqueta = f"{c} {s*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x, max(0, y - th - 6)), (x + tw + 4, y), cor, -1)
        cv2.putText(out, etiqueta, (x + 2, max(10, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if titulo:
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        texto = titulo if fps is None else f"{titulo}   {fps:.1f} FPS"
        cv2.putText(out, texto, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    return out
