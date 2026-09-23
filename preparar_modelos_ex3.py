"""
PREPARAÇÃO DOS MODELOS DO EXERCÍCIO 3 (YOLO e SSD)

Baixa, uma única vez, os arquivos que o material de aula espera encontrar em
modelos/ (ver 14_yolov4_tiny_opencv_dnn.py e 15_ssd_mobilenet_opencv_dnn.py):

YOLOv4-tiny (Darknet)
    yolov4-tiny.cfg      arquitetura, do repositório oficial AlexeyAB/darknet
    yolov4-tiny.weights  pesos treinados em COCO, do release oficial
    coco.names           as 80 classes do COCO

SSD MobileNetV2 (TensorFlow Object Detection API)
    ssd_mobilenet_v2_coco.pb     grafo congelado, do Model Zoo do TensorFlow
    ssd_mobilenet_v2_coco.pbtxt  descrição da topologia para o importador do
                                 OpenCV, do repositório opencv_extra

Por que esses dois modelos: são exatamente os citados no enunciado, ambos
treinados no mesmo conjunto (COCO, 80 classes, com person, car, bicycle,
bus, truck e traffic light) e ambos carregáveis pelo módulo DNN do OpenCV.
Rodar os dois no MESMO backend é o que torna a comparação de FPS honesta:
diferenças de runtime deixam de contaminar a medida, e o que sobra é a
diferença de arquitetura.

Observação sobre o download do SSD: o Model Zoo distribui um .tar.gz de ~188 MB
que contém checkpoints de treino além do grafo. Extraímos apenas o
frozen_inference_graph.pb (~67 MB) e descartamos o resto.

Execução:
    .venv/Scripts/python.exe preparar_modelos_ex3.py
"""

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELOS = ROOT / "modelos"

BASE_DARKNET = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/"
ARQUIVOS = {
    "yolov4-tiny.cfg": BASE_DARKNET + "cfg/yolov4-tiny.cfg",
    "coco.names": BASE_DARKNET + "data/coco.names",
    "yolov4-tiny.weights": ("https://github.com/AlexeyAB/darknet/releases/"
                            "download/darknet_yolo_v4_pre/yolov4-tiny.weights"),
    "ssd_mobilenet_v2_coco.pbtxt": ("https://raw.githubusercontent.com/opencv/"
                                    "opencv_extra/master/testdata/dnn/"
                                    "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"),
    # Mapa de rótulos do COCO com 90 ids. O SSD da TensorFlow Object Detection
    # API devolve o id "de papel" do COCO (1 a 90, com lacunas), enquanto o
    # coco.names do Darknet lista as 80 classes sem lacuna. Sem este mapa, as
    # classes do SSD saem deslocadas em relação às do YOLO.
    "mscoco_label_map.pbtxt": ("https://raw.githubusercontent.com/tensorflow/"
                               "models/master/research/object_detection/data/"
                               "mscoco_label_map.pbtxt"),
}

SSD_TAR = ("http://download.tensorflow.org/models/object_detection/"
           "ssd_mobilenet_v2_coco_2018_03_29.tar.gz")
SSD_PB = "ssd_mobilenet_v2_coco.pb"

CABECALHO = {"User-Agent": "AT-DR2-visao-computacional"}


def baixar(url, destino, descricao=""):
    if destino.exists() and destino.stat().st_size > 0:
        print(f"   [ok] {destino.name} já existe "
              f"({destino.stat().st_size/1024/1024:.1f} MB)")
        return destino
    print(f"   baixando {destino.name} {descricao}...")
    req = urllib.request.Request(url, headers=CABECALHO)
    with urllib.request.urlopen(req, timeout=600) as r, \
            open(destino, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"   [ok] {destino.name} ({destino.stat().st_size/1024/1024:.1f} MB)")
    return destino


def baixar_ssd():
    destino = MODELOS / SSD_PB
    if destino.exists() and destino.stat().st_size > 0:
        print(f"   [ok] {SSD_PB} já existe "
              f"({destino.stat().st_size/1024/1024:.1f} MB)")
        return destino

    tmp = Path(tempfile.gettempdir()) / "ssd_mobilenet_v2_coco.tar.gz"
    if not tmp.exists() or tmp.stat().st_size < 1_000_000:
        print("   baixando pacote do SSD MobileNetV2 (~188 MB)...")
        req = urllib.request.Request(SSD_TAR, headers=CABECALHO)
        with urllib.request.urlopen(req, timeout=1800) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)

    print("   extraindo apenas o grafo congelado...")
    with tarfile.open(tmp, "r:gz") as tar:
        membro = next(m for m in tar.getmembers()
                      if m.name.endswith("frozen_inference_graph.pb"))
        extraido = tar.extractfile(membro)
        destino.write_bytes(extraido.read())
    print(f"   [ok] {SSD_PB} ({destino.stat().st_size/1024/1024:.1f} MB)")
    tmp.unlink(missing_ok=True)
    return destino


def main():
    MODELOS.mkdir(parents=True, exist_ok=True)
    print("MODELOS DO EXERCÍCIO 3\n")

    print("1) YOLOv4-tiny e classes do COCO")
    for nome, url in ARQUIVOS.items():
        if nome.startswith("ssd"):
            continue
        baixar(url, MODELOS / nome)

    print("\n2) SSD MobileNetV2")
    baixar(ARQUIVOS["ssd_mobilenet_v2_coco.pbtxt"],
           MODELOS / "ssd_mobilenet_v2_coco.pbtxt")
    baixar_ssd()

    print("\n3) Verificação")
    import cv2

    from common import caminho_ascii
    yolo = cv2.dnn.readNetFromDarknet(
        caminho_ascii(MODELOS / "yolov4-tiny.cfg"),
        caminho_ascii(MODELOS / "yolov4-tiny.weights"))
    print("   YOLOv4-tiny carregado. Camadas de saída:",
          yolo.getUnconnectedOutLayersNames())

    ssd = cv2.dnn.readNetFromTensorflow(
        caminho_ascii(MODELOS / SSD_PB),
        caminho_ascii(MODELOS / "ssd_mobilenet_v2_coco.pbtxt"))
    print("   SSD MobileNetV2 carregado. Camadas:", len(ssd.getLayerNames()))

    classes = (MODELOS / "coco.names").read_text(encoding="utf-8").splitlines()
    print(f"   {len(classes)} classes COCO; as de trânsito: "
          + ", ".join(c for c in classes
                      if c in {"person", "bicycle", "car", "motorbike",
                               "bus", "truck", "traffic"}))


if __name__ == "__main__":
    main()
