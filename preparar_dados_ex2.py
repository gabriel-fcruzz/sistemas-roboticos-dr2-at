"""
PREPARAÇÃO DOS DADOS DO EXERCÍCIO 2

Baixa, uma única vez, o que o ex2a.py precisa:

1. imagenet_class_index.json
   Mapa oficial índice -> (wnid, nome da classe) usado pelo Keras em
   decode_predictions. É a mesma tabela que dá sentido à saída de 1000 valores
   da MobileNetV2, tanto no Keras quanto no OpenCV DNN.

2. 12 fotografias reais, uma por categoria, do repositório público
   EliSchwartz/imagenet-sample-images (uma imagem de exemplo por classe do
   ImageNet, com o wnid no nome do arquivo).

POR QUE NÃO USAR AS IMAGENS SINTÉTICAS DO MATERIAL DE AULA
Os exemplos de aula geram retângulos e círculos coloridos para exercitar o fluxo
de arquivos. Serve para testar o pipeline, mas a acurácia top-1 medida sobre
elas é sempre zero: a MobileNetV2 nunca viu figuras geométricas rotuladas como
"caneca". Como o enunciado do item A exige acurácia top-1 na tabela comparativa,
são necessárias fotografias reais de categorias que existam no ImageNet.

O wnid no nome do arquivo dá o rótulo verdadeiro sem qualquer ambiguidade de
digitação, o que elimina a comparação frágil por substring de texto.

RESSALVA HONESTA SOBRE A ACURÁCIA
Estas imagens vêm do próprio ImageNet, a base em que a MobileNetV2 foi treinada.
A acurácia medida aqui é, portanto, otimista em relação ao que a mesma rede
entregaria na câmera de um robô, onde aparecem iluminação ruim, desfoque de
movimento, oclusão e objetos fora das 1000 categorias. O objetivo da métrica
neste exercício é comparar DOIS BACKENDS executando a MESMA rede; para esse fim,
qualquer viés do conjunto afeta os dois igualmente.

Categorias escolhidas: predominantemente urbanas (semáforo, placa, ônibus
escolar, carro esportivo, motoneta, bicicleta, banco de praça, caminhão de
bombeiros, guarda-chuva, cachorro, gato) mais um objeto de interior, por
aderência ao tema de percepção em veículos autônomos e robôs móveis.

Execução:
    .venv/Scripts/python.exe preparar_dados_ex2.py
"""

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "classificacao"
MODELOS = ROOT / "modelos"

URL_CLASS_INDEX = ("https://storage.googleapis.com/download.tensorflow.org/"
                   "data/imagenet_class_index.json")
API_REPO = ("https://api.github.com/repos/EliSchwartz/"
            "imagenet-sample-images/contents/")
RAW_REPO = ("https://raw.githubusercontent.com/EliSchwartz/"
            "imagenet-sample-images/master/")

CATEGORIAS = [
    "traffic_light", "street_sign", "school_bus", "sports_car",
    "moped", "mountain_bike", "park_bench", "fire_engine",
    "umbrella", "golden_retriever", "tabby", "coffee_mug",
]

CABECALHO = {"User-Agent": "AT-DR2-visao-computacional"}


def baixar(url, destino):
    destino.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=CABECALHO)
    with urllib.request.urlopen(req, timeout=120) as r:
        destino.write_bytes(r.read())
    return destino


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    MODELOS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print("1) Índice de classes do ImageNet")
    idx_json = MODELOS / "imagenet_class_index.json"
    if not idx_json.exists():
        baixar(URL_CLASS_INDEX, idx_json)
    mapa = json.loads(idx_json.read_text(encoding="utf-8"))

    wnid_para = {}
    labels = [""] * 1000
    for idx, (wnid, nome) in mapa.items():
        wnid_para[wnid] = (int(idx), nome)
        labels[int(idx)] = nome

    (MODELOS / "imagenet_labels.txt").write_text("\n".join(labels),
                                                 encoding="utf-8")
    print(f"   {len(mapa)} classes -> modelos/imagenet_labels.txt")

    # ------------------------------------------------------------------
    print("2) Lista de imagens de exemplo do repositório público")
    req = urllib.request.Request(API_REPO, headers=CABECALHO)
    with urllib.request.urlopen(req, timeout=120) as r:
        conteudo = json.load(r)
    arquivos = sorted(d["name"] for d in conteudo
                      if d["name"].lower().endswith(".jpeg"))
    print(f"   {len(arquivos)} imagens disponíveis")

    # ------------------------------------------------------------------
    print("3) Download das categorias escolhidas")
    linhas = []
    for categoria in CATEGORIAS:
        nome_arquivo = next((a for a in arquivos if categoria in a), None)
        if nome_arquivo is None:
            print(f"   [ignorada] {categoria}: não encontrada no repositório")
            continue

        wnid = nome_arquivo.split("_")[0]
        if wnid not in wnid_para:
            print(f"   [ignorada] {categoria}: wnid {wnid} fora do índice")
            continue

        idx, nome_classe = wnid_para[wnid]
        destino = DATA / f"{idx:03d}_{nome_classe}.jpg"
        if not destino.exists():
            baixar(RAW_REPO + urllib.parse.quote(nome_arquivo), destino)
        tam_kb = destino.stat().st_size / 1024
        print(f"   {destino.name:38s} classe {idx:3d}  {tam_kb:7.1f} KB")
        linhas.append([destino.name, idx, nome_classe, wnid])

    csv_path = ROOT / "data" / "rotulos_ex2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "classe_esperada", "nome_classe", "wnid"])
        w.writerows(linhas)

    print(f"\n{len(linhas)} imagens em data/classificacao/")
    print(f"Gabarito em {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
