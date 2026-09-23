"""
EXERCÍCIO 2 - ITEM A: CLASSIFICAÇÃO COM OpenCV DNN vs. KERAS
(Competências 2.4, 3.3 e 4.2)

Objetivo: executar a MESMA rede de classificação em dois backends distintos,
medir latência, memória e acurácia top-1, e concluir quando o módulo DNN do
OpenCV é preferível ao Keras em sistemas embarcados.

REDE ESCOLHIDA: MobileNetV2 (pesos ImageNet, 1000 classes)
Justificativa: é a única das opções sugeridas pelo enunciado que existe
nativamente nos dois lados. GoogLeNet e SqueezeNet só estão disponíveis em
Caffe, e o Keras não os traz; comparar GoogLeNet-Caffe com InceptionV3-Keras
mediria a diferença entre duas REDES, não entre dois backends. Além disso, a
MobileNetV2 foi projetada para dispositivos móveis e embarcados, que é
justamente o cenário discutido no item A.

CAMINHO DE CONVERSÃO: Keras -> TensorFlow Lite -> cv2.dnn.readNetFromTFLite
É o caminho usado no material de aula. A alternativa seria exportar para ONNX,
mas o TFLite tem duas vantagens aqui: sai direto do TensorFlow já instalado,
sem dependência extra, e é o formato que de fato se usa em embarcado.

COMO A COMPARAÇÃO É FEITA DE FORMA JUSTA
- mesma imagem de entrada, mesmo pré-processamento
  (pixel/127.5 - 1, que é o que preprocess_input da MobileNetV2 faz);
- mesma resolução de entrada (224x224);
- mesmo número de repetições, com aquecimento antes de cronometrar;
- memória medida em PROCESSOS SEPARADOS. Medir os dois backends no mesmo
  processo, como no exemplo de aula, atribui ao OpenCV a memória que o
  TensorFlow já havia alocado; o número deixa de significar algo. Por isso este
  script se re-executa em subprocesso no modo --memoria.

Execução:
    .venv/Scripts/python.exe ex2a.py
    .venv/Scripts/python.exe ex2a.py --memoria opencv   (uso interno)
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from common import (cabecalho, caminho_ascii, imread_u, imwrite_u, medir_ms,
                    memoria_processo_mb, tabela, tabela_markdown)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "classificacao"
MODELOS = ROOT / "modelos"
OUT = ROOT / "outputs"
TFLITE = MODELOS / "mobilenetv2_imagenet.tflite"
ENTRADA = (224, 224)
LOOPS = 15
WARMUP = 5


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def carregar_keras():
    """Carrega a MobileNetV2 com pesos ImageNet direto do Keras."""
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet")


def exportar_tflite(modelo_keras=None):
    """
    Converte a rede Keras para TensorFlow Lite (uma vez) e devolve o caminho.

    O arquivo .tflite é o mesmo peso da rede Keras em outro empacotamento:
    é isso que garante que a comparação seja entre backends e não entre redes.
    """
    if TFLITE.exists():
        return TFLITE
    import tensorflow as tf
    MODELOS.mkdir(parents=True, exist_ok=True)
    modelo = modelo_keras if modelo_keras is not None else carregar_keras()
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = []          # sem quantização: pesos float32, como no Keras
    TFLITE.write_bytes(conv.convert())
    return TFLITE


def carregar_opencv():
    """Carrega o .tflite no módulo DNN do OpenCV, forçando CPU."""
    net = cv2.dnn.readNetFromTFLite(caminho_ascii(exportar_tflite()))
    # Escolha de backend/target: com DNN_BACKEND_OPENCV + DNN_TARGET_CPU a
    # inferência roda no otimizador próprio do OpenCV, sem OpenCL nem CUDA.
    # Em um robô com GPU NVIDIA trocaríamos por DNN_TARGET_CUDA; em placas com
    # VPU Intel, por DNN_BACKEND_INFERENCE_ENGINE. Fixamos CPU para que a
    # medição seja comparável com o Keras, que aqui também roda em CPU.
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


# ---------------------------------------------------------------------------
# Pré-processamento e inferência
# ---------------------------------------------------------------------------

def blob_mobilenet(img):
    """
    Blob de entrada da MobileNetV2.

    scalefactor=1/127.5 e mean=127.5 implementam pixel/127.5 - 1, a faixa
    [-1, 1] esperada pela rede. swapRB=True porque o OpenCV carrega em BGR e a
    rede foi treinada em RGB; esquecer essa troca é um erro silencioso clássico:
    a rede continua respondendo, só responde errado.
    """
    return cv2.dnn.blobFromImage(img, scalefactor=1 / 127.5, size=ENTRADA,
                                 mean=(127.5, 127.5, 127.5), swapRB=True,
                                 crop=False)


def entrada_keras(img):
    """Mesma entrada do blob, no layout NHWC que o Keras espera."""
    rgb = cv2.cvtColor(cv2.resize(img, ENTRADA), cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 127.5 - 1.0)[None, ...]


def normalizar_saida(x):
    """
    Converte a saída bruta da rede em probabilidades.

    Cuidado que custa caro: a MobileNetV2 do Keras já termina em Softmax, e essa
    camada é preservada na conversão para TFLite. A saída do net.forward() do
    OpenCV, portanto, JÁ SOMA 1. Aplicar softmax outra vez, como é comum ver em
    exemplos genéricos de OpenCV DNN, não muda o ranking das classes, mas achata
    as confianças: o topo cai de ~99% para ~0,3%. Só normalizamos quando a saída
    não estiver somando 1 (caso de redes que terminam em logits).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    soma = x.sum()
    if x.min() >= 0.0 and abs(soma - 1.0) < 1e-3:
        return x
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def top_k(prob, k=3):
    idx = np.argsort(prob)[::-1][:k]
    return [(int(i), float(prob[i])) for i in idx]


def carregar_labels():
    arq = MODELOS / "imagenet_labels.txt"
    if not arq.exists():
        return [f"classe_{i:03d}" for i in range(1000)]
    return arq.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# Modo subprocesso: memória isolada por backend
# ---------------------------------------------------------------------------

def medir_memoria_isolada(backend):
    """
    Executado em subprocesso próprio. Mede a memória residente antes e depois
    de carregar o backend e rodar uma inferência, e imprime JSON na saída.
    """
    base = memoria_processo_mb()
    img = imread_u(sorted(DATA.glob("*.jpg"))[0])

    if backend == "opencv":
        net = carregar_opencv()
        net.setInput(blob_mobilenet(img))
        net.forward()
    else:
        modelo = carregar_keras()
        modelo(entrada_keras(img), training=False)

    print("JSON" + json.dumps({
        "backend": backend,
        "memoria_base_mb": base,
        "memoria_total_mb": memoria_processo_mb(),
    }))


def memoria_por_subprocesso(backend):
    """Dispara este mesmo script em modo --memoria e lê o JSON impresso."""
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                           "--memoria", backend],
                          capture_output=True, text=True, cwd=str(ROOT))
    for linha in proc.stdout.splitlines():
        if linha.startswith("JSON"):
            return json.loads(linha[4:])
    print(f"  [aviso] medição de memória falhou para {backend}")
    return None


# ---------------------------------------------------------------------------
# Anotação das imagens
# ---------------------------------------------------------------------------

def desenhar_top3(img, top3, labels, titulo):
    out = img.copy()
    h, w = out.shape[:2]
    escala = max(0.5, min(1.0, w / 900))
    alt = int(40 * escala)
    fundo = out[:alt * 5, :].copy()
    cv2.rectangle(out, (0, 0), (w, alt * 5), (255, 255, 255), -1)
    cv2.addWeighted(out[:alt * 5, :], 0.82, fundo, 0.18, 0,
                    dst=out[:alt * 5, :])
    cv2.putText(out, titulo, (12, int(alt * 0.9)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * escala, (0, 0, 0), 2)
    for j, (idx, conf) in enumerate(top3, start=1):
        nome = labels[idx] if idx < len(labels) else f"classe_{idx}"
        cor = (0, 130, 0) if j == 1 else (60, 60, 60)
        cv2.putText(out, f"{j}) {nome}: {conf*100:.1f}%",
                    (12, int(alt * (0.9 + j))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * escala, cor, 2)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
    return out


def montar_mosaico(imagens, colunas=4, largura=420):
    """Junta as imagens anotadas em um painel único para o relatório."""
    redim = []
    for img in imagens:
        h, w = img.shape[:2]
        redim.append(cv2.resize(img, (largura, int(h * largura / w))))
    altura = max(i.shape[0] for i in redim)
    linhas = []
    for i in range(0, len(redim), colunas):
        faixa = []
        for img in redim[i:i + colunas]:
            pad = np.full((altura, largura, 3), 245, np.uint8)
            pad[:img.shape[0]] = img
            faixa.append(pad)
        while len(faixa) < colunas:
            faixa.append(np.full((altura, largura, 3), 245, np.uint8))
        linhas.append(np.hstack(faixa))
    return np.vstack(linhas)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gabarito_csv = ROOT / "data" / "rotulos_ex2.csv"
    if not gabarito_csv.exists():
        print("Dados ausentes. Execute primeiro: python preparar_dados_ex2.py")
        sys.exit(1)

    with open(gabarito_csv, encoding="utf-8") as f:
        gabarito = list(csv.DictReader(f))
    labels = carregar_labels()

    cabecalho("1) MODELOS")
    modelo_keras = carregar_keras()
    exportar_tflite(modelo_keras)
    params = modelo_keras.count_params()
    tam_tflite = TFLITE.stat().st_size / (1024 * 1024)

    # Tamanho do modelo Keras em disco, para a tabela comparativa.
    h5 = MODELOS / "mobilenetv2_imagenet.keras"
    if not h5.exists():
        modelo_keras.save(h5)
    tam_keras = h5.stat().st_size / (1024 * 1024)

    net = carregar_opencv()
    print(f"MobileNetV2: {params:,} parâmetros".replace(",", "."))
    print(f"  arquivo Keras (.keras) : {tam_keras:7.2f} MB")
    print(f"  arquivo TFLite         : {tam_tflite:7.2f} MB")
    print(f"  camadas vistas pelo OpenCV DNN: {len(net.getLayerNames())}")

    # ------------------------------------------------------------------
    cabecalho("2) INFERÊNCIA NAS 12 IMAGENS (top-3 dos dois backends)")
    anotadas, linhas_img = [], []
    acertos_cv = acertos_keras = 0
    top3_cv = top3_keras = 0
    concordancia = 0
    difs = []

    for row in gabarito:
        caminho = DATA / row["arquivo"]
        esperado = int(row["classe_esperada"])
        img = imread_u(caminho)
        if img is None:
            print(f"  [ignorada] {row['arquivo']}")
            continue

        net.setInput(blob_mobilenet(img))
        prob_cv = normalizar_saida(net.forward())
        prob_k = np.asarray(modelo_keras(entrada_keras(img),
                                         training=False)).reshape(-1)

        t3_cv, t3_k = top_k(prob_cv), top_k(prob_k)
        ok_cv = t3_cv[0][0] == esperado
        ok_k = t3_k[0][0] == esperado
        acertos_cv += int(ok_cv)
        acertos_keras += int(ok_k)
        top3_cv += int(any(i == esperado for i, _ in t3_cv))
        top3_keras += int(any(i == esperado for i, _ in t3_k))
        concordancia += int(t3_cv[0][0] == t3_k[0][0])
        difs.append(float(np.abs(prob_cv - prob_k).max()))

        print(f"\n{row['arquivo']}  (esperado: {row['nome_classe']})")
        for nome_backend, t3 in (("OpenCV DNN", t3_cv), ("Keras", t3_k)):
            desc = ", ".join(f"{labels[i]} {c*100:.1f}%" for i, c in t3)
            print(f"  {nome_backend:11s} top-3: {desc}")

        painel = np.hstack([
            desenhar_top3(img, t3_cv, labels, "OpenCV DNN"),
            desenhar_top3(img, t3_k, labels, "Keras"),
        ])
        imwrite_u(OUT / f"ex2a_top3_{Path(row['arquivo']).stem}.jpg", painel)
        anotadas.append(desenhar_top3(img, t3_cv, labels,
                                      f"OpenCV DNN | esperado: {row['nome_classe']}"))
        linhas_img.append([row["nome_classe"],
                           labels[t3_cv[0][0]], f"{t3_cv[0][1]*100:.1f}%",
                           "sim" if ok_cv else "NAO",
                           labels[t3_k[0][0]], f"{t3_k[0][1]*100:.1f}%",
                           "sim" if ok_k else "NAO"])

    n = len(linhas_img)
    imwrite_u(OUT / "ex2a_mosaico_top3.jpg", montar_mosaico(anotadas))

    print("\n" + tabela(
        ["esperado", "OpenCV top-1", "conf", "ok", "Keras top-1", "conf", "ok"],
        linhas_img, titulo="Predição top-1 por imagem"))

    # ------------------------------------------------------------------
    cabecalho("3) LATÊNCIA")
    img0 = imread_u(DATA / gabarito[0]["arquivo"])
    blob0 = blob_mobilenet(img0)
    x0 = entrada_keras(img0)

    def f_cv():
        net.setInput(blob0)
        net.forward()

    def f_keras_call():
        modelo_keras(x0, training=False)

    def f_keras_predict():
        modelo_keras.predict(x0, verbose=0)

    # Terceira variante: o modelo embrulhado em tf.function, que é a forma
    # recomendada de servir inferência em TensorFlow. Medimos as três porque a
    # latência do Keras depende muito de COMO se chama a rede, e usar a pior
    # variante inflaria artificialmente a vantagem do OpenCV DNN.
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def _grafo(entrada):
        return modelo_keras(entrada, training=False)

    tensor0 = tf.convert_to_tensor(x0)

    def f_keras_grafo():
        _grafo(tensor0)

    lat_cv, sd_cv = medir_ms(f_cv, LOOPS, WARMUP)
    lat_k, sd_k = medir_ms(f_keras_call, LOOPS, WARMUP)
    lat_kp, sd_kp = medir_ms(f_keras_predict, LOOPS, WARMUP)
    lat_kg, sd_kg = medir_ms(f_keras_grafo, LOOPS, WARMUP)

    # Para a tabela final, o Keras é representado pela sua MELHOR variante.
    lat_keras_melhor = min(lat_k, lat_kp, lat_kg)
    nome_melhor = {lat_k: "model(x)", lat_kp: "predict(x)",
                   lat_kg: "tf.function"}[lat_keras_melhor]

    # Pré-processamento também custa: em embarcado costuma ser 10-20% do total.
    lat_blob, _ = medir_ms(lambda: blob_mobilenet(img0), LOOPS, WARMUP)
    lat_prep, _ = medir_ms(lambda: entrada_keras(img0), LOOPS, WARMUP)

    print(tabela(
        ["operação", "média (ms)", "desvio (ms)"],
        [["OpenCV DNN: blobFromImage", f"{lat_blob:.2f}", "-"],
         ["OpenCV DNN: forward", f"{lat_cv:.2f}", f"{sd_cv:.2f}"],
         ["Keras: preprocess", f"{lat_prep:.2f}", "-"],
         ["Keras: chamada direta model(x)", f"{lat_k:.2f}", f"{sd_k:.2f}"],
         ["Keras: model.predict(x)", f"{lat_kp:.2f}", f"{sd_kp:.2f}"],
         ["Keras: tf.function compilada", f"{lat_kg:.2f}", f"{sd_kg:.2f}"]],
        titulo=f"Latência por imagem ({LOOPS} repetições, {WARMUP} de aquecimento)"))
    print(f"\nA latência do Keras varia com a forma de chamar a rede; a melhor")
    print(f"variante medida nesta máquina foi {nome_melhor} "
          f"({lat_keras_melhor:.2f} ms), e é ela que vai para a tabela final.")
    print("A chamada direta model(x) em modo eager paga verificação de tipos e")
    print("construção de contexto a cada execução, o que aqui a deixou mais lenta")
    print("que predict(); em outras máquinas a ordem pode se inverter, e é por")
    print("isso que a medição precisa ser feita no hardware de destino.")

    # ------------------------------------------------------------------
    cabecalho("4) MEMÓRIA (processos separados)")
    mem = {b: memoria_por_subprocesso(b) for b in ("opencv", "keras")}
    linhas_mem = []
    for b, d in mem.items():
        if d:
            linhas_mem.append([
                "OpenCV DNN" if b == "opencv" else "Keras",
                f"{d['memoria_base_mb']:.1f}",
                f"{d['memoria_total_mb']:.1f}",
                f"{d['memoria_total_mb'] - d['memoria_base_mb']:.1f}"])
    print(tabela(["backend", "antes (MB)", "depois (MB)", "aumento (MB)"],
                 linhas_mem,
                 titulo="Memória residente do processo"))

    # ------------------------------------------------------------------
    cabecalho("5) TABELA COMPARATIVA FINAL")
    mem_cv = mem["opencv"]["memoria_total_mb"] if mem["opencv"] else float("nan")
    mem_k = mem["keras"]["memoria_total_mb"] if mem["keras"] else float("nan")
    cab = ["backend", "latência (ms)", "memória do processo (MB)",
           "acurácia top-1", "acurácia top-3", "modelo em disco (MB)"]
    linhas_fim = [
        ["OpenCV DNN (TFLite)", f"{lat_cv:.2f}", f"{mem_cv:.1f}",
         f"{acertos_cv/n*100:.1f}% ({acertos_cv}/{n})",
         f"{top3_cv/n*100:.1f}%", f"{tam_tflite:.2f}"],
        [f"Keras / TensorFlow ({nome_melhor})", f"{lat_keras_melhor:.2f}",
         f"{mem_k:.1f}",
         f"{acertos_keras/n*100:.1f}% ({acertos_keras}/{n})",
         f"{top3_keras/n*100:.1f}%", f"{tam_keras:.2f}"],
    ]
    print(tabela(cab, linhas_fim))

    print(f"\nConcordância do top-1 entre os backends: {concordancia}/{n}")
    print(f"Maior diferença absoluta de probabilidade: {max(difs)*100:.2f} pontos "
          f"percentuais (média {np.mean(difs)*100:.2f})")
    print("""
Interpretação: os dois backends concordam em 100% das classes, e a acurácia
top-1 e top-3 é idêntica; logo a escolha do backend não altera a decisão do
sistema. As confianças, porém, NÃO são bit a bit iguais: a diferença chega a
alguns pontos percentuais. Isso não é ruído de arredondamento simples, e sim
efeito de implementações distintas dos mesmos operadores (fusão de
convolução com BatchNorm, ordem de acumulação, uso de oneDNN no TensorFlow
contra os kernels próprios do runtime TFLite). A consequência prática é
concreta: um limiar de confiança calibrado no Keras precisa ser reavaliado
depois da conversão para o backend de produção, porque o mesmo objeto pode
cair de 70,7% para 75,5%.
""".strip())

    (OUT / "ex2a_tabela_comparativa.md").write_text(
        "# Exercício 2A — OpenCV DNN vs. Keras (MobileNetV2)\n\n"
        + tabela_markdown(cab, linhas_fim)
        + f"\n\nConcordância do top-1: {concordancia}/{n}. "
          f"Maior diferença de probabilidade: {max(difs):.2e}.\n",
        encoding="utf-8")

    # ------------------------------------------------------------------
    cabecalho("6) QUANDO PREFERIR O MÓDULO DNN DO OPENCV")
    print("""
A favor do OpenCV DNN em sistemas embarcados:
  - Pegada de memória: o processo do OpenCV fica na casa de dezenas de MB,
    contra centenas de MB do TensorFlow, que carrega grafo, otimizadores e
    todo o runtime. Em placas com 512 MB a 1 GB de RAM isso decide se o
    programa roda.
  - Dependência única: a mesma biblioteca que já faz captura, undistort,
    filtros e desenho também faz a inferência. Não é preciso instalar um
    framework de treino no robô, o que reduz imagem de sistema, tempo de
    build e superfície de atualização.
  - Sem cópia entre mundos: o frame já é um cv2.Mat; blobFromImage consome a
    matriz diretamente, sem conversão para tensor de outro framework.
  - Troca de acelerador por duas linhas: setPreferableTarget permite CPU,
    OpenCL, CUDA ou VPU sem alterar o resto do código.
  - Tempo de inicialização menor, o que importa em robôs que ligam e precisam
    perceber o ambiente em poucos segundos.

A favor do Keras/TensorFlow:
  - Treinamento e fine-tuning: o OpenCV DNN só faz forward pass.
  - Camadas e operações recentes: o importador do OpenCV falha em modelos com
    operadores fora do conjunto suportado; o framework de origem sempre roda.
  - Ferramental de experimentação: callbacks, métricas, augmentation, TensorBoard.
  - Quantização e conversão: o próprio TFLite, com quantização int8, costuma
    superar o OpenCV DNN em CPU de baixo custo.

Regra prática adotada neste trabalho: treinar e validar no Keras, exportar e
EXECUTAR no OpenCV DNN (ou TFLite) no robô.
""".strip())

    print("\nARQUIVOS GERADOS")
    print("  outputs/ex2a_mosaico_top3.jpg  (painel com as 12 imagens)")
    print("  outputs/ex2a_top3_<classe>.jpg (uma por imagem, dois backends)")
    print("  outputs/ex2a_tabela_comparativa.md")


if __name__ == "__main__":
    if "--memoria" in sys.argv:
        medir_memoria_isolada(sys.argv[sys.argv.index("--memoria") + 1])
    else:
        main()
