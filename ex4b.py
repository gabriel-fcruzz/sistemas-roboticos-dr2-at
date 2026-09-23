"""
EXERCÍCIO 4 - ITEM B: APOIO AO RELATÓRIO INTEGRATIVO
(Competência 4.4 e todas as anteriores)

Este script não escreve o relatório: ele produz os artefatos verificáveis que o
relatório usa e, ao final, confere se o arquivo entregue cumpre os requisitos do
enunciado.

  1. outputs/ex4b_diagrama_pipeline.png
     Diagrama do pipeline completo (calibração -> pré-processamento -> detecção
     clássica -> detecção profunda -> rastreamento -> segmentação), exigido na
     seção 1 do relatório.

  2. outputs/ex4b_metricas.md e outputs/ex4b_metricas.json
     Tabela consolidada de todas as técnicas com as métricas REAIS medidas nos
     TPs e neste AT, exigida na seção 2.

  3. outputs/ex4b_grafico_custo.png
     Custo por técnica em escala logarítmica, que sustenta a análise de 5 W.

  4. Verificação final do relatorio_integrativo.md: contagem de palavras,
     presença das cinco seções obrigatórias e quantas técnicas com dado real
     aparecem citadas.

PROCEDÊNCIA DE CADA NÚMERO
Toda métrica abaixo tem origem declarada no campo "fonte": ou foi medida pelos
scripts deste AT (e está nos arquivos outputs/*_terminal.txt), ou veio da
execução dos TPs anteriores. Nada aqui é estimativa de catálogo do fabricante.

Execução:
    .venv/Scripts/python.exe ex4b.py
"""

import json
import re
from pathlib import Path

import numpy as np

from common import cabecalho, tabela, tabela_markdown

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
RELATORIO = ROOT / "relatorio_integrativo.md"

# ---------------------------------------------------------------------------
# Métricas reais. "ms" é por imagem/frame, salvo indicação em contrário.
# ---------------------------------------------------------------------------
METRICAS = [
    dict(etapa="Pré-processamento", tecnica="Segmentação HSV + morfologia",
         ms=2.28, memoria_mb=None, qualidade="81% das regiões candidatas descartadas por filtros geométricos",
         complexidade="O(n) por pixel, sem treino",
         fonte="AT ex2b.ipynb (média de 30 frames)"),
    dict(etapa="Pré-processamento", tecnica="Segmentação HSV da via (asfalto)",
         ms=3.4, memoria_mb=None, qualidade="30,8% da imagem marcada como navegável; não separa via de calçada",
         complexidade="O(n) por pixel, sem treino",
         fonte="AT ex4a.py (cena campus_via)"),
    dict(etapa="Calibração", tecnica="calibrateCamera (24 imagens)",
         ms=None, memoria_mb=None, qualidade="erro de reprojeção 0,0202 px; erro de fx 0,078%",
         complexidade="offline, uma vez por câmera",
         fonte="AT ex1a.py"),
    dict(etapa="Calibração", tecnica="undistort por frame",
         ms=6.45, memoria_mb=None, qualidade="corrige deslocamento radial de até 27 px na borda",
         complexidade="O(n) por pixel, mapas pré-calculáveis",
         fonte="AT ex2b.ipynb"),
    dict(etapa="Pose", tecnica="solvePnP + projectPoints (tabuleiro)",
         ms=0.27, memoria_mb=None, qualidade="erro de translação 0,31 mm; rotação 0,028°",
         complexidade="iterativo sobre 42 pontos",
         fonte="AT ex1b.py"),
    dict(etapa="Características", tecnica="ORB (1000 keypoints)",
         ms=5.75, memoria_mb=None, qualidade="descritor binário de 32 bytes",
         complexidade="FAST + BRIEF, sem treino",
         fonte="TP2 ex3a.py"),
    dict(etapa="Características", tecnica="AKAZE",
         ms=29.4, memoria_mb=None, qualidade="2666 keypoints, descritor de 61 bytes",
         complexidade="difusão não linear",
         fonte="TP2 ex3a.py"),
    dict(etapa="Características", tecnica="SIFT",
         ms=52.4, memoria_mb=None, qualidade="2419 keypoints, descritor de 128 floats",
         complexidade="pirâmide DoG",
         fonte="TP2 ex3a.py"),
    dict(etapa="Detecção clássica", tecnica="HOG+SVM pré-treinado (rápido)",
         ms=24.9, memoria_mb=None, qualidade="1,39 detecção/frame",
         complexidade="janela deslizante, 1 escala grossa",
         fonte="TP3 ex1a.py"),
    dict(etapa="Detecção clássica", tecnica="HOG+SVM pré-treinado (preciso)",
         ms=202.3, memoria_mb=None, qualidade="4,22 detecções/frame",
         complexidade="janela deslizante, pirâmide fina",
         fonte="TP3 ex1a.py"),
    dict(etapa="Detecção clássica", tecnica="HOG+SVM treinado por nós",
         ms=5800.0, memoria_mb=None, qualidade="precisão 0,89 e F1 0,64 com hard negatives",
         complexidade="treino próprio + janela deslizante",
         fonte="TP3 ex1b.py"),
    dict(etapa="Detecção clássica", tecnica="Haar Cascade (rostos)",
         ms=None, memoria_mb=None, qualidade="0 rostos em cena de vigilância a 40 m",
         complexidade="cascata de classificadores fracos",
         fonte="AT ex2b.ipynb"),
    dict(etapa="Subtração de fundo", tecnica="MOG2",
         ms=4.48, memoria_mb=None, qualidade="4,28 objetos/frame (223 FPS)",
         complexidade="modelo de mistura por pixel",
         fonte="TP3 ex2a.py"),
    dict(etapa="Subtração de fundo", tecnica="KNN",
         ms=3.66, memoria_mb=None, qualidade="4,39 objetos/frame (273 FPS)",
         complexidade="vizinhos por pixel",
         fonte="TP3 ex2a.py"),
    dict(etapa="Classificação", tecnica="MLP no MNIST",
         ms=None, memoria_mb=None, qualidade="98,00% de acurácia, 535.818 parâmetros",
         complexidade="2 camadas densas",
         fonte="TP3 ex3a.py"),
    dict(etapa="Classificação", tecnica="CNN no MNIST",
         ms=None, memoria_mb=None, qualidade="99,41% de acurácia, 421.642 parâmetros",
         complexidade="2 blocos convolucionais",
         fonte="TP3 ex3a.py"),
    dict(etapa="Classificação", tecnica="MobileNetV2 fine-tuned (gênero)",
         ms=14.63, memoria_mb=11.6, qualidade="90,67% de acurácia em 150 rostos",
         complexidade="transfer learning, 10 épocas",
         fonte="TP3 ex4b.py"),
    dict(etapa="Classificação", tecnica="Caffe gender_net",
         ms=8.19, memoria_mb=45.7, qualidade="88,00% de acurácia",
         complexidade="rede pronta, sem treino",
         fonte="TP3 ex4b.py"),
    dict(etapa="Classificação", tecnica="MobileNetV2 via OpenCV DNN (TFLite)",
         ms=7.06, memoria_mb=81.8, qualidade="91,7% top-1 e 100% top-3 em 12 imagens",
         complexidade="3.538.984 parâmetros",
         fonte="AT ex2a.py"),
    dict(etapa="Classificação", tecnica="MobileNetV2 via Keras/TensorFlow",
         ms=13.87, memoria_mb=437.0, qualidade="mesma acurácia do OpenCV DNN",
         complexidade="mesma rede, outro runtime",
         fonte="AT ex2a.py"),
    dict(etapa="Detecção profunda", tecnica="YOLOv4-tiny (416x416)",
         ms=40.6, memoria_mb=23.1, qualidade="24,6 FPS; confiança média 80,4%; 6,98 objetos/frame",
         complexidade="6.062.826 parâmetros",
         fonte="AT ex3a.py (300 frames)"),
    dict(etapa="Detecção profunda", tecnica="YOLOv4-tiny (320x320)",
         ms=31.3, memoria_mb=23.1, qualidade="31,9 FPS; 5,22 objetos/frame",
         complexidade="mesma rede, entrada menor",
         fonte="AT ex3a.py (varredura)"),
    dict(etapa="Detecção profunda", tecnica="SSD MobileNetV2 (300x300)",
         ms=32.2, memoria_mb=66.6, qualidade="31,1 FPS; confiança média 72,3%; 7,74 objetos/frame",
         complexidade="16.876.287 parâmetros",
         fonte="AT ex3a.py (300 frames)"),
    dict(etapa="Rastreamento", tecnica="IoU + ID persistente (sobre YOLO)",
         ms=36.3, memoria_mb=None, qualidade="27,6 FPS; 98 IDs; 10,19 ID switches/min",
         complexidade="associação gulosa O(t*d)",
         fonte="AT ex3b.py (795 frames)"),
    dict(etapa="Rastreamento", tecnica="CamShift",
         ms=None, memoria_mb=None, qualidade="perdeu o alvo no frame 275",
         complexidade="deslocamento de média sobre histograma",
         fonte="TP3 ex2a.py"),
    dict(etapa="Segmentação", tecnica="DeepLabV3-ResNet50 (VOC)",
         ms=951.0, memoria_mb=538.0, qualidade="93,9% dos pixels caem em 'fundo' nas cenas externas",
         complexidade="ResNet50 + ASPP",
         fonte="AT ex4a.py (7 cenas)"),
    dict(etapa="Segmentação", tecnica="FCN-ResNet50 (VOC)",
         ms=777.0, memoria_mb=538.0, qualidade="99,4% de concordância com o DeepLabV3",
         complexidade="ResNet50 + cabeça FCN",
         fonte="AT ex4a.py (7 cenas)"),
]

SECOES_OBRIGATORIAS = [
    "1. Diagrama do pipeline completo",
    "2. Tabela comparativa das técnicas",
    "3. Viabilidade em hardware embarcado",
    "4. Arquitetura de percepção",
    "5. Três lacunas",
]


def gerar_diagrama():
    """Diagrama de blocos do pipeline completo de percepção."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    etapas = [
        ("1. CALIBRAÇÃO", "calibrateCamera\nundistort\nsolvePnP",
         "0,02 px de erro\n6,5 ms/frame", "#2f6fb3"),
        ("2. PRÉ-PROCESSAMENTO", "HSV + morfologia\nROI por cor",
         "2,3 ms/frame", "#3d8b5f"),
        ("3. CARACTERÍSTICAS", "ORB\n(FAST + BRIEF)",
         "5,8 ms\n32 B/ponto", "#7a5aa8"),
        ("4. DETECÇÃO CLÁSSICA", "HOG+SVM\nHaar Cascade",
         "24,9 a 202 ms", "#b3792f"),
        ("5. DETECÇÃO PROFUNDA", "YOLOv4-tiny\nSSD MobileNetV2",
         "24,6 e 31,1 FPS", "#b33b3b"),
        ("6. RASTREAMENTO", "associação por IoU\nID persistente",
         "27,6 FPS\n10,2 switches/min", "#2f8f9b"),
        ("7. SEGMENTAÇÃO", "DeepLabV3 / FCN\nResNet50",
         "951 ms/imagem", "#6b6b6b"),
    ]

    fig, ax = plt.subplots(figsize=(15, 5.2))
    largura, altura, espaco = 1.75, 2.0, 0.42
    for i, (titulo, conteudo, metrica, cor) in enumerate(etapas):
        x = i * (largura + espaco)
        caixa = FancyBboxPatch((x, 1.0), largura, altura,
                               boxstyle="round,pad=0.06", linewidth=2,
                               edgecolor=cor, facecolor=cor + "22")
        ax.add_patch(caixa)
        ax.text(x + largura / 2, 2.72, titulo, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=cor)
        ax.text(x + largura / 2, 2.15, conteudo, ha="center", va="center",
                fontsize=8.2)
        ax.text(x + largura / 2, 1.28, metrica, ha="center", va="center",
                fontsize=7.6, style="italic", color="#333333")
        if i:
            ax.add_patch(FancyArrowPatch(
                (x - espaco + 0.04, 2.0), (x - 0.04, 2.0),
                arrowstyle="-|>", mutation_scale=16, linewidth=1.6,
                color="#555555"))

    largura_total = len(etapas) * (largura + espaco) - espaco
    ax.text(largura_total / 2, 0.55,
            "Entrada: frame da câmera        "
            "Saída: agentes com ID e trajetória + mapa de área navegável",
            ha="center", fontsize=9, color="#222222")
    ax.text(largura_total / 2, 0.15,
            "métricas medidas em CPU, nos scripts deste AT e nos TPs da disciplina",
            ha="center", fontsize=7.5, style="italic", color="#666666")

    ax.set_xlim(-0.3, largura_total + 0.3)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    fig.tight_layout()
    destino = OUT / "ex4b_diagrama_pipeline.png"
    fig.savefig(destino, dpi=150, facecolor="white")
    plt.close(fig)
    return destino


def gerar_grafico_custo():
    """Custo por técnica, em escala logarítmica, com a fronteira de 5 W."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dados = [(m["tecnica"], m["ms"]) for m in METRICAS if m["ms"]]
    dados.sort(key=lambda kv: kv[1])
    nomes = [d[0] for d in dados]
    valores = [d[1] for d in dados]

    # Em uma placa de 5 W (Raspberry Pi 4, Jetson Nano em modo 5 W), a CPU
    # costuma ser de 5 a 10 vezes mais lenta que este desktop. A faixa
    # sombreada marca o que, multiplicado por 8, ainda caberia em 100 ms.
    fig, ax = plt.subplots(figsize=(11, 8))
    cores = ["#3d8b5f" if v * 8 <= 100 else
             "#b3792f" if v * 8 <= 1000 else "#b33b3b" for v in valores]
    barras = ax.barh(nomes, valores, color=cores)
    ax.bar_label(barras, fmt="%.1f ms", padding=3, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("tempo por imagem ou frame (ms, escala logarítmica)")
    ax.set_title("Custo medido por técnica\n"
                 "verde: caberia em 100 ms numa placa 8x mais lenta   "
                 "laranja: entre 100 ms e 1 s   vermelho: acima de 1 s",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3, which="both")
    fig.tight_layout()
    destino = OUT / "ex4b_grafico_custo.png"
    fig.savefig(destino, dpi=140, facecolor="white")
    plt.close(fig)
    return destino


def gerar_tabela_metricas():
    cab = ["etapa", "técnica", "tempo (ms)", "memória/disco (MB)",
           "qualidade medida", "complexidade", "fonte"]
    linhas = [[m["etapa"], m["tecnica"],
               f"{m['ms']:.2f}" if m["ms"] else "-",
               f"{m['memoria_mb']:.1f}" if m["memoria_mb"] else "-",
               m["qualidade"], m["complexidade"], m["fonte"]]
              for m in METRICAS]

    (OUT / "ex4b_metricas.md").write_text(
        "# Métricas consolidadas da disciplina\n\n"
        "Todos os valores foram medidos nesta máquina (CPU), pelos scripts "
        "indicados na coluna fonte.\n\n"
        + tabela_markdown(cab, linhas) + "\n", encoding="utf-8")
    (OUT / "ex4b_metricas.json").write_text(
        json.dumps(METRICAS, ensure_ascii=False, indent=2), encoding="utf-8")
    return cab, linhas


def verificar_relatorio():
    """Checklist do enunciado sobre o relatório entregue."""
    if not RELATORIO.exists():
        print(f"[FALTA] {RELATORIO.name} ainda não existe.")
        return False

    texto = RELATORIO.read_text(encoding="utf-8")
    palavras = len(re.findall(r"\b[\wÀ-ÿ]+\b", texto))

    print(f"arquivo            : {RELATORIO.name}")
    print(f"palavras           : {palavras} "
          f"({'OK' if palavras >= 800 else 'ABAIXO DO MÍNIMO DE 800'})")

    faltando = [s for s in SECOES_OBRIGATORIAS
                if s.lower() not in texto.lower()]
    if faltando:
        print("seções ausentes    : " + "; ".join(faltando))
    else:
        print("seções obrigatórias: 5 de 5 presentes")

    citadas = [m["tecnica"] for m in METRICAS
               if m["tecnica"].split("(")[0].strip().lower() in texto.lower()]
    print(f"técnicas com dado real citadas: {len(citadas)} "
          f"({'OK, mínimo é 4' if len(citadas) >= 4 else 'ABAIXO DO MÍNIMO'})")

    numeros = len(re.findall(r"\d+[,.]\d+", texto))
    print(f"valores numéricos no texto     : {numeros}")

    return palavras >= 800 and not faltando and len(citadas) >= 4


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    cabecalho("1) DIAGRAMA DO PIPELINE")
    print("gerado:", gerar_diagrama().relative_to(ROOT))

    cabecalho("2) TABELA CONSOLIDADA DE MÉTRICAS")
    cab, linhas = gerar_tabela_metricas()
    print(tabela(cab[:4] + [cab[6]],
                 [[l[0], l[1], l[2], l[3], l[6]] for l in linhas]))
    print(f"\n{len(METRICAS)} técnicas catalogadas; "
          f"detalhes em outputs/ex4b_metricas.md")

    cabecalho("3) CUSTO POR TÉCNICA")
    print("gerado:", gerar_grafico_custo().relative_to(ROOT))
    rapidas = [m for m in METRICAS if m["ms"] and m["ms"] * 8 <= 100]
    lentas = [m for m in METRICAS if m["ms"] and m["ms"] * 8 > 1000]
    print(f"\ntécnicas que caberiam em 100 ms numa placa 8x mais lenta: "
          f"{len(rapidas)}")
    print("  " + "; ".join(m["tecnica"] for m in rapidas))
    print(f"técnicas que passariam de 1 s nessa mesma placa: {len(lentas)}")
    print("  " + "; ".join(m["tecnica"] for m in lentas))

    cabecalho("4) VERIFICAÇÃO DO RELATÓRIO INTEGRATIVO")
    ok = verificar_relatorio()
    print("\nstatus:", "aprovado no checklist" if ok else "pendências acima")


if __name__ == "__main__":
    main()
