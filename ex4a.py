"""
EXERCÍCIO 4 - ITEM A: SEGMENTAÇÃO SEMÂNTICA vs. SEGMENTAÇÃO POR COR
(Competência 4.4)

Objetivo: implementar segmentação semântica com modelo pré-treinado, processar
pelo menos cinco cenas externas gerando mapa de classes colorido e máscara
semitransparente, imprimir a porcentagem de área por classe, comparar lado a
lado com a segmentação por cor HSV do TP1 e discutir vantagens e limitações de
cada abordagem para um veículo autônomo.

MODELOS: DeepLabV3-ResNet50 e FCN-ResNet50 (torchvision), ambos citados no
enunciado. Rodam os dois para que a comparação não dependa de uma única
arquitetura, e porque a divergência entre eles na mesma imagem é, por si, uma
medida de confiabilidade.

A LIMITAÇÃO CENTRAL, DECLARADA DESDE O INÍCIO
Os pesos disponíveis no torchvision para esses dois modelos foram treinados no
vocabulário do PASCAL VOC: 20 classes de objeto mais fundo. Esse vocabulário
NÃO CONTÉM pista, calçada, faixa de pedestre, poste, prédio nem vegetação. Em
uma cena de rua, portanto, o asfalto e a calçada são necessariamente rotulados
como "fundo" — não por erro do modelo, mas porque a classe não existe para ele.
Esse é o achado técnico mais importante do exercício, e o script o mede: a
fração de pixels classificados como fundo em cada cena é a medida direta de
quanto do problema de percepção de via fica fora do alcance desses pesos.
Um veículo autônomo real usa modelos treinados em Cityscapes, BDD100K ou Mapillary,
cujos vocabulários incluem road, sidewalk, lane marking, pole e traffic sign.

Execução:
    .venv/Scripts/python.exe ex4a.py
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import (cabecalho, imread_u, imwrite_u, medir_ms,
                    memoria_processo_mb, tabela, tabela_markdown)

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / "modelos" / "torch"))

ROOT = Path(__file__).resolve().parent
CENAS = ROOT / "data" / "cenas"
OUT = ROOT / "outputs"

# Paleta fixa por classe do VOC. Cores fixas (e não aleatórias) são obrigatórias
# em percepção: a mesma classe precisa ter a mesma cor em toda a análise, senão
# a comparação visual entre imagens e entre modelos fica impossível.
CORES_VOC = {
    "__background__": (60, 60, 60),
    "person": (60, 60, 220),
    "car": (220, 140, 40),
    "bus": (40, 200, 220),
    "bicycle": (200, 60, 200),
    "motorbike": (120, 60, 220),
    "train": (160, 160, 60),
    "aeroplane": (200, 200, 120),
    "boat": (180, 120, 60),
    "cat": (60, 200, 120),
    "dog": (120, 200, 60),
    "horse": (60, 160, 160),
    "sheep": (160, 200, 200),
    "cow": (140, 100, 60),
    "bird": (220, 220, 60),
    "bottle": (100, 100, 200),
    "chair": (200, 100, 100),
    "diningtable": (140, 140, 100),
    "pottedplant": (60, 220, 60),
    "sofa": (180, 60, 120),
    "tvmonitor": (100, 180, 220),
}

# HSV do asfalto. Cinza é, por definição, baixa saturação; a matiz é irrelevante
# (qualquer H serve) e o brilho varia com sol e sombra. Os limites vieram de
# medição nos próprios frames: via S ~ 15 a 20 e V ~ 136 a 200; grama S ~ 197
# (excluída pela saturação); calçada S ~ 55 e V ~ 167, isto é, DENTRO da faixa.
# A calçada entrar junto com a via não é ajuste malfeito: é a informação que
# falta à cor. Asfalto e concreto são ambos acinzentados, e distinguir "onde
# posso trafegar" de "onde o pedestre anda" exige justamente o tipo de
# conhecimento semântico que a rede traria, se tivesse essas classes.
HSV_ASFALTO_BAIXO = np.array([0, 0, 90])
HSV_ASFALTO_ALTO = np.array([180, 70, 235])


def carregar_modelos():
    """Carrega DeepLabV3-ResNet50 e FCN-ResNet50 já em modo de avaliação."""
    import torch
    from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                                 FCN_ResNet50_Weights,
                                                 deeplabv3_resnet50,
                                                 fcn_resnet50)
    torch.set_num_threads(os.cpu_count() or 4)

    pesos_dl = DeepLabV3_ResNet50_Weights.DEFAULT
    pesos_fcn = FCN_ResNet50_Weights.DEFAULT
    modelos = {
        "DeepLabV3-ResNet50": (deeplabv3_resnet50(weights=pesos_dl).eval(),
                               pesos_dl),
        "FCN-ResNet50": (fcn_resnet50(weights=pesos_fcn).eval(), pesos_fcn),
    }
    classes = pesos_dl.meta["categories"]
    return modelos, classes


def preparar_tensor(img_bgr, lado_maior=520):
    """
    BGR do OpenCV -> tensor normalizado para a rede.

    A normalização usa média e desvio do ImageNet porque o backbone ResNet50 foi
    pré-treinado lá; alimentar a rede com pixels em 0-1 sem normalizar desloca a
    distribuição de entrada e degrada a segmentação.

    Reduzimos o lado maior para 520 px, que é a resolução de avaliação usada no
    treino desses pesos no torchvision. Manter a resolução nativa de uma foto
    grande aumenta o custo sem ganho, e mudar muito a escala dos objetos piora
    o resultado.
    """
    import torch

    h, w = img_bgr.shape[:2]
    escala = lado_maior / max(h, w)
    if escala < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * escala), int(h * escala)),
                             interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    media = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    desvio = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - media) / desvio
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, img_bgr


def segmentar(modelo, tensor):
    """Executa a rede e devolve o mapa de classes (argmax por pixel)."""
    import torch

    with torch.no_grad():
        saida = modelo(tensor)["out"][0]
    return saida.argmax(0).byte().cpu().numpy()


def colorir(mapa, classes):
    """Mapa de ids -> imagem BGR com a paleta fixa."""
    cor = np.zeros((*mapa.shape, 3), np.uint8)
    for idc in np.unique(mapa):
        nome = classes[idc] if idc < len(classes) else f"classe_{idc}"
        cor[mapa == idc] = CORES_VOC.get(nome, (200, 200, 200))
    return cor


def sobrepor(img, mapa, classes, alpha=0.55):
    """Máscara semitransparente sobre a imagem original."""
    colorida = colorir(mapa, classes)
    return cv2.addWeighted(colorida, alpha, img, 1 - alpha, 0)


def area_por_classe(mapa, classes, minimo=0.1):
    """Porcentagem de área ocupada por cada classe detectada."""
    ids, contagens = np.unique(mapa, return_counts=True)
    total = mapa.size
    resultado = []
    for idc, n in zip(ids, contagens):
        pct = 100.0 * n / total
        if pct >= minimo:
            nome = classes[idc] if idc < len(classes) else f"classe_{idc}"
            resultado.append((nome, pct))
    return sorted(resultado, key=lambda kv: -kv[1])


def segmentar_hsv(img):
    """
    Segmentação da via por cor, no estilo do TP1: limiar em HSV + morfologia +
    maior componente conexa.

    A busca é pelo asfalto, que é o alvo mais útil para um veículo: saber onde
    dá para trafegar. O asfalto não tem cor própria, apenas baixa saturação, e
    é aí que a abordagem por cor começa a falhar.
    """
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_ASFALTO_BAIXO, HSV_ASFALTO_ALTO)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Mantém só a maior região conexa: a via é contínua, manchas isoladas de
    # cinza (telhado, sombra, parede) não são.
    n, etiquetas, estatisticas, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        maior = 1 + int(np.argmax(estatisticas[1:, cv2.CC_STAT_AREA]))
        mask = np.where(etiquetas == maior, 255, 0).astype(np.uint8)

    ms = (time.perf_counter() - t0) * 1000
    return mask, ms


def painel_comparativo(img, overlay_dl, overlay_fcn, mask_hsv, nome):
    """Painel 2x2: original, DeepLabV3, FCN e HSV."""
    h, w = img.shape[:2]

    # Overlay da máscara HSV com a mesma transparência usada na semântica,
    # para que a comparação visual seja entre segmentações e não entre estilos
    # de desenho. Feito com np.where para funcionar também com máscara vazia.
    camada = np.full_like(img, (40, 200, 255))
    misturado = cv2.addWeighted(camada, 0.45, img, 0.55, 0)
    hsv_vis = np.where(mask_hsv[:, :, None] > 0, misturado, img)

    def rotular(im, texto, cor=(0, 0, 0)):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (w, 30), (255, 255, 255), -1)
        cv2.putText(im, texto, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        cv2.rectangle(im, (0, 0), (w - 1, h - 1), (0, 0, 0), 2)
        return im

    linha1 = np.hstack([rotular(img, f"original: {nome}"),
                        rotular(overlay_dl, "DeepLabV3-ResNet50 (VOC)")])
    linha2 = np.hstack([rotular(overlay_fcn, "FCN-ResNet50 (VOC)"),
                        rotular(hsv_vis, "HSV do TP1: via por cor")])
    return np.vstack([linha1, linha2])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cenas = sorted(CENAS.glob("*.png"))
    if not cenas:
        print("Sem imagens. Execute: python preparar_dados_ex4.py")
        sys.exit(1)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch não instalado. Rode:")
        print("  .venv/Scripts/python.exe -m pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    cabecalho("1) MODELOS DE SEGMENTAÇÃO SEMÂNTICA")
    mem_antes = memoria_processo_mb()
    modelos, classes = carregar_modelos()
    mem_depois = memoria_processo_mb()
    print(f"classes do vocabulário: {len(classes)}")
    print("classes:", ", ".join(classes))
    print(f"\nmemória do processo: {mem_antes:.0f} MB -> {mem_depois:.0f} MB "
          f"(+{mem_depois-mem_antes:.0f} MB com os dois modelos carregados)")
    print("\nAusentes do vocabulário e essenciais para um veículo autônomo:")
    print("  pista, calçada, faixa de pedestre, poste, prédio, vegetação,")
    print("  semáforo (como região), placa de trânsito.")

    # ------------------------------------------------------------------
    cabecalho("2) SEGMENTAÇÃO DAS CENAS")
    linhas_tempo, linhas_area, linhas_fundo = [], [], []
    paineis = []

    for caminho in cenas:
        img_original = imread_u(caminho)
        if img_original is None:
            continue
        nome = caminho.stem.replace("cena_", "")
        tensor, img = preparar_tensor(img_original)

        mapas, tempos = {}, {}
        for nome_modelo, (modelo, _) in modelos.items():
            t0 = time.perf_counter()
            mapas[nome_modelo] = segmentar(modelo, tensor)
            tempos[nome_modelo] = (time.perf_counter() - t0) * 1000

        mask_hsv, ms_hsv = segmentar_hsv(img)

        print(f"\n--- {nome}  ({img.shape[1]}x{img.shape[0]})")
        for nome_modelo, mapa in mapas.items():
            areas = area_por_classe(mapa, classes)
            texto = ", ".join(f"{c} {p:.1f}%" for c, p in areas)
            print(f"  {nome_modelo:20s} {tempos[nome_modelo]:7.0f} ms  {texto}")
            fundo = next((p for c, p in areas if c == "__background__"), 0.0)
            linhas_area.append([nome, nome_modelo, texto])
            if nome_modelo == "DeepLabV3-ResNet50":
                linhas_fundo.append([nome, f"{fundo:.1f}",
                                     f"{100-fundo:.1f}",
                                     f"{100*np.count_nonzero(mask_hsv)/mask_hsv.size:.1f}"])

        pct_hsv = 100.0 * np.count_nonzero(mask_hsv) / mask_hsv.size
        print(f"  {'HSV (via por cor)':20s} {ms_hsv:7.1f} ms  "
              f"região selecionada: {pct_hsv:.1f}% da imagem")

        linhas_tempo.append([nome, f"{tempos['DeepLabV3-ResNet50']:.0f}",
                             f"{tempos['FCN-ResNet50']:.0f}", f"{ms_hsv:.1f}",
                             f"{tempos['DeepLabV3-ResNet50']/max(ms_hsv,1e-9):.0f}x"])

        # Concordância entre os dois modelos: fração de pixels com o mesmo
        # rótulo. Alta concordância não garante acerto, mas divergência alta é
        # sinal claro de cena fora do domínio de treino.
        concord = float(np.mean(mapas["DeepLabV3-ResNet50"] ==
                                mapas["FCN-ResNet50"])) * 100
        print(f"  concordância DeepLabV3 vs FCN: {concord:.1f}% dos pixels")

        overlay_dl = sobrepor(img, mapas["DeepLabV3-ResNet50"], classes)
        overlay_fcn = sobrepor(img, mapas["FCN-ResNet50"], classes)
        imwrite_u(OUT / f"ex4a_overlay_{nome}.png", overlay_dl)
        imwrite_u(OUT / f"ex4a_mapa_{nome}.png",
                  colorir(mapas["DeepLabV3-ResNet50"], classes))

        painel = painel_comparativo(img, overlay_dl, overlay_fcn, mask_hsv, nome)
        imwrite_u(OUT / f"ex4a_painel_{nome}.png", painel)
        paineis.append(nome)

    # ------------------------------------------------------------------
    cabecalho("3) CUSTO COMPUTACIONAL")
    print(tabela(["cena", "DeepLabV3 (ms)", "FCN (ms)", "HSV (ms)",
                  "quanto a rede é mais lenta"], linhas_tempo))

    cabecalho("4) QUANTO DA CENA O VOCABULÁRIO NÃO COBRE")
    print(tabela(["cena", "pixels como 'fundo' (%)",
                  "pixels com classe útil (%)", "via pelo HSV (%)"],
                 linhas_fundo))
    media_fundo = np.mean([float(l[1]) for l in linhas_fundo])
    print(f"\nMédia de pixels rotulados como fundo: {media_fundo:.1f}%.")
    print("Em cena de rua, quase tudo que importa para dirigir (asfalto,")
    print("calçada, faixa, guia, prédio) cai nesse 'fundo'. A segmentação")
    print("semântica não está errada: o vocabulário é que é de objetos, não de")
    print("cena urbana. É exatamente por isso que veículos autônomos usam")
    print("modelos treinados em Cityscapes, BDD100K ou Mapillary.")

    # ------------------------------------------------------------------
    cabecalho("5) DISCUSSÃO: COR (TP1) vs. SEGMENTAÇÃO SEMÂNTICA")
    print("""
SEGMENTAÇÃO POR COR (HSV, TP1)
  Vantagens
    - Custo: uma conversão de espaço de cor, uma limiarização e morfologia.
      Medimos abaixo de 5 ms por imagem em CPU, contra centenas de ms da rede;
      roda em microcontrolador, sem acelerador e sem framework.
    - Determinística e auditável: o resultado é explicado por seis números
      (limites de H, S e V). Quando erra, sabe-se exatamente por quê e o ajuste
      é imediato.
    - Latência constante, independente do conteúdo da cena — propriedade
      valiosa em sistema de tempo real com prazo rígido.
  Limitações
    - Não tem noção de objeto: seleciona pixels, não coisas. Asfalto, telhado
      de zinco, sombra na calçada e carro cinza caem na mesma faixa.
    - Frágil a iluminação: a faixa calibrada ao meio-dia falha ao anoitecer, na
      chuva e contra o sol.
    - Não generaliza: cada cenário novo exige recalibrar limiares à mão.

SEGMENTAÇÃO SEMÂNTICA (DeepLabV3 / FCN)
  Vantagens
    - Decide por contexto e forma, não por cor: um pedestre de roupa escura
      contra asfalto escuro continua sendo separado, o que a cor não faz.
    - Fronteira densa por pixel, e não caixa: dá para estimar área livre
      navegável, e não apenas "há um obstáculo em algum lugar desta caixa".
    - Transferível: trocar o vocabulário é trocar o conjunto de treino, sem
      reescrever o algoritmo.
  Limitações
    - Custo: duas a três ordens de grandeza acima do HSV nesta medição, com uso
      de memória de centenas de MB. Em 5 W, exige quantização, poda,
      arquitetura leve ou NPU dedicada.
    - Vocabulário fechado: só enxerga o que existe no conjunto de treino, como
      este exercício mostra numericamente com a fração de "fundo".
    - Falha silenciosa fora do domínio: entrega uma máscara plausível mesmo
      quando erra, sem sinalizar baixa confiança, o que é perigoso em cena
      noturna, com neve ou com objeto inédito na via.
    - Latência variável com a carga do sistema, o que complica garantia de prazo.

PARA UM VEÍCULO AUTÔNOMO, QUAL USAR
Não é escolha exclusiva, e sim divisão de papéis por criticidade e prazo:
  - segmentação semântica (ou panóptica) como camada principal de compreensão
    de cena, rodando em acelerador, definindo área navegável, calçada, faixa e
    agentes;
  - técnicas clássicas de cor e morfologia como camada rápida e verificável
    para alvos de cor normatizada, onde a cor é a própria informação: cone de
    obra, faixa amarela, luz de freio, estado do semáforo;
  - detecção por caixas (Exercício 3) para agentes que exigem rastreamento e
    previsão de trajetória, porque caixa com ID é mais barata de associar no
    tempo que máscara por pixel;
  - e, acima de tudo, redundância: o classificador de cor serve de sanidade
    para a rede. Divergência forte entre as duas camadas é um sinal utilizável
    para reduzir velocidade ou pedir intervenção, em vez de confiar em uma
    máscara que pode estar plausivelmente errada.
""".strip())

    # ------------------------------------------------------------------
    (OUT / "ex4a_tabelas.md").write_text(
        "# Exercício 4A — segmentação semântica vs. HSV\n\n"
        "## Custo por cena\n\n"
        + tabela_markdown(["cena", "DeepLabV3 (ms)", "FCN (ms)", "HSV (ms)",
                           "razão rede/HSV"], linhas_tempo)
        + "\n\n## Cobertura do vocabulário\n\n"
        + tabela_markdown(["cena", "fundo (%)", "classe útil (%)",
                           "via pelo HSV (%)"], linhas_fundo)
        + "\n\n## Área por classe (DeepLabV3 e FCN)\n\n"
        + tabela_markdown(["cena", "modelo", "áreas"], linhas_area)
        + "\n", encoding="utf-8")

    cabecalho("ARQUIVOS GERADOS")
    for nome in paineis:
        print(f"  outputs/ex4a_painel_{nome}.png   (painel 2x2 comparativo)")
    print("  outputs/ex4a_overlay_<cena>.png  (máscara semitransparente)")
    print("  outputs/ex4a_mapa_<cena>.png     (mapa de classes colorido)")
    print("  outputs/ex4a_tabelas.md")


if __name__ == "__main__":
    main()
