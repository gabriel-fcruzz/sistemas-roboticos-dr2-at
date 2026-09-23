"""
ETAPAS DO PIPELINE INTEGRADO DO EXERCÍCIO 2B

Módulo de apoio do notebook ex2b.ipynb. Cada função é uma etapa do pipeline e
devolve, além do resultado, o tempo gasto em milissegundos. Manter as etapas
aqui permite que o notebook fique legível e que o mesmo código seja reaproveitado
na medição sobre vários frames, sem duplicação.

Encadeamento (uma técnica de cada TP da disciplina):
    1. undistort .................. Exercício 1 do AT (calibração)
    2. segmentação HSV + morfologia ................... TP1
    3. descritores ORB ................................ TP2
    4. detector HOG+SVM de pessoas e Haar de rostos ... TP3
    5. classificação da ROI com MobileNetV2 no OpenCV DNN ... Exercício 2A
"""

import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Etapa 1 - calibração
# ---------------------------------------------------------------------------

def reescalar_K(K, tamanho_origem, tamanho_destino):
    """
    Ajusta a matriz intrínseca para outra resolução de imagem.

    fx, fy, cx e cy são medidos em pixels, então mudam proporcionalmente quando
    a imagem é redimensionada. Os coeficientes de distorção são adimensionais
    (definidos em coordenadas normalizadas) e não mudam. Esse ajuste é rotina em
    robótica: calibra-se em alta resolução e processa-se em resolução reduzida
    para ganhar FPS.
    """
    sx = tamanho_destino[0] / tamanho_origem[0]
    sy = tamanho_destino[1] / tamanho_origem[1]
    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def mapas_distorcao(K, dist, tamanho):
    """Mapas para SIMULAR a distorção de lente em um frame já retificado."""
    w, h = tamanho
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    grid = np.stack([us.ravel(), vs.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(grid, K, dist).reshape(-1, 2)
    map_x = (ideal[:, 0] * K[0, 0] + K[0, 2]).reshape(h, w).astype(np.float32)
    map_y = (ideal[:, 1] * K[1, 1] + K[1, 2]).reshape(h, w).astype(np.float32)
    return map_x, map_y


def etapa_undistort(frame, K, dist):
    """
    Corrige a distorção de lente do frame.

    Em um robô esta é a primeira etapa depois da captura, e não é opcional: sem
    ela, toda medição de posição feita depois (largura de faixa, distância a um
    obstáculo, ângulo de um alvo) carrega erro sistemático que cresce com a
    distância ao centro da imagem.
    """
    t0 = time.perf_counter()
    corrigido = cv2.undistort(frame, K, dist)
    return corrigido, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Etapa 2 - segmentação por cor (TP1)
# ---------------------------------------------------------------------------

# Cones de sinalização amarelo/laranja. O intervalo é definido em HSV porque
# nesse espaço a cor (H) fica separada do brilho (V): a mesma faixa de matiz
# funciona no sol e na sombra, o que não acontece em RGB. Os limites de S e V
# foram ajustados medindo os pixels dos cones do próprio vídeo (S > 100 e
# V > 150), o que separa o amarelo saturado do verde escuro da grama (H ~ 40,
# V ~ 94) e do tijolo (H ~ 14, V ~ 91).
HSV_CONE_BAIXO = np.array([15, 100, 150])
HSV_CONE_ALTO = np.array([38, 255, 255])


def etapa_hsv(frame, baixo=HSV_CONE_BAIXO, alto=HSV_CONE_ALTO, area_min=25,
              razao_min=0.7, corte_superior=0.20):
    """
    Segmenta por cor em HSV e devolve máscara, caixas e diagnóstico.

    Dois estágios, como no TP1:
      1. limiarização em HSV + morfologia (abertura remove pixels isolados,
         fechamento preenche buracos; kernel 3x3 porque os cones têm poucas
         dezenas de pixels e um kernel maior os apagaria);
      2. filtros geométricos sobre os contornos.

    Os filtros geométricos são necessários porque a cor sozinha não distingue
    objeto de cenário: neste vídeo, o tijolo e os caixilhos das janelas caem na
    mesma faixa de matiz dos cones. Aplicamos dois critérios que um robô real
    também usaria:
      - razão altura/largura mínima, já que o cone é vertical;
      - descartar a faixa superior do quadro, acima da linha do horizonte, onde
        não existe piso e portanto não existe cone.
    O retorno inclui quantas regiões havia ANTES dos filtros, para dimensionar
    o quanto a segmentação por cor depende de conhecimento externo da cena.
    """
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, baixo, alto)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    limite_y = int(frame.shape[0] * corte_superior)

    brutas, caixas = [], []
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            continue
        x, y, w, h = cv2.boundingRect(c)
        brutas.append((x, y, w, h))
        if y < limite_y:
            continue
        if h / max(w, 1) < razao_min:
            continue
        caixas.append((x, y, w, h))

    caixas.sort(key=lambda b: -b[2] * b[3])
    diag = {"regioes_brutas": len(brutas), "regioes_filtradas": len(caixas)}
    return mask, caixas, (time.perf_counter() - t0) * 1000, diag


# ---------------------------------------------------------------------------
# Etapa 3 - descritores locais (TP2)
# ---------------------------------------------------------------------------

def etapa_orb(frame, caixa=None, n_features=300):
    """
    Extrai pontos-chave e descritores ORB, opcionalmente restritos a uma ROI.

    ORB (FAST + BRIEF com orientação) é a escolha prática em robótica embarcada:
    é livre de patente, tem descritor binário de 32 bytes comparável por
    distância de Hamming e roda em CPU modesta. Descritores são a base de
    rastreamento por correspondência, odometria visual e fechamento de laço
    em SLAM.
    """
    t0 = time.perf_counter()
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mascara = None
    if caixa is not None:
        x, y, w, h = caixa
        mascara = np.zeros(cinza.shape, np.uint8)
        mascara[y:y + h, x:x + w] = 255

    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(cinza, mascara)
    return kp, des, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Etapa 4 - detectores clássicos (TP3)
# ---------------------------------------------------------------------------

def contar_kp_em_caixas(kp, caixas):
    """Quantos pontos-chave caem dentro de alguma das caixas informadas."""
    if not kp or not caixas:
        return 0
    total = 0
    for p in kp:
        x, y = p.pt
        for (bx, by, bw, bh) in caixas:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                total += 1
                break
    return total


def criar_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def criar_haar():
    from common import caminho_ascii
    caminho = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(caminho_ascii(caminho))


def etapa_deteccao(frame, hog, haar, conf_min=0.3):
    """
    Detecta pessoas com HOG+SVM e rostos com Haar Cascade.

    Os dois são detectores "clássicos", anteriores às redes profundas: HOG
    descreve a silhueta por histogramas de orientação de gradiente e um SVM
    linear decide; Haar usa cascata de classificadores fracos sobre diferenças
    de intensidade. São rápidos e não exigem GPU, mas sofrem com oclusão,
    escala pequena e pose fora do padrão de treino.
    """
    t0 = time.perf_counter()
    pessoas, pesos = hog.detectMultiScale(frame, winStride=(8, 8),
                                          padding=(8, 8), scale=1.05)
    caixas_pessoas = [(tuple(map(int, r)), float(p))
                      for r, p in zip(pessoas, np.ravel(pesos))
                      if p >= conf_min]

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostos = haar.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=4)
    caixas_rostos = [tuple(map(int, r)) for r in rostos]
    return caixas_pessoas, caixas_rostos, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Etapa 5 - classificação profunda (Exercício 2A)
# ---------------------------------------------------------------------------

def etapa_classificacao(frame, caixa, net, labels, margem=0.12):
    """
    Recorta a ROI, monta o blob e classifica com a MobileNetV2 no OpenCV DNN.

    A margem extra em volta da caixa existe porque classificadores de imagem
    inteira são treinados com o objeto centralizado e com contexto ao redor;
    recortes justos costumam derrubar a confiança.
    """
    t0 = time.perf_counter()
    x, y, w, h = caixa
    mx, my = int(w * margem), int(h * margem)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(frame.shape[1], x + w + mx)
    y1 = min(frame.shape[0], y + h + my)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return [], (time.perf_counter() - t0) * 1000

    blob = cv2.dnn.blobFromImage(roi, 1 / 127.5, (224, 224),
                                 (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    prob = np.asarray(net.forward(), dtype=np.float64).reshape(-1)
    if abs(prob.sum() - 1.0) > 1e-3:      # rede que termina em logits
        e = np.exp(prob - prob.max())
        prob = e / e.sum()

    idx = np.argsort(prob)[::-1][:3]
    top3 = [(labels[i] if i < len(labels) else f"classe_{i}", float(prob[i]))
            for i in idx]
    return top3, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Desenho das anotações
# ---------------------------------------------------------------------------

COR_HSV = (0, 200, 255)
COR_PESSOA = (60, 220, 60)
COR_ROSTO = (255, 120, 0)
COR_ORB = (200, 0, 200)


def anotar(frame, caixas_hsv, kp, caixas_pessoas, caixas_rostos, rotulos):
    """Desenha, sobre o mesmo frame, o resultado de todas as etapas."""
    out = frame.copy()

    if kp:
        # Pontos simples, sem círculo de escala: no frame final há caixas e
        # texto sobrepostos, e o desenho "rico" dos 300 pontos ORB encobriria
        # as demais anotações. A figura dedicada da etapa 3 usa o desenho rico.
        out = cv2.drawKeypoints(out, kp, None, color=COR_ORB, flags=0)

    for i, (x, y, w, h) in enumerate(caixas_hsv):
        cv2.rectangle(out, (x, y), (x + w, y + h), COR_HSV, 2)
        cv2.putText(out, f"HSV{i}", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_HSV, 1)

    for i, ((x, y, w, h), p) in enumerate(caixas_pessoas):
        cv2.rectangle(out, (x, y), (x + w, y + h), COR_PESSOA, 2)
        cv2.putText(out, f"HOG{i} {p:.2f}", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_PESSOA, 1)

    for (x, y, w, h) in caixas_rostos:
        cv2.rectangle(out, (x, y), (x + w, y + h), COR_ROSTO, 2)
        cv2.putText(out, "Haar", (x, max(12, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_ROSTO, 1)

    # Legenda das classificações no rodapé.
    if rotulos:
        alt = 18 * (len(rotulos) + 1)
        base = out.shape[0] - alt - 6
        cv2.rectangle(out, (6, base), (6 + 430, out.shape[0] - 6),
                      (255, 255, 255), -1)
        cv2.rectangle(out, (6, base), (6 + 430, out.shape[0] - 6), (0, 0, 0), 1)
        cv2.putText(out, "MobileNetV2 (OpenCV DNN) nas ROIs:", (12, base + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        for j, txt in enumerate(rotulos, start=1):
            cv2.putText(out, txt, (12, base + 14 + 18 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1)
    return out


def rodar_pipeline(frame_bruto, K, dist, net, labels, hog, haar,
                   max_roi_classificadas=3):
    """
    Executa as cinco etapas em sequência e devolve resultados e tempos.

    Esta é a função usada tanto na demonstração de um frame quanto na medição
    sobre vários frames.
    """
    tempos = {}
    frame, tempos["1_undistort"] = etapa_undistort(frame_bruto, K, dist)
    mask, caixas_hsv, tempos["2_hsv"], diag_hsv = etapa_hsv(frame)
    # ORB no quadro inteiro: as ROIs de cor têm poucas dezenas de pixels e
    # renderiam quase nenhum ponto-chave. No quadro inteiro os descritores
    # cumprem o papel que têm em robótica (odometria visual e correspondência
    # entre frames), e ainda podemos contar quantos caem dentro das ROIs.
    kp, des, tempos["3_orb"] = etapa_orb(frame, None)
    pessoas, rostos, tempos["4_deteccao"] = etapa_deteccao(frame, hog, haar)

    rotulos, t_cls = [], 0.0
    alvos = [(f"HOG{i}", b) for i, (b, _) in enumerate(pessoas)]
    if caixas_hsv:
        alvos.append(("HSV0", caixas_hsv[0]))
    for nome, caixa in alvos[:max_roi_classificadas]:
        top3, dt = etapa_classificacao(frame, caixa, net, labels)
        t_cls += dt
        if top3:
            rotulos.append(f"{nome}: " + ", ".join(
                f"{c} {p*100:.0f}%" for c, p in top3))
    tempos["5_classificacao"] = t_cls

    return dict(frame=frame, mask=mask, caixas_hsv=caixas_hsv, kp=kp, des=des,
                pessoas=pessoas, rostos=rostos, rotulos=rotulos, tempos=tempos,
                diag_hsv=diag_hsv,
                kp_em_pessoas=contar_kp_em_caixas(kp, [b for b, _ in pessoas]),
                kp_em_hsv=contar_kp_em_caixas(kp, caixas_hsv))
