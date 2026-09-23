"""
EXERCÍCIO 3 - ITEM B: RASTREAMENTO COM ID PERSISTENTE POR IoU
(Competências 3.1, 3.2 e 4.3)

Objetivo: integrar o YOLO do item A com rastreamento por IoU entre frames
consecutivos, atribuindo ID persistente a cada objeto, mantendo a trilha dos
últimos 30 frames, detectando entradas e saídas com contagem cumulativa e
medindo a taxa de ID switches por minuto de vídeo.

POR QUE DETECTAR NÃO BASTA
O detector é sem memória: em cada frame ele devolve um conjunto de caixas sem
qualquer vínculo com o frame anterior. Para um veículo autônomo isso é
insuficiente, porque as decisões dependem de histórico: velocidade e direção de
um pedestre, tempo até a colisão, se aquele ciclista é o mesmo de meio segundo
atrás. Rastrear é justamente manter a identidade ao longo do tempo.

COMO A ASSOCIAÇÃO É FEITA
A cada frame, calculamos a IoU entre cada trilha ativa e cada nova detecção, e
casamos os pares de maior IoU acima de um limiar (associação gulosa). É o núcleo
do algoritmo SORT, sem o filtro de Kalman:

    IoU alta  -> mesmo objeto, a trilha recebe a nova caixa
    sem par   -> detecção nova vira trilha candidata
    trilha sem detecção -> conta falha; após MAX_FALHAS, é encerrada

Duas salvaguardas contra ruído do detector:
  - MIN_ACERTOS: uma trilha só é confirmada (e contada) após aparecer em alguns
    frames seguidos, o que evita contar falso positivo isolado;
  - MAX_FALHAS: a trilha sobrevive a alguns frames sem detecção, o que evita
    perder o ID quando o objeto passa atrás de um poste.

LIMITAÇÃO CONHECIDA DA ASSOCIAÇÃO SÓ POR IoU
Sem modelo de movimento, a associação depende de as caixas se sobreporem entre
frames consecutivos. Objeto rápido, vídeo com poucos FPS ou oclusão longa
quebram essa premissa, e o ID troca. É o que o filtro de Kalman do TP3 resolve:
prevendo onde o objeto deveria estar, a associação passa a comparar a detecção
com a PREVISÃO, não com a última posição observada.

Execução:
    .venv/Scripts/python.exe ex3b.py
    .venv/Scripts/python.exe ex3b.py --frames 200
"""

import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from common import (cabecalho, finish_video, imwrite_u, tabela,
                    video_capture_u, video_writer_u)
from detectores import CONF_MIN, NMS_LIMIAR, DetectorYOLO, desenhar_deteccoes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
VIDEO = ROOT.parent / "TP3" / "data" / "vtest.avi"

FPS_VIDEO = 15.0          # taxa do vtest.avi, usada para converter frames em tempo
N_FRAMES = 795            # o vídeo inteiro
IOU_MIN = 0.30            # limiar de associação entre trilha e detecção
MAX_FALHAS = 8            # frames sem detecção antes de encerrar a trilha
MIN_ACERTOS = 3           # frames para confirmar uma trilha nova
TRILHA_MAX = 30           # exigência do enunciado: trilha dos últimos 30 frames
CLASSES_RASTREADAS = {"person"}   # foco em pedestres, o caso de uso discutido
FRAMES_SALVOS = (60, 200, 400, 600)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def centro(caixa):
    x, y, w, h = caixa
    return (int(x + w / 2), int(y + h / 2))


def cor_do_id(tid):
    """Cor estável por ID, para a trilha e a caixa não mudarem de cor."""
    rng = np.random.RandomState(tid * 7919)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


class Trilha:
    def __init__(self, tid, caixa, classe, frame_idx):
        self.id = tid
        self.caixa = caixa
        self.classe = classe
        self.acertos = 1
        self.falhas = 0
        self.nascimento = frame_idx
        self.ultimo_frame = frame_idx
        self.confirmada = False
        self.historico = deque([centro(caixa)], maxlen=TRILHA_MAX)
        self.lado_x = None          # lado em relação à linha vertical
        self.lado_y = None          # lado em relação à linha horizontal

    def atualizar(self, caixa, frame_idx):
        self.caixa = caixa
        self.acertos += 1
        self.falhas = 0
        self.ultimo_frame = frame_idx
        self.historico.append(centro(caixa))
        if self.acertos >= MIN_ACERTOS:
            self.confirmada = True


class RastreadorIoU:
    """
    Rastreador por IoU com ID persistente, entradas/saídas e ID switches.

    Mantém também o registro das trilhas recém-encerradas, usado na estimativa
    de ID switches: se uma trilha morre e, poucos frames depois, nasce outra
    praticamente no mesmo lugar, o mais provável é que seja o MESMO objeto que
    perdeu a identidade (fragmentação de trilha), e não um objeto novo.
    """

    def __init__(self):
        self.trilhas = []
        self.proximo_id = 1
        self.entradas = 0            # trilhas confirmadas criadas
        self.saidas = 0              # trilhas confirmadas encerradas
        self.id_switches = 0
        self.cemiterio = deque(maxlen=64)   # (caixa, frame, id)
        self.ids_confirmados = set()

    def _nova_trilha(self, caixa, classe, frame_idx):
        # Antes de criar, verifica se esta detecção coincide com uma trilha
        # encerrada há pouco: nesse caso, é reaparecimento com ID novo.
        for c_caixa, c_frame, c_id in self.cemiterio:
            if frame_idx - c_frame <= MAX_FALHAS + 4 and iou(caixa, c_caixa) >= IOU_MIN:
                self.id_switches += 1
                break

        t = Trilha(self.proximo_id, caixa, classe, frame_idx)
        self.proximo_id += 1
        self.trilhas.append(t)
        return t

    def atualizar(self, caixas, classes, frame_idx):
        # Considera apenas as classes de interesse.
        dets = [(c, cl) for c, cl in zip(caixas, classes)
                if cl in CLASSES_RASTREADAS]

        # --- associação gulosa por IoU -----------------------------------
        pares = []
        for i, t in enumerate(self.trilhas):
            for j, (caixa, classe) in enumerate(dets):
                if t.classe != classe:
                    continue
                v = iou(t.caixa, caixa)
                if v >= IOU_MIN:
                    pares.append((v, i, j))
        pares.sort(reverse=True)

        trilhas_usadas, dets_usadas = set(), set()
        for v, i, j in pares:
            if i in trilhas_usadas or j in dets_usadas:
                continue
            self.trilhas[i].atualizar(dets[j][0], frame_idx)
            trilhas_usadas.add(i)
            dets_usadas.add(j)

        # --- detecções sem par viram trilhas novas -----------------------
        for j, (caixa, classe) in enumerate(dets):
            if j not in dets_usadas:
                self._nova_trilha(caixa, classe, frame_idx)

        # --- trilhas sem par acumulam falhas e podem morrer --------------
        vivas = []
        for i, t in enumerate(self.trilhas):
            if i not in trilhas_usadas and t.ultimo_frame != frame_idx:
                t.falhas += 1
            if t.falhas > MAX_FALHAS:
                if t.confirmada:
                    self.saidas += 1
                self.cemiterio.append((t.caixa, frame_idx, t.id))
            else:
                vivas.append(t)
        self.trilhas = vivas

        # Contagem cumulativa: cada ID confirmado é contado uma única vez.
        for t in self.trilhas:
            if t.confirmada and t.id not in self.ids_confirmados:
                self.ids_confirmados.add(t.id)
                self.entradas += 1

        return [t for t in self.trilhas if t.confirmada]


def desenhar_trilhas(frame, trilhas, linha_x=None, linha_y=None, info=None):
    out = frame.copy()

    # Duas linhas virtuais, porque o fluxo desta cena não é unidirecional:
    # parte dos pedestres atravessa o quadro na horizontal (passeio superior) e
    # parte desce pela via à direita. Em contagem de fluxo real, a posição das
    # linhas é definida pela geometria da via, não pelo centro da imagem.
    if linha_x is not None:
        cv2.line(out, (linha_x, 0), (linha_x, out.shape[0]), (0, 255, 255), 2)
        cv2.putText(out, "linha V", (linha_x + 6, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if linha_y is not None:
        cv2.line(out, (0, linha_y), (out.shape[1], linha_y), (255, 200, 0), 2)
        cv2.putText(out, "linha H", (out.shape[1] - 90, linha_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    for t in trilhas:
        cor = cor_do_id(t.id)
        x, y, w, h = t.caixa
        cv2.rectangle(out, (x, y), (x + w, y + h), cor, 2)
        etiqueta = f"ID {t.id}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x, max(0, y - th - 6)), (x + tw + 6, y), cor, -1)
        cv2.putText(out, etiqueta, (x + 3, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Trilha dos últimos 30 frames, na cor do ID.
        pontos = list(t.historico)
        for k in range(1, len(pontos)):
            espessura = 1 + int(2 * k / max(len(pontos), 1))
            cv2.line(out, pontos[k - 1], pontos[k], cor, espessura)
        if pontos:
            cv2.circle(out, pontos[-1], 3, cor, -1)

    if info:
        alt = 22 * (len(info) + 1)
        cv2.rectangle(out, (0, 0), (430, alt), (0, 0, 0), -1)
        for k, texto in enumerate(info, start=1):
            cv2.putText(out, texto, (10, 20 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_frames = N_FRAMES
    if "--frames" in sys.argv:
        n_frames = int(sys.argv[sys.argv.index("--frames") + 1])

    detector = DetectorYOLO()
    rastreador = RastreadorIoU()

    cap, _tmp = video_capture_u(VIDEO)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    linha_x = int(largura * 0.80)     # via à direita, onde há descida
    linha_y = int(altura * 0.33)      # passeio superior, fluxo horizontal

    writer, tmp_video, destino = video_writer_u(
        OUT / "ex3b_rastreamento.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_VIDEO, (largura, altura))

    cabecalho("CONFIGURAÇÃO")
    print(f"vídeo                  : {VIDEO.name} ({largura}x{altura}, "
          f"{FPS_VIDEO:.0f} fps)")
    print(f"frames processados     : {n_frames}")
    print(f"detector               : {detector.nome} "
          f"(conf >= {CONF_MIN}, NMS = {NMS_LIMIAR})")
    print(f"classe rastreada       : {', '.join(CLASSES_RASTREADAS)}")
    print(f"linhas de contagem     : vertical em x={linha_x}, "
          f"horizontal em y={linha_y}")
    print(f"limiar de IoU          : {IOU_MIN}")
    print(f"trilha                 : últimos {TRILHA_MAX} frames")
    print(f"confirmação / paciência: {MIN_ACERTOS} acertos / {MAX_FALHAS} falhas")

    cruz_dir = cruz_esq = cruz_baixo = cruz_cima = 0
    ativos_por_frame, tempos = [], []
    frames_salvos = []
    vida_das_trilhas = Counter()

    print("\nprocessando...")
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        t0 = cv2.getTickCount()
        caixas, scores, classes, _ = detector.detectar(frame)
        confirmadas = rastreador.atualizar(caixas, classes, i)

        # Contagem direcional por linha virtual: compara o lado atual do centro
        # com o lado no frame anterior. É a técnica usada em contagem de fluxo
        # (pedestres por hora, veículos por faixa).
        for t in confirmadas:
            cx, cy = centro(t.caixa)

            lado_x = "dir" if cx >= linha_x else "esq"
            if t.lado_x is not None and lado_x != t.lado_x:
                if lado_x == "dir":
                    cruz_dir += 1
                else:
                    cruz_esq += 1
            t.lado_x = lado_x

            lado_y = "baixo" if cy >= linha_y else "cima"
            if t.lado_y is not None and lado_y != t.lado_y:
                if lado_y == "baixo":
                    cruz_baixo += 1
                else:
                    cruz_cima += 1
            t.lado_y = lado_y

            vida_das_trilhas[t.id] += 1

        ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
        tempos.append(ms)
        ativos_por_frame.append(len(confirmadas))

        segundos = (i + 1) / FPS_VIDEO
        minutos = max(segundos / 60.0, 1e-9)
        info = [
            f"frame {i:03d}  |  {ms:.0f} ms  |  {1000/max(ms,1e-6):.1f} FPS",
            f"pedestres no quadro: {len(confirmadas)}",
            f"contagem cumulativa (IDs unicos): {rastreador.entradas}",
            f"entradas: {rastreador.entradas}   saidas: {rastreador.saidas}",
            f"linha V: -> {cruz_dir}  <- {cruz_esq}   "
            f"linha H: v {cruz_baixo}  ^ {cruz_cima}",
            f"ID switches: {rastreador.id_switches} "
            f"({rastreador.id_switches/minutos:.1f}/min)",
        ]
        anotado = desenhar_trilhas(frame, confirmadas, linha_x, linha_y, info)
        writer.write(anotado)

        if i in FRAMES_SALVOS:
            arq = OUT / f"ex3b_frame{i:03d}.png"
            imwrite_u(arq, anotado)
            frames_salvos.append(arq.name)

        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{n_frames} | {len(confirmadas)} ativos | "
                  f"{rastreador.entradas} IDs | "
                  f"{rastreador.id_switches} switches")

    cap.release()
    finish_video(writer, tmp_video, destino)

    # ------------------------------------------------------------------
    frames = len(tempos)
    segundos = frames / FPS_VIDEO
    minutos = segundos / 60.0

    cabecalho("RESULTADOS DO RASTREAMENTO")
    print(tabela(
        ["métrica", "valor"],
        [["frames processados", frames],
         ["duração do vídeo (s)", f"{segundos:.1f}"],
         ["pedestres por frame (média)", f"{np.mean(ativos_por_frame):.2f}"],
         ["pedestres por frame (máximo)", max(ativos_por_frame)],
         ["contagem cumulativa de IDs únicos", rastreador.entradas],
         ["entradas detectadas", rastreador.entradas],
         ["saídas detectadas", rastreador.saidas],
         ["linha vertical: cruzamentos esq->dir", cruz_dir],
         ["linha vertical: cruzamentos dir->esq", cruz_esq],
         ["linha horizontal: cruzamentos para baixo", cruz_baixo],
         ["linha horizontal: cruzamentos para cima", cruz_cima],
         ["total de cruzamentos contados",
          cruz_dir + cruz_esq + cruz_baixo + cruz_cima],
         ["ID switches estimados", rastreador.id_switches],
         ["ID switches por minuto", f"{rastreador.id_switches/minutos:.2f}"],
         ["latência média por frame (ms)", f"{np.mean(tempos):.1f}"],
         ["FPS médio (detecção + rastreamento)",
          f"{1000/np.mean(tempos):.1f}"]]))

    duracoes = np.array(list(vida_das_trilhas.values()))
    if len(duracoes):
        print(f"\nDuração das trilhas confirmadas: média "
              f"{duracoes.mean():.1f} frames ({duracoes.mean()/FPS_VIDEO:.1f} s), "
              f"mediana {np.median(duracoes):.0f}, máximo {duracoes.max()}")
        curtas = int((duracoes < 15).sum())
        print(f"Trilhas com menos de 1 segundo de vida: {curtas} de "
              f"{len(duracoes)} ({curtas/len(duracoes)*100:.0f}%) — "
              f"indicador de fragmentação.")

    # ------------------------------------------------------------------
    cabecalho("COMO A TAXA DE ID SWITCHES FOI ESTIMADA")
    print("""
Não existe anotação de identidade para este vídeo, então não é possível calcular
a métrica formal de ID switches (a do padrão MOT, que exige o ID verdadeiro de
cada pessoa em cada frame). O que medimos é um ESTIMADOR por reaparecimento:
conta-se um switch quando uma trilha é encerrada e, em até poucos frames, nasce
outra com IoU >= 0.30 em relação à última caixa da trilha morta. O caso típico é
a pessoa que passa atrás de um obstáculo e volta como ID novo.

O estimador tem viés conhecido e declarado:
  - SUBESTIMA a troca clássica entre duas pessoas que se cruzam, porque nessa
    situação nenhuma trilha morre: os IDs simplesmente se trocam entre si;
  - SUPERESTIMA quando uma pessoa realmente sai de cena e outra entra pelo mesmo
    ponto pouco depois, o que é comum em corredor ou porta.
Por isso o número deve ser lido como uma taxa de FRAGMENTAÇÃO de trilha, útil
para comparar configurações do próprio rastreador (limiar de IoU, paciência,
com e sem Kalman), e não como acurácia absoluta de identidade.
""".strip())

    # ------------------------------------------------------------------
    cabecalho("APLICAÇÃO: CONTAGEM DE PEDESTRES POR DRONE DE VIGILÂNCIA URBANA")
    print("""
VIABILIDADE TÉCNICA
O pipeline deste item é, em essência, o que um drone de monitoramento urbano
executaria: detector de estágio único, associação temporal leve e contagem por
linha virtual. As métricas medidas aqui dizem o que precisaria mudar:

  - a contagem cumulativa herda todo erro do rastreador. Com a fragmentação
    medida acima, a contagem de pessoas ÚNICAS fica superestimada, porque cada
    perda de identidade cria um "novo" pedestre. Para contagem de fluxo, a
    contagem por cruzamento de linha é mais robusta que a contagem de IDs;
  - câmera em movimento é o caso mais difícil: o drone se desloca, então a
    premissa de sobreposição entre frames consecutivos enfraquece. A correção
    usual é compensar o movimento da câmera (homografia entre frames, como no
    TP2) antes de associar, ou usar filtro de Kalman com modelo de velocidade;
  - vista de cima muda a aparência do pedestre. Detector treinado em COCO, com
    fotos em nível do solo, degrada em vista aérea; seria necessário treinar ou
    ajustar com imagens da própria altura de voo;
  - iluminação, chuva e sombra alteram a taxa de detecção ao longo do dia, o que
    exige recalibrar limiares por período, não fixá-los uma vez.

CONSIDERAÇÕES ÉTICAS
Contar pessoas e identificar pessoas são coisas diferentes, e a fronteira entre
as duas é uma decisão de projeto, não uma consequência inevitável da tecnologia.

  1. Finalidade e proporcionalidade. Contagem agregada de fluxo para dimensionar
     calçada, semáforo ou transporte público é finalidade legítima. A mesma
     câmera, com reconhecimento facial acoplado, passa a ser vigilância
     individual. A LGPD exige finalidade específica e informada, e minimização:
     coletar o mínimo necessário para aquela finalidade.
  2. Minimização de dados por construção. O sistema pode ser projetado para
     nunca gravar imagem: processa o frame em memória, incrementa contadores e
     descarta o quadro. O que persiste é "12 pedestres cruzaram no sentido
     leste entre 14h e 15h", não o rosto de ninguém. Onde a imagem for
     necessária para auditoria, o caminho é anonimizar em borda (desfoque
     irreversível de rosto e placa) ANTES de qualquer armazenamento.
  3. Não identificação e não reidentificação. O ID deste rastreador vale para
     um único trecho de vídeo e não deve ser reutilizado para religar a mesma
     pessoa em dias diferentes ou entre câmeras: esse cruzamento transforma
     contagem em rastreio de rotina individual.
  4. Viés e erro desigual. Detectores treinados em bases desbalanceadas erram
     mais para pele escura, crianças, cadeirantes e pessoas com bagagem ou
     carrinho. Em contagem, isso vira sub-representação sistemática de grupos no
     dado que orienta política pública. Auditar a taxa de detecção por subgrupo
     é parte do trabalho de engenharia, não um extra.
  5. Transparência e consentimento situacional. Em espaço público não há
     consentimento individual viável, o que aumenta o dever de sinalização
     clara, publicação da finalidade, do prazo de retenção e do responsável,
     além de canal para contestação.
  6. Uso secundário e controle de acesso. O risco maior costuma não estar no
     algoritmo, mas no depósito de dados: um histórico de trajetórias coletado
     para "otimizar o trânsito" é atraente para fins policiais, comerciais ou
     de perseguição. Retenção curta, agregação na origem e registro de acesso
     são as barreiras técnicas contra o desvio de finalidade.
  7. Assimetria de poder e efeito inibidor. Vigilância aérea persistente muda o
     comportamento de quem circula, inclusive o direito de manifestação. A
     decisão de sobrevoar um bairro não é só técnica: exige participação de quem
     mora ali.
""".strip())

    cabecalho("ARQUIVOS GERADOS")
    print(f"  outputs/{Path(destino).name}  (vídeo de demonstração)")
    for f in frames_salvos:
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
