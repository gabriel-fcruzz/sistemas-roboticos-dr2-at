"""
EXERCÍCIO 1 - ITEM B: REALIDADE AUMENTADA SOBRE O TABULEIRO
(Competências 1.2, 1.3 e 4.1)

Objetivo: usando a câmera calibrada no item A, estimar a pose do tabuleiro a
cada frame com cv2.solvePnP, projetar um cubo virtual de aresta igual a 1
quadrado sobre o canto de origem com cv2.projectPoints, desenhar as arestas com
cores distintas por face e exibir os vetores de rotação e translação estimados.

PIPELINE POR FRAME (é o mesmo de um robô fazendo estimativa de pose visual)
    frame -> detectar padrão -> refinar cantos (subpixel)
          -> solvePnP  : pose do padrão em relação à câmera (rvec, tvec)
          -> projectPoints : pontos 3D virtuais -> pixels
          -> desenho   : cubo com faces coloridas
A diferença entre isto e um sistema de realidade aumentada comercial, ou entre
isto e a localização de um AGV por marcadores no piso, é apenas o padrão usado
(tabuleiro, ArUco, AprilTag) e o filtro temporal aplicado depois.

O QUE rvec E tvec SIGNIFICAM
    tvec : posição da origem do padrão no sistema de coordenadas da câmera, em
           metros. tvec[2] é a profundidade: a distância câmera-padrão.
    rvec : vetor de Rodrigues. Sua direção é o eixo de rotação e sua norma é o
           ângulo em radianos. cv2.Rodrigues converte para matriz 3x3.
Juntos formam a transformação rígida que leva o referencial do padrão ao
referencial da câmera. Em robótica é a mesma estrutura de dado de uma pose de
odometria, e é ela que alimenta um filtro de Kalman ou um grafo de SLAM.

MODOS DE EXECUÇÃO
    .venv/Scripts/python.exe ex1b.py              # vídeo sintético (padrão)
    .venv/Scripts/python.exe ex1b.py --webcam     # webcam real, se houver
    MOSTRAR_JANELAS=1 .venv/Scripts/python.exe ex1b.py   # abre a janela do OpenCV

No modo sintético conhecemos a pose verdadeira de cada frame, o que permite
medir o ERRO de pose em mm e em graus, e não apenas afirmar que o cubo parece
estável. Essa é a validação quantitativa da exigência "cubo estável ao mover a
câmera".
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common import Cronometro, imwrite_u, finish_video, tabela, video_writer_u
from camera_virtual import (
    IMAGE_SIZE,
    SQUARE_SIZE_M,
    find_refined_corners,
    generate_view,
    load_calibration,
    object_points,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
N_FRAMES = 90
FPS_VIDEO = 15.0

# Cubo de aresta igual a UM quadrado do tabuleiro (30 mm), apoiado no canto de
# origem do padrão. O eixo Z do padrão aponta para dentro do plano, por isso a
# face superior usa -s: o cubo cresce "para fora", na direção da câmera.
S = SQUARE_SIZE_M
CUBO_3D = np.float32([
    [0, 0, 0], [S, 0, 0], [S, S, 0], [0, S, 0],          # base, sobre o tabuleiro
    [0, 0, -S], [S, 0, -S], [S, S, -S], [0, S, -S],      # topo
])

# Uma cor por face, como pede o enunciado (BGR).
FACES = [
    ([0, 1, 2, 3], (0, 200, 0),   "base"),
    ([4, 5, 6, 7], (255, 80, 0),  "topo"),
    ([0, 1, 5, 4], (0, 220, 220), "frente"),
    ([1, 2, 6, 5], (0, 0, 230),   "direita"),
    ([2, 3, 7, 6], (230, 230, 0), "tras"),
    ([3, 0, 4, 7], (230, 0, 230), "esquerda"),
]

EIXOS_3D = np.float32([[0, 0, 0], [2 * S, 0, 0], [0, 2 * S, 0], [0, 0, -2 * S]])


def desenhar_cubo(frame, pts):
    """Preenche as faces com transparência e traça as arestas na cor da face."""
    overlay = frame.copy()
    # Desenha de trás para frente usando a profundidade média em y na imagem:
    # aproximação suficiente para um cubo pequeno e evita que a face de trás
    # apareça sobre a da frente.
    ordem = sorted(FACES, key=lambda f: -np.mean([pts[i][1] for i in f[0]]))
    for ids, cor, _ in ordem:
        poly = np.array([pts[i] for i in ids], np.int32)
        cv2.fillConvexPoly(overlay, poly, cor)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

    for ids, cor, _ in FACES:
        poly = np.array([pts[i] for i in ids], np.int32)
        cv2.polylines(frame, [poly], True, cor, 3, cv2.LINE_AA)
    for i in range(8):
        cv2.circle(frame, tuple(pts[i]), 4, (20, 20, 20), -1)
    return frame


def desenhar_eixos(frame, pts_eixos):
    """Eixos do padrão: X vermelho, Y verde, Z azul (convenção usual)."""
    o = tuple(pts_eixos[0])
    for i, cor in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.arrowedLine(frame, o, tuple(pts_eixos[i]), cor, 3, cv2.LINE_AA,
                        tipLength=0.2)
    return frame


def inset_zoom(frame, pts, escala=3, lado=220):
    """Insere no canto superior direito um recorte ampliado da região do cubo."""
    h, w = frame.shape[:2]
    cx, cy = np.mean(pts, axis=0).astype(int)
    # Meia-janela proporcional ao tamanho aparente do cubo, com folga de 70%,
    # para o recorte acompanhar a aproximação e o afastamento da câmera.
    span = max(pts[:, 0].max() - pts[:, 0].min(),
               pts[:, 1].max() - pts[:, 1].min())
    meio = int(max(24, span * 0.85))
    x0 = int(np.clip(cx - meio, 0, w - 2 * meio - 1))
    y0 = int(np.clip(cy - meio, 0, h - 2 * meio - 1))
    recorte = frame[y0:y0 + 2 * meio, x0:x0 + 2 * meio]
    if recorte.size == 0:
        return frame

    amp = cv2.resize(recorte, (lado, lado), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(amp, (0, 0), (lado - 1, lado - 1), (0, 0, 0), 2)
    fator = lado / max(2 * meio, 1)
    cv2.putText(amp, f"zoom {fator:.1f}x", (8, lado - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    frame[10:10 + lado, w - lado - 10:w - 10] = amp
    # Retângulo indicando de onde veio o recorte.
    cv2.rectangle(frame, (x0, y0), (x0 + 2 * meio, y0 + 2 * meio),
                  (0, 0, 0), 2)
    return frame


def angulo_entre_rotacoes(rvec_a, rvec_b):
    """Diferença angular entre duas rotações, em graus."""
    Ra, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64))
    Rb, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64))
    R = Ra.T @ Rb
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pose_sintetica(i, n=N_FRAMES):
    """
    Trajetória de câmera do vídeo sintético (herdada do material de aula):
    o padrão oscila em inclinação e distância, simulando a câmera se movendo.
    """
    a = i / float(n - 1)
    # Inclinações maiores que as do exemplo de aula: com o padrão quase
    # frontal, a face superior do cubo se sobrepõe à base e o cubo é visto como
    # um quadrado. Em vista oblíqua as três faces visíveis aparecem, o que
    # comprova que a pose 3D está correta e não apenas uma homografia 2D.
    return dict(
        rx=26 + 10 * np.sin(a * np.pi * 2),
        ry=-28 + 16 * np.sin(a * np.pi),
        rz=8 * np.sin(a * np.pi * 2),
        tx=-0.09 + 0.02 * np.sin(a * np.pi * 2),
        ty=-0.075,
        tz=0.42 + 0.06 * np.sin(a * np.pi * 2),
    )


def processar_frame(frame, K, dist, obj):
    """
    Detecta o padrão, estima a pose e desenha o cubo.

    Devolve (frame_anotado, rvec, tvec, tempos_ms) ou (frame, None, None, tempos).
    """
    tempos = {}
    with Cronometro() as c:
        ok, corners = find_refined_corners(frame)
    tempos["deteccao"] = c.ms

    if not ok:
        cv2.putText(frame, "padrao nao detectado", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        tempos["solvepnp"] = 0.0
        tempos["projecao"] = 0.0
        return frame, None, None, tempos

    with Cronometro() as c:
        # SOLVEPNP_ITERATIVE (padrão) parte de uma solução por homografia e
        # refina minimizando o erro de reprojeção. Para padrão planar é a
        # escolha usual; ITERATIVE é estável quando todos os pontos são coplanares.
        sucesso, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
    tempos["solvepnp"] = c.ms

    if not sucesso:
        tempos["projecao"] = 0.0
        return frame, None, None, tempos

    with Cronometro() as c:
        pts_cubo, _ = cv2.projectPoints(CUBO_3D, rvec, tvec, K, dist)
        pts_eixos, _ = cv2.projectPoints(EIXOS_3D, rvec, tvec, K, dist)
    tempos["projecao"] = c.ms

    p = pts_cubo.reshape(-1, 2).astype(int)
    e = pts_eixos.reshape(-1, 2).astype(int)

    # Eixos primeiro, cubo depois: o cubo é o elemento principal e deve ficar
    # visualmente por cima das setas dos eixos.
    desenhar_eixos(frame, e)
    desenhar_cubo(frame, p)
    # Com aresta de um único quadrado (30 mm), o cubo ocupa poucas dezenas de
    # pixels. O recorte ampliado serve de evidência visual de que as faces e as
    # arestas estão corretas, sem alterar a geometria exigida pelo enunciado.
    inset_zoom(frame, p)
    return frame, rvec, tvec, tempos


def anotar_hud(frame, rvec, tvec, fps, extra=None):
    """Painel de texto com a pose estimada, no próprio frame."""
    cv2.rectangle(frame, (10, 10), (560, 150), (255, 255, 255), -1)
    cv2.rectangle(frame, (10, 10), (560, 150), (0, 0, 0), 2)
    if rvec is None:
        cv2.putText(frame, "pose indisponivel", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        return frame
    r = rvec.ravel()
    t = tvec.ravel()
    linhas = [
        f"rvec [rad] = [{r[0]:+.3f} {r[1]:+.3f} {r[2]:+.3f}]",
        f"tvec [m]   = [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]",
        f"distancia  = {t[2]*100:.1f} cm     FPS = {fps:.1f}",
    ]
    if extra:
        linhas.append(extra)
    for j, txt in enumerate(linhas):
        cv2.putText(frame, txt, (25, 45 + 30 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
    return frame


def rodar_sintetico(K, dist, mostrar):
    obj = object_points()
    writer, tmp, final = video_writer_u(OUT / "ex1b_ar_video.mp4",
                                        cv2.VideoWriter_fourcc(*"mp4v"),
                                        FPS_VIDEO, IMAGE_SIZE)

    erros_t, erros_r, erros_z, tempos_todos, fps_todos = [], [], [], [], []
    detectados = 0
    frames_salvos = []

    print(f"Processando {N_FRAMES} frames sintéticos...\n")
    for i in range(N_FRAMES):
        pose = pose_sintetica(i)
        frame, rvec_real, tvec_real = generate_view(**pose)

        t0 = time.perf_counter()
        frame, rvec, tvec, tempos = processar_frame(frame, K, dist, obj)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        fps_todos.append(fps)
        tempos_todos.append(tempos)

        extra = None
        if rvec is not None:
            detectados += 1
            err_t = float(np.linalg.norm(tvec.ravel() - tvec_real.ravel()))
            err_z = float(abs(tvec.ravel()[2] - tvec_real.ravel()[2]))
            err_r = angulo_entre_rotacoes(rvec_real, rvec)
            erros_t.append(err_t)
            erros_z.append(err_z)
            erros_r.append(err_r)
            extra = f"erro pose: {err_t*1000:.2f} mm / {err_r:.3f} graus"

            # Exigência do enunciado: exibir rotação e translação a cada frame.
            r, t = rvec.ravel(), tvec.ravel()
            print(f"frame {i:03d} | rvec=[{r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}] "
                  f"tvec=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m | "
                  f"erro {err_t*1000:6.2f} mm  {err_r:5.3f} deg")
        else:
            print(f"frame {i:03d} | padrão não detectado")

        anotar_hud(frame, rvec, tvec, fps, extra)
        writer.write(frame)

        if i in (0, N_FRAMES // 3, 2 * N_FRAMES // 3, N_FRAMES - 1):
            nome = OUT / f"ex1b_frame{i:03d}.png"
            imwrite_u(nome, frame)
            frames_salvos.append(nome.name)

        if mostrar:
            cv2.imshow("ex1b - realidade aumentada", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    finish_video(writer, tmp, final)
    if mostrar:
        cv2.destroyAllWindows()

    return dict(detectados=detectados, erros_t=erros_t, erros_r=erros_r,
                erros_z=erros_z, tempos=tempos_todos, fps=fps_todos,
                frames=frames_salvos, video=final)


def rodar_webcam(K, dist, mostrar):
    """
    Modo com câmera real. Só faz sentido com uma calibração DA PRÓPRIA câmera:
    reaproveitar o K da câmera virtual em outra câmera introduz erro sistemático
    de escala e de ponto principal. Mantido para demonstrar que o pipeline é o
    mesmo, trocando generate_view() por cap.read().
    """
    obj = object_points()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam não disponível.")
        return None

    print("ESC encerra. Aviso: K/dist vêm da câmera virtual; para medição real,")
    print("recalibre com fotos da sua própria câmera (ver ex1a.py).\n")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        frame, rvec, tvec, _ = processar_frame(frame, K, dist, obj)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        anotar_hud(frame, rvec, tvec, fps)
        if rvec is not None and i % 5 == 0:
            r, t = rvec.ravel(), tvec.ravel()
            print(f"rvec=[{r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}] "
                  f"tvec=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m")
        cv2.imshow("ex1b - webcam", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    npz = ROOT / "camera_calibrada.npz"
    if not npz.exists():
        print("Calibração não encontrada. Execute primeiro: python ex1a.py")
        sys.exit(1)

    K, dist, size, rms = load_calibration(npz)
    print("=" * 68)
    print("CALIBRAÇÃO CARREGADA DO ITEM A")
    print("=" * 68)
    print(np.array2string(K, precision=3, suppress_small=True))
    print(f"dist = {np.array2string(dist.ravel()[:5], precision=6)}")
    print(f"RMS do calibrateCamera = {rms:.4f} px")
    print(f"aresta do cubo virtual = {S*1000:.0f} mm (1 quadrado)\n")

    mostrar = os.environ.get("MOSTRAR_JANELAS", "0") == "1"

    if "--webcam" in sys.argv:
        rodar_webcam(K, dist, mostrar)
        return

    res = rodar_sintetico(K, dist, mostrar)

    print("\n" + "=" * 68)
    print("RESULTADOS")
    print("=" * 68)
    det = res["detectados"]
    print(f"Padrão detectado em {det}/{N_FRAMES} frames "
          f"({det/N_FRAMES*100:.1f}%)")

    if res["erros_t"]:
        et = np.array(res["erros_t"]) * 1000.0     # mm
        ez = np.array(res["erros_z"]) * 1000.0     # mm
        er = np.array(res["erros_r"])              # graus
        print(tabela(
            ["métrica", "média", "mediana", "máximo", "desvio"],
            [["erro de translação (mm)", f"{et.mean():.3f}", f"{np.median(et):.3f}",
              f"{et.max():.3f}", f"{et.std():.3f}"],
             ["erro de profundidade Z (mm)", f"{ez.mean():.3f}", f"{np.median(ez):.3f}",
              f"{ez.max():.3f}", f"{ez.std():.3f}"],
             ["erro de rotação (graus)", f"{er.mean():.4f}", f"{np.median(er):.4f}",
              f"{er.max():.4f}", f"{er.std():.4f}"]],
            titulo="Erro de pose em relação à pose verdadeira do simulador"))
        print("\nO desvio padrão pequeno é a evidência de estabilidade pedida no")
        print("enunciado: o cubo não 'tremula' porque a pose estimada não oscila")
        print("de um frame para o outro.")

    tempos = res["tempos"]
    med = {k: float(np.mean([t[k] for t in tempos])) for k in tempos[0]}
    total = sum(med.values())
    print("\n" + tabela(
        ["etapa", "tempo médio (ms)", "% do frame"],
        [[k, f"{v:.2f}", f"{v/total*100:.1f}"] for k, v in med.items()]
        + [["TOTAL", f"{total:.2f}", "100.0"]],
        titulo="Custo por etapa do pipeline de pose"))
    print(f"\nFPS médio do pipeline (sem a síntese do frame): "
          f"{np.mean(res['fps']):.1f}")
    print("A detecção do padrão domina o custo; solvePnP e projectPoints são")
    print("desprezíveis. Em robótica isso significa que trocar o tabuleiro por")
    print("um marcador mais leve (ArUco/AprilTag) é o caminho para ganhar FPS.")

    print("\nARQUIVOS GERADOS")
    print(f"  outputs/{Path(res['video']).name}  (vídeo com o cubo sobreposto)")
    for nome in res["frames"]:
        print(f"  outputs/{nome}")


if __name__ == "__main__":
    main()
