"""
EXERCÍCIO 1 - ITEM A: CALIBRAÇÃO DE CÂMERA (Competências 1.2, 1.3 e 4.1)

Objetivo: obter a matriz intrínseca K, os 5 coeficientes de distorção e os erros
de reprojeção de uma câmera a partir de imagens de um padrão de tabuleiro de
xadrez, e demonstrar a correção da distorção com cv2.undistort.

O que este script faz:
  1. gera 18 vistas do tabuleiro 7x6 em ângulos e distâncias variados
     (o enunciado pede no mínimo 15);
  2. detecta os cantos com cv2.findChessboardCorners e refina em subpixel;
  3. calibra com cv2.calibrateCamera;
  4. imprime K, os coeficientes de distorção e o erro de reprojeção por imagem
     e médio;
  5. compara os valores estimados com os valores verdadeiros da câmera virtual;
  6. aplica cv2.undistort e salva o painel "original vs. corrigida".

POR QUE UMA CÂMERA VIRTUAL
O padrão adotado na disciplina é o de um "laboratório virtual": as imagens são
sintetizadas por uma câmera de parâmetros conhecidos. Isso permite medir o erro
real da calibração (comparando com o valor verdadeiro), o que é impossível com
uma webcam, onde só se conhece o erro de reprojeção. O procedimento de
calibração em si é idêntico ao de uma câmera física: trocar generate_dataset()
por uma pasta de fotos reais não altera nenhuma outra linha.

ERRO DE REPROJEÇÃO: QUAL VALOR É ACEITÁVEL
O erro de reprojeção mede, em pixels, a distância entre o canto detectado na
imagem e o canto reprojetado pelo modelo calibrado. É a métrica padrão de
qualidade da calibração. Referências para uso em robótica:
  < 0.3 px  : excelente; adequado a odometria visual, SLAM e estéreo, onde o
              erro de calibração se propaga e se acumula ao longo da trajetória;
  0.3-0.5 px: bom; suficiente para estimativa de pose, realidade aumentada e
              detecção de faixa;
  0.5-1.0 px: aceitável apenas para tarefas qualitativas (presença/ausência de
              obstáculo, detecção sem medição métrica);
  > 1.0 px  : inaceitável; indica cantos mal detectados, tabuleiro não plano,
              poses pouco variadas ou número insuficiente de imagens.
Um erro de 1 px a 10 m de distância, com fx de 900 px e alvo de 1 m, equivale a
mais de 10 cm de erro de profundidade: é o tipo de viés sistemático que faz um
veículo autônomo frear no lugar errado.

Execução:
    .venv/Scripts/python.exe ex1a.py
Saídas em outputs/.
"""

from pathlib import Path

import cv2
import numpy as np

from common import imread_u, imwrite_u, tabela
from camera_virtual import (
    IMAGE_SIZE,
    PATTERN_SIZE,
    SQUARE_SIZE_M,
    TRUE_DIST,
    TRUE_K,
    calibrate_from_paths,
    ensure_dirs,
    find_refined_corners,
    generate_dataset,
    reprojection_errors,
    save_calibration,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
N_IMAGENS = 24        # 18 poses do material de aula + 6 poses periféricas nossas
N_CENTRAIS = 18       # subconjunto usado na comparação de cobertura


def erro_modelo_distorcao(K_est, dist_est, image_size, passo=16):
    """
    Erro do MODELO de distorção estimado, em pixels.

    O erro de reprojeção não é suficiente para julgar os coeficientes: k1, k2 e
    k3 são fortemente correlacionados e várias combinações diferentes produzem
    praticamente o mesmo deslocamento na região onde o padrão foi observado.
    Comparar coeficiente por coeficiente com o valor verdadeiro, portanto, pode
    sugerir um erro grosseiro onde não há. A métrica honesta é comparar o efeito
    do modelo: para cada pixel do quadro, de quanto os dois modelos (verdadeiro
    e estimado) discordam ao mapear o ponto ideal para o pixel distorcido.
    """
    w, h = image_size
    us, vs = np.meshgrid(np.arange(0, w, passo, dtype=np.float64),
                         np.arange(0, h, passo, dtype=np.float64))

    # Pixels -> pontos normalizados ideais (projeção pinhole inversa).
    x = (us.ravel() - TRUE_K[0, 2]) / TRUE_K[0, 0]
    y = (vs.ravel() - TRUE_K[1, 2]) / TRUE_K[1, 1]
    pts3d = np.stack([x, y, np.ones_like(x)], axis=1).astype(np.float64)

    zero = np.zeros((3, 1))
    p_true, _ = cv2.projectPoints(pts3d, zero, zero, TRUE_K, TRUE_DIST)
    p_est, _ = cv2.projectPoints(pts3d, zero, zero, K_est, dist_est)

    d = np.linalg.norm(p_true.reshape(-1, 2) - p_est.reshape(-1, 2), axis=1)
    raio = np.hypot(us.ravel() - TRUE_K[0, 2], vs.ravel() - TRUE_K[1, 2])
    borda = raio > 0.75 * raio.max()
    return float(np.sqrt(np.mean(d ** 2))), float(d.max()), float(np.mean(d[borda]))


def linha(titulo=""):
    print("\n" + "=" * 68)
    if titulo:
        print(titulo)
        print("=" * 68)


def main():
    ensure_dirs(ROOT)

    linha("1) GERAÇÃO DAS IMAGENS DE CALIBRAÇÃO")
    paths = generate_dataset(ROOT, n=N_IMAGENS)
    print(f"{len(paths)} imagens {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} geradas em data/tabuleiro/")
    print(f"Padrão: {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} cantos internos, "
          f"quadrado de {SQUARE_SIZE_M*1000:.0f} mm")

    # ------------------------------------------------------------------
    # Detecção dos cantos. O refinamento subpixel (cv2.cornerSubPix) é o que
    # leva o erro de reprojeção da casa de 1 px para a casa de 0.1 px: sem ele
    # o canto fica limitado à resolução inteira do pixel.
    # ------------------------------------------------------------------
    linha("2) DETECÇÃO DOS CANTOS (findChessboardCorners + cornerSubPix)")
    detectadas = 0
    for p in paths:
        ok, _ = find_refined_corners(imread_u(p))
        detectadas += int(ok)
        print(f"  {p.name}: {'cantos encontrados' if ok else 'FALHOU'}")
    print(f"\nTabuleiro detectado em {detectadas}/{len(paths)} imagens.")

    linha("3) CALIBRAÇÃO (calibrateCamera)")
    rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, size = \
        calibrate_from_paths(paths)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()[:5]

    print("Matriz intrínseca K (pixels):")
    print(np.array2string(K, precision=4, suppress_small=True))
    print(f"\n  fx = {fx:10.4f} px   distância focal horizontal")
    print(f"  fy = {fy:10.4f} px   distância focal vertical")
    print(f"  cx = {cx:10.4f} px   ponto principal em x")
    print(f"  cy = {cy:10.4f} px   ponto principal em y")
    print(f"  razão de aspecto fy/fx = {fy/fx:.5f} (≈1 indica pixels quadrados)")

    print("\nCoeficientes de distorção [k1, k2, p1, p2, k3]:")
    print(f"  k1 = {k1:12.8f}   radial de 2ª ordem (efeito barril se < 0)")
    print(f"  k2 = {k2:12.8f}   radial de 4ª ordem")
    print(f"  p1 = {p1:12.8f}   tangencial (lente descentrada)")
    print(f"  p2 = {p2:12.8f}   tangencial")
    print(f"  k3 = {k3:12.8f}   radial de 6ª ordem (relevante em grande-angular)")

    # ------------------------------------------------------------------
    linha("4) ERRO DE REPROJEÇÃO")
    errors = reprojection_errors(K, dist, rvecs, tvecs, objpoints, imgpoints)
    for p, e in zip(used, errors):
        print(f"  {p.name}: {e:.4f} px")

    err_medio = float(np.mean(errors))
    pior = int(np.argmax(errors))
    print(f"\n  RMS global do calibrateCamera : {rms:.4f} px")
    print(f"  Erro médio de reprojeção      : {err_medio:.4f} px")
    print(f"  Pior imagem                   : {used[pior].name} ({errors[pior]:.4f} px)")
    print(f"  Melhor imagem                 : {used[int(np.argmin(errors))].name} "
          f"({min(errors):.4f} px)")

    if err_medio < 0.3:
        veredito = "EXCELENTE - adequado a odometria visual, SLAM e estéreo"
    elif err_medio < 0.5:
        veredito = "BOM - adequado a estimativa de pose e realidade aumentada"
    elif err_medio < 1.0:
        veredito = "ACEITÁVEL apenas para tarefas qualitativas"
    else:
        veredito = "INACEITÁVEL - recalibrar com mais imagens e poses variadas"
    print(f"  Classificação para uso robótico: {veredito}")

    # ------------------------------------------------------------------
    # Validação que só o laboratório virtual permite: comparar o estimado com
    # o valor verdadeiro usado para sintetizar as imagens.
    # ------------------------------------------------------------------
    linha("5) ESTIMADO vs. VERDADEIRO (possível por ser câmera virtual)")
    print("K verdadeira:")
    print(np.array2string(TRUE_K, precision=2, suppress_small=True))
    print("\nDiferença K_estimada - K_verdadeira:")
    print(np.array2string(K - TRUE_K, precision=4, suppress_small=True))
    for nome, est, real in [("fx", fx, TRUE_K[0, 0]), ("fy", fy, TRUE_K[1, 1]),
                            ("cx", cx, TRUE_K[0, 2]), ("cy", cy, TRUE_K[1, 2])]:
        print(f"  erro relativo em {nome}: {abs(est-real)/real*100:.4f}%")

    print("\nCoeficientes de distorção (verdadeiro -> estimado):")
    nomes = ["k1", "k2", "p1", "p2", "k3"]
    for nome, real, est in zip(nomes, TRUE_DIST.ravel()[:5], dist.ravel()[:5]):
        print(f"  {nome}: {real:+.6f} -> {est:+.6f}   (erro {abs(est-real):.6f})")

    rms_mod, max_mod, borda_mod = erro_modelo_distorcao(K, dist, size)
    print("\nErro do modelo de distorção (o que de fato importa):")
    print(f"  discordância RMS entre modelo estimado e verdadeiro: {rms_mod:.4f} px")
    print(f"  discordância máxima no quadro                      : {max_mod:.4f} px")
    print(f"  discordância média no terço periférico             : {borda_mod:.4f} px")
    print("  Obs.: k2 e k3 podem divergir muito do valor verdadeiro sem prejuízo,")
    print("        pois são correlacionados; o efeito combinado é o que vale.")

    # ------------------------------------------------------------------
    # Justificativa medida para as 6 poses periféricas: recalibramos usando
    # apenas as 18 poses centrais e comparamos o erro do modelo.
    # ------------------------------------------------------------------
    linha("5b) IMPACTO DA COBERTURA DO CAMPO DE VISÃO")
    rms_c, K_c, dist_c, rvecs_c, tvecs_c, obj_c, img_c, used_c, size_c = \
        calibrate_from_paths(paths[:N_CENTRAIS])
    err_c = float(np.mean(reprojection_errors(K_c, dist_c, rvecs_c, tvecs_c,
                                              obj_c, img_c)))
    rms_mod_c, max_mod_c, borda_mod_c = erro_modelo_distorcao(K_c, dist_c, size_c)

    print(tabela(
        ["conjunto", "imgs", "erro reproj. (px)", "modelo RMS (px)",
         "modelo na borda (px)", "erro fx (%)"],
        [["18 poses centrais", N_CENTRAIS, f"{err_c:.4f}", f"{rms_mod_c:.3f}",
          f"{borda_mod_c:.3f}",
          f"{abs(K_c[0,0]-TRUE_K[0,0])/TRUE_K[0,0]*100:.4f}"],
         ["+ 6 poses periféricas", N_IMAGENS, f"{err_medio:.4f}", f"{rms_mod:.3f}",
          f"{borda_mod:.3f}", f"{abs(fx-TRUE_K[0,0])/TRUE_K[0,0]*100:.4f}"]]))
    print("\nLeitura: o erro de reprojeção mal se altera, mas o erro do modelo de")
    print("distorção na periferia cai de forma expressiva. Em um robô, isso é a")
    print("diferença entre medir bem no centro da imagem e medir bem em toda ela,")
    print("onde normalmente aparecem os obstáculos laterais.")

    # Mapa de cobertura dos cantos detectados sobre o quadro.
    cobertura = np.full((size[1], size[0], 3), 255, np.uint8)
    for pontos in imgpoints:
        for (px, py) in pontos.reshape(-1, 2):
            cv2.circle(cobertura, (int(px), int(py)), 3, (200, 60, 40), -1)
    for pontos in img_c:
        for (px, py) in pontos.reshape(-1, 2):
            cv2.circle(cobertura, (int(px), int(py)), 3, (60, 160, 60), -1)
    cv2.putText(cobertura, "verde: 18 poses centrais   vermelho: poses perifericas",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    imwrite_u(OUT / "ex1a_cobertura_cantos.png", cobertura)

    # ------------------------------------------------------------------
    linha("6) CORREÇÃO DA DISTORÇÃO (undistort) E PAINEL COMPARATIVO")
    idx = 0  # vista frontal: a curvatura das bordas fica mais evidente
    original = imread_u(paths[idx])
    corrigida = cv2.undistort(original, K, dist)

    # Diferença absoluta: evidencia onde a correção atuou (bordas da imagem,
    # onde o deslocamento radial é maior).
    diferenca = cv2.absdiff(original, corrigida)
    desloc_max = float(np.max(cv2.cvtColor(diferenca, cv2.COLOR_BGR2GRAY)))
    print(f"Imagem usada: {paths[idx].name}")
    print(f"Diferença máxima de intensidade entre original e corrigida: {desloc_max:.0f}/255")

    painel = np.hstack([original, corrigida])
    h, w = original.shape[:2]
    cv2.putText(painel, "ORIGINAL (com distorcao)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    cv2.putText(painel, "CORRIGIDA (undistort)", (w + 30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 150, 0), 3)
    cv2.line(painel, (w, 0), (w, h), (0, 0, 0), 3)

    imwrite_u(OUT / "ex1a_painel_original_corrigida.png", painel)
    imwrite_u(OUT / "ex1a_diferenca_undistort.png",
              cv2.applyColorMap(cv2.convertScaleAbs(diferenca, alpha=3.0),
                                cv2.COLORMAP_INFERNO))

    # Gráfico do erro por imagem (evidência para o relatório).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(1, len(errors) + 1), errors, color="#3b7dd8")
        ax.axhline(err_medio, color="#d8443b", linestyle="--",
                   label=f"média = {err_medio:.4f} px")
        ax.axhline(0.5, color="#888", linestyle=":", label="limite robótico 0.5 px")
        ax.set_xlabel("imagem de calibração")
        ax.set_ylabel("erro de reprojeção (px)")
        ax.set_title("Erro de reprojeção por imagem")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "ex1a_erro_reprojecao.png", dpi=120)
        plt.close(fig)
    except ImportError:
        print("matplotlib ausente: gráfico do erro não gerado.")

    save_calibration(ROOT / "camera_calibrada.npz", K, dist, size, rms)

    linha("ARQUIVOS GERADOS")
    for nome in ["ex1a_painel_original_corrigida.png",
                 "ex1a_diferenca_undistort.png",
                 "ex1a_erro_reprojecao.png",
                 "ex1a_cobertura_cantos.png"]:
        print(f"  outputs/{nome}")
    print("  camera_calibrada.npz  (K e dist reutilizados pelo ex1b e ex2b)")

    janelas = __import__("os").environ.get("MOSTRAR_JANELAS", "0") == "1"
    if janelas:
        cv2.imshow("ex1a - original vs corrigida",
                   cv2.resize(painel, None, fx=0.45, fy=0.45))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
