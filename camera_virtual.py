"""
CÂMERA VIRTUAL PARA CALIBRAÇÃO E REALIDADE AUMENTADA (Exercício 1)

Módulo de apoio dos scripts ex1a.py e ex1b.py.

Mantemos a abordagem de "laboratório virtual" usada em aula: em vez de depender de
uma webcam e de um tabuleiro impresso, as imagens do padrão de xadrez são geradas
por projeção perspectiva a partir de uma câmera de parâmetros conhecidos. A
vantagem didática é que existe um "valor verdadeiro" (TRUE_K, TRUE_DIST) para
comparar com o "valor estimado" pelo cv2.calibrateCamera.

Diferença em relação ao material de aula: aqui a câmera virtual possui distorção
de lente NÃO nula. Sem distorção, cv2.undistort devolve uma imagem visualmente
idêntica à original e o painel "original vs. corrigida" exigido no item A não
mostra efeito algum. Os coeficientes abaixo correspondem a uma barril moderada,
típica de webcams e de câmeras grande-angulares usadas em robótica móvel.

Significado físico dos parâmetros intrínsecos (matriz K):
    fx, fy : distância focal expressa em pixels (focal em mm dividida pelo
             tamanho físico do pixel). Relaciona ângulo de visada e pixels;
             é o que permite converter disparidade/tamanho aparente em metros.
    cx, cy : ponto principal, isto é, onde o eixo óptico atravessa o sensor.
             Idealmente no centro da imagem, mas o sensor nunca está
             perfeitamente alinhado com a lente.
    skew   : assumido zero em sensores modernos (pixels retangulares).

Significado físico dos coeficientes de distorção (k1, k2, p1, p2, k3):
    k1, k2, k3 : distorção RADIAL. A lente amplia ou comprime a imagem em
                 função da distância ao centro óptico. k1 < 0 produz efeito
                 barril (linhas retas curvam para fora nas bordas); k1 > 0
                 produz efeito almofada. É o erro dominante em lentes baratas.
    p1, p2     : distorção TANGENCIAL, causada por a lente não estar
                 perfeitamente paralela ao plano do sensor (descentragem).
                 Normalmente uma ordem de grandeza menor que a radial.
"""

from pathlib import Path

import cv2
import numpy as np

from common import imread_u, imwrite_u

# ---------------------------------------------------------------------------
# Geometria do padrão e da câmera virtual (mesma convenção usada em aula)
# ---------------------------------------------------------------------------

PATTERN_SIZE = (7, 6)        # cantos internos: colunas, linhas -> tabuleiro 8x7 quadrados
SQUARE_SIZE_M = 0.03         # 30 mm por quadrado
IMAGE_SIZE = (1280, 720)     # largura, altura em pixels

TRUE_K = np.array([
    [900.0,   0.0, 640.0],
    [  0.0, 910.0, 360.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

# [k1, k2, p1, p2, k3] -> barril moderada + leve descentragem da lente.
TRUE_DIST = np.array([[-0.26, 0.09, 0.0012, -0.0008, -0.015]], dtype=np.float64)

ZERO_DIST = np.zeros((1, 5), dtype=np.float64)

# Poses (rx, ry, rz em graus; tx, ty, tz em metros) cobrindo ângulos e
# distâncias variados, como pede o item A. Lista herdada do material de aula.
POSES = [
    (0, 0, 0, -0.12, -0.09, 0.43),
    (-8, 10, 4, -0.10, -0.07, 0.40),
    (10, -12, -5, -0.08, -0.06, 0.45),
    (15, 8, 8, -0.14, -0.08, 0.48),
    (-14, -10, -10, -0.07, -0.10, 0.42),
    (20, 5, 12, -0.11, -0.05, 0.51),
    (-18, 15, -8, -0.09, -0.08, 0.46),
    (5, 20, 15, -0.13, -0.07, 0.53),
    (-5, -20, -15, -0.06, -0.06, 0.45),
    (12, 18, -4, -0.10, -0.10, 0.56),
    (-12, -8, 14, -0.08, -0.04, 0.50),
    (22, -15, 6, -0.15, -0.06, 0.55),
    (-20, 12, 10, -0.05, -0.09, 0.48),
    (8, -22, 3, -0.12, -0.11, 0.52),
    (-7, 25, -6, -0.07, -0.05, 0.57),
    (16, -5, -12, -0.11, -0.09, 0.46),
    (-16, 6, 11, -0.09, -0.03, 0.53),
    (3, -10, 18, -0.13, -0.08, 0.49),
    # Poses acrescentadas por nós: levam o padrão para a PERIFERIA do quadro.
    # Sem elas, o tabuleiro aparece sempre na região central e os coeficientes
    # radiais de alta ordem (k2, k3) ficam mal determinados nas bordas, onde a
    # distorção é maior. Em calibração de câmera de robô vale a mesma regra:
    # o padrão precisa visitar os quatro cantos do campo de visão, além de
    # variar inclinação e distância.
    (6, -6, 0, -0.32, -0.21, 0.60),     # canto superior esquerdo
    (-6, 6, 0, 0.16, -0.21, 0.60),      # canto superior direito
    (6, 6, 0, -0.32, 0.055, 0.60),      # canto inferior esquerdo
    (-6, -6, 0, 0.16, 0.055, 0.60),     # canto inferior direito
    (0, 0, 0, -0.09, -0.075, 0.36),     # muito próximo, preenchendo o quadro
    (0, 0, 0, -0.09, -0.075, 0.95),     # distante, padrão pequeno
]


def ensure_dirs(root):
    root = Path(root)
    (root / "data" / "tabuleiro").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)


MARGEM_QUADRADOS = 3         # borda do plano em volta do tabuleiro
SQUARE_PX = 100              # resolução da textura: pixels por quadrado


def make_checkerboard(square_px=SQUARE_PX):
    """Textura com 8x7 quadrados, produzindo 7x6 cantos internos."""
    cols = PATTERN_SIZE[0] + 1
    rows = PATTERN_SIZE[1] + 1
    img = np.full((rows * square_px, cols * square_px), 255, np.uint8)
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x * square_px, y * square_px),
                    ((x + 1) * square_px, (y + 1) * square_px),
                    0, -1,
                )
    return img


def make_scene_texture(square_px=SQUARE_PX, margem=MARGEM_QUADRADOS):
    """
    Textura do plano da cena: tabuleiro no centro e grade de linhas retas em volta.

    A grade não é enfeite. A distorção radial só é perceptível a olho nu longe do
    centro óptico, e o tabuleiro sozinho ocupa a região central da imagem. Com
    linhas retas conhecidas cobrindo a borda do quadro, o efeito barril fica
    visível no painel do item A e a ação do cv2.undistort passa a ser
    verificável visualmente (as linhas voltam a ficar retas). O plano inteiro
    está em Z=0, portanto a geometria continua fisicamente consistente: é
    equivalente a imprimir o tabuleiro no centro de uma folha quadriculada.
    """
    board = make_checkerboard(square_px)
    bh, bw = board.shape
    m = margem * square_px

    scene = np.full((bh + 2 * m, bw + 2 * m), 235, np.uint8)

    # Grade de linhas retas com passo de um quadrado (30 mm no mundo real).
    for x in range(0, scene.shape[1] + 1, square_px):
        cv2.line(scene, (x, 0), (x, scene.shape[0]), 90, 3)
    for y in range(0, scene.shape[0] + 1, square_px):
        cv2.line(scene, (0, y), (scene.shape[1], y), 90, 3)

    # Zona de silêncio branca em volta do tabuleiro: o findChessboardCorners
    # exige margem clara para fechar o contorno do padrão.
    quiet = square_px
    cv2.rectangle(scene,
                  (m - quiet, m - quiet),
                  (m + bw + quiet, m + bh + quiet),
                  255, -1)

    scene[m:m + bh, m:m + bw] = board
    return scene


def scene_size_m(margem=MARGEM_QUADRADOS):
    """Dimensões físicas (largura, altura) do plano da cena, em metros."""
    cols = PATTERN_SIZE[0] + 1 + 2 * margem
    rows = PATTERN_SIZE[1] + 1 + 2 * margem
    return cols * SQUARE_SIZE_M, rows * SQUARE_SIZE_M


def board_origin_offset_m(margem=MARGEM_QUADRADOS):
    """Deslocamento da origem do tabuleiro em relação ao canto do plano, em metros."""
    return margem * SQUARE_SIZE_M, margem * SQUARE_SIZE_M


def euler_to_rvec(rx_deg, ry_deg, rz_deg):
    """Ângulos de Euler (XYZ, em graus) -> vetor de rotação de Rodrigues."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]], dtype=np.float64)
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64)
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(Rz @ Ry @ Rx)
    return rvec


# ---------------------------------------------------------------------------
# Distorção de lente
# ---------------------------------------------------------------------------

_DISTORTION_MAPS = {}


def _distortion_maps(K, dist, image_size):
    """
    Mapas que transformam a imagem ideal (pinhole puro) na imagem distorcida.

    O modelo do OpenCV vai do ponto normalizado ideal para o pixel distorcido.
    Para *gerar* a imagem distorcida precisamos do caminho inverso: para cada
    pixel de destino (distorcido), qual pixel da imagem ideal deve ser
    amostrado. cv2.undistortPoints resolve exatamente essa inversão de forma
    iterativa; aplicamos sobre a grade completa de pixels e guardamos o
    resultado em cache, já que a câmera virtual é sempre a mesma.
    """
    key = (image_size, K.tobytes(), dist.tobytes())
    if key in _DISTORTION_MAPS:
        return _DISTORTION_MAPS[key]

    w, h = image_size
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    grid = np.stack([us.ravel(), vs.ravel()], axis=1).reshape(-1, 1, 2)

    # Normalizado ideal correspondente a cada pixel distorcido.
    ideal = cv2.undistortPoints(grid, K, dist).reshape(-1, 2)

    # Normalizado ideal -> pixel na imagem ideal (projeção pinhole, sem distorção).
    map_x = (ideal[:, 0] * K[0, 0] + K[0, 2]).reshape(h, w).astype(np.float32)
    map_y = (ideal[:, 1] * K[1, 1] + K[1, 2]).reshape(h, w).astype(np.float32)

    _DISTORTION_MAPS[key] = (map_x, map_y)
    return map_x, map_y


def apply_distortion(img, K=TRUE_K, dist=TRUE_DIST, background=220):
    """Aplica a distorção da câmera virtual a uma imagem ideal."""
    h, w = img.shape[:2]
    map_x, map_y = _distortion_maps(K, dist, (w, h))
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=background)


# ---------------------------------------------------------------------------
# Síntese das vistas
# ---------------------------------------------------------------------------

def generate_view(rx=0, ry=0, rz=0, tx=-0.09, ty=-0.075, tz=0.45,
                  image_size=IMAGE_SIZE, K=TRUE_K, background=220,
                  blur=0.0, noise_std=0.0, distort=True):
    """
    Gera uma imagem sintética do tabuleiro e devolve (imagem, rvec, tvec).

    rx, ry, rz : orientação do tabuleiro em graus.
    tx, ty, tz : posição da origem do tabuleiro em metros.
    distort    : se True, aplica a distorção de lente TRUE_DIST.
    """
    w, h = image_size
    texture = make_scene_texture()

    # Cantos do plano da cena expressos no MESMO sistema de coordenadas de
    # object_points(), cuja origem é o primeiro canto interno do tabuleiro.
    # Assim rvec/tvec devolvidos aqui são diretamente comparáveis com o que o
    # cv2.solvePnP estima no item B.
    scene_w, scene_h = scene_size_m()
    off_x, off_y = board_origin_offset_m()
    x0 = -(off_x + SQUARE_SIZE_M)
    y0 = -(off_y + SQUARE_SIZE_M)

    outer_3d = np.array([
        [x0,           y0,           0.0],
        [x0 + scene_w, y0,           0.0],
        [x0 + scene_w, y0 + scene_h, 0.0],
        [x0,           y0 + scene_h, 0.0],
    ], dtype=np.float32)

    rvec = euler_to_rvec(rx, ry, rz)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    # O plano é projetado sem distorção: o resultado é uma homografia exata.
    # A distorção entra depois, como deformação da imagem inteira.
    dst, _ = cv2.projectPoints(outer_3d, rvec, tvec, K, ZERO_DIST)
    dst = dst.reshape(-1, 2).astype(np.float32)

    src = np.array([
        [0, 0],
        [texture.shape[1] - 1, 0],
        [texture.shape[1] - 1, texture.shape[0] - 1],
        [0, texture.shape[0] - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    bg = np.full((h, w), background, dtype=np.uint8)
    warped = cv2.warpPerspective(texture, H, (w, h), borderValue=background)
    mask = cv2.warpPerspective(np.full(texture.shape, 255, np.uint8), H,
                               (w, h), borderValue=0)
    bg[mask > 0] = warped[mask > 0]

    if distort:
        bg = apply_distortion(bg, K, TRUE_DIST, background)

    if blur > 0:
        k = int(max(3, round(blur) * 2 + 1))
        if k % 2 == 0:
            k += 1
        bg = cv2.GaussianBlur(bg, (k, k), blur)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, bg.shape)
        bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), rvec, tvec


def generate_dataset(root, n=18, noisy=False, distort=True):
    """Gera n vistas de calibração em data/tabuleiro e devolve os caminhos."""
    root = Path(root)
    ensure_dirs(root)
    out = root / "data" / "tabuleiro"

    paths = []
    for i, pose in enumerate(POSES[:n], start=1):
        image, _, _ = generate_view(
            *pose,
            blur=0.8 if noisy else 0.0,
            noise_std=2.0 if noisy else 0.0,
            distort=distort,
        )
        path = out / f"calib_{i:02d}.png"
        imwrite_u(path, image)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Detecção de cantos, calibração e erro de reprojeção
# ---------------------------------------------------------------------------

def object_points():
    """Coordenadas 3D dos 7x6 cantos internos, no plano Z=0, em metros."""
    obj = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    obj *= SQUARE_SIZE_M
    return obj


def find_refined_corners(image):
    """Detecta os cantos do tabuleiro e refina para precisão subpixel."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def calibrate_from_paths(paths):
    """Executa o pipeline completo de calibração sobre uma lista de imagens."""
    objpoints, imgpoints, used = [], [], []
    image_size = None
    obj = object_points()

    for path in paths:
        img = imread_u(path)
        if img is None:
            continue
        image_size = (img.shape[1], img.shape[0])
        ok, corners = find_refined_corners(img)
        if ok:
            objpoints.append(obj.copy())
            imgpoints.append(corners)
            used.append(Path(path))

    if len(objpoints) < 3:
        raise RuntimeError("Poucas imagens válidas para calibração.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, image_size


def reprojection_errors(K, dist, rvecs, tvecs, objpoints, imgpoints):
    """Erro médio de reprojeção (em pixels) de cada imagem."""
    errors = []
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
        errors.append(float(err))
    return errors


def save_calibration(path, K, dist, image_size, rms):
    np.savez(str(path), K=K, dist=dist,
             width=image_size[0], height=image_size[1], rms=rms)


def load_calibration(path):
    data = np.load(str(path))
    return (data["K"], data["dist"],
            (int(data["width"]), int(data["height"])), float(data["rms"]))
