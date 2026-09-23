"""
UTILITÁRIOS COMPARTILHADOS PELOS EXERCÍCIOS DO AT

Concentra três coisas usadas por todos os scripts:

1. Leitura e escrita de imagens à prova de acento no caminho.
   No Windows, cv2.imread e cv2.imwrite repassam o caminho ao decodificador em
   codificação ANSI. Como a pasta deste trabalho contém acento ("Sistemas
   Robóticos"), as duas funções falham silenciosamente: imwrite devolve False e
   imread devolve None. A solução é fazer a I/O em Python (np.fromfile /
   ndarray.tofile) e usar cv2.imdecode / cv2.imencode apenas para converter
   entre bytes e matriz. Em pastas sem acento o comportamento é idêntico.

2. Medição de tempo, para as métricas de latência exigidas nos exercícios 2, 3 e 4.

3. Impressão de tabelas em texto, reaproveitada nos comparativos.
"""

import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O de imagens e vídeo
# ---------------------------------------------------------------------------

def imread_u(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread tolerante a acentos no caminho. Devolve None se falhar."""
    path = Path(path)
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_u(path, img):
    """cv2.imwrite tolerante a acentos no caminho. Devolve True em caso de sucesso."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


_CACHE_ASCII = {}


def caminho_ascii(path):
    """
    Devolve um caminho que os carregadores do OpenCV conseguem abrir.

    Vale para cv2.dnn.readNetFrom* , cv2.CascadeClassifier e cv2.VideoCapture:
    todos recebem o caminho como bytes em codificação ANSI. Com acento no
    caminho (aqui, "Sistemas Robóticos") a abertura falha. Quando detectamos
    caractere não-ASCII, copiamos o arquivo uma única vez para a pasta
    temporária do sistema e devolvemos esse caminho; em pastas sem acento o
    caminho original é devolvido sem cópia.
    """
    import shutil
    import tempfile

    path = Path(path)
    texto = str(path)
    if texto.isascii():
        return texto
    if texto in _CACHE_ASCII and Path(_CACHE_ASCII[texto]).exists():
        return _CACHE_ASCII[texto]

    destino = Path(tempfile.gettempdir()) / "at_dr2_modelos" / path.name
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists() or destino.stat().st_size != path.stat().st_size:
        shutil.copy2(str(path), str(destino))
    _CACHE_ASCII[texto] = str(destino)
    return str(destino)


def video_writer_u(path, fourcc, fps, size):
    """
    cv2.VideoWriter também não abre caminhos com acento. Escrevemos o vídeo em
    um caminho temporário ASCII e o movemos ao final (ver finish_video).
    """
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()) / f"at_tmp_{path.name}"
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, size)
    return writer, tmp, path


def finish_video(writer, tmp, final):
    """Fecha o writer e move o arquivo temporário para o destino final."""
    import shutil

    writer.release()
    tmp, final = Path(tmp), Path(final)
    if tmp.exists():
        shutil.move(str(tmp), str(final))
    return final


def video_capture_u(path):
    """
    cv2.VideoCapture com caminho acentuado: copia para um temporário ASCII.
    Devolve (capture, caminho_temporario_ou_None).
    """
    import shutil
    import tempfile

    path = Path(path)
    try:
        path.resolve().relative_to(path.resolve())
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            return cap, None
    except Exception:
        pass
    tmp = Path(tempfile.gettempdir()) / f"at_in_{path.name}"
    shutil.copy2(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp)), tmp


# ---------------------------------------------------------------------------
# Medição de tempo
# ---------------------------------------------------------------------------

class Cronometro:
    """
    Mede o tempo de um bloco em milissegundos.

        with Cronometro() as c:
            faz_algo()
        print(c.ms)
    """

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False


def medir_ms(func, loops=10, warmup=3):
    """
    Latência média e desvio padrão de uma função, em ms.

    O aquecimento (warmup) é obrigatório em medições de inferência: a primeira
    chamada inclui alocação de buffers e otimização de grafo, e chega a ser uma
    ordem de grandeza mais lenta que o regime permanente.
    """
    for _ in range(warmup):
        func()
    tempos = []
    for _ in range(loops):
        t0 = time.perf_counter()
        func()
        tempos.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(tempos)), float(np.std(tempos))


def memoria_processo_mb():
    """Memória residente do processo em MB (requer psutil)."""
    try:
        import os

        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return float("nan")


# ---------------------------------------------------------------------------
# Tabelas em texto e markdown
# ---------------------------------------------------------------------------

def tabela(cabecalho, linhas, titulo=None):
    """Formata uma tabela de texto alinhada para o terminal."""
    cols = len(cabecalho)
    larguras = [len(str(cabecalho[i])) for i in range(cols)]
    for linha in linhas:
        for i in range(cols):
            larguras[i] = max(larguras[i], len(str(linha[i])))

    sep = "+" + "+".join("-" * (w + 2) for w in larguras) + "+"
    out = []
    if titulo:
        out.append(titulo)
    out.append(sep)
    out.append("| " + " | ".join(str(cabecalho[i]).ljust(larguras[i])
                                 for i in range(cols)) + " |")
    out.append(sep)
    for linha in linhas:
        out.append("| " + " | ".join(str(linha[i]).ljust(larguras[i])
                                     for i in range(cols)) + " |")
    out.append(sep)
    return "\n".join(out)


def tabela_markdown(cabecalho, linhas):
    """Mesma tabela em markdown, para colar no relatório."""
    out = ["| " + " | ".join(str(c) for c in cabecalho) + " |",
           "|" + "|".join("---" for _ in cabecalho) + "|"]
    for linha in linhas:
        out.append("| " + " | ".join(str(c) for c in linha) + " |")
    return "\n".join(out)


def cabecalho(titulo):
    print("\n" + "=" * 68)
    print(titulo)
    print("=" * 68)
