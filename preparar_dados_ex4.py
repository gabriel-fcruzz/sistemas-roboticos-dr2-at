"""
PREPARAÇÃO DAS IMAGENS DO EXERCÍCIO 4

Monta o conjunto de cenas externas em data/cenas/, sem baixar nada novo:

  1. três frames do vídeo TP3/data/vtest.avi, em momentos distintos.
     São cenas externas reais de um campus, com via, calçada, gramado,
     pedestres e veículos estacionados. Usar o mesmo vídeo dos exercícios 2B e
     3 permite comparar as três abordagens (cor, detecção e segmentação) sobre
     a MESMA cena, que é o objetivo integrativo do AT.

  2. quatro fotografias reais já baixadas para o Exercício 2A: ônibus escolar,
     placa de rua, banco de praça e semáforo. Trazem variedade de
     enquadramento, iluminação e escala, o que evita concluir algo a partir de
     uma única cena.

Total: sete imagens, acima do mínimo de cinco pedido no enunciado.

Execução:
    .venv/Scripts/python.exe preparar_dados_ex4.py
"""

from pathlib import Path

import cv2

from common import imread_u, imwrite_u, video_capture_u

ROOT = Path(__file__).resolve().parent
DESTINO = ROOT / "data" / "cenas"
VIDEO = ROOT.parent / "TP3" / "data" / "vtest.avi"
CLASSIFICACAO = ROOT / "data" / "classificacao"

FRAMES_VIDEO = [(120, "campus_via"), (400, "campus_passeio"),
                (640, "campus_gramado")]
FOTOS = ["779_school_bus.jpg", "919_street_sign.jpg",
         "703_park_bench.jpg", "920_traffic_light.jpg"]


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    salvos = []

    print("1) Frames do vídeo do campus")
    cap, _tmp = video_capture_u(VIDEO)
    for idx, nome in FRAMES_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"   [falhou] frame {idx}")
            continue
        destino = DESTINO / f"cena_{nome}.png"
        imwrite_u(destino, frame)
        print(f"   {destino.name:32s} {frame.shape[1]}x{frame.shape[0]}")
        salvos.append(destino.name)
    cap.release()

    print("\n2) Fotografias reais reaproveitadas do Exercício 2A")
    for arquivo in FOTOS:
        origem = CLASSIFICACAO / arquivo
        img = imread_u(origem)
        if img is None:
            print(f"   [ausente] {arquivo} — execute preparar_dados_ex2.py")
            continue
        destino = DESTINO / f"cena_{origem.stem.split('_', 1)[1]}.png"
        imwrite_u(destino, img)
        print(f"   {destino.name:32s} {img.shape[1]}x{img.shape[0]}")
        salvos.append(destino.name)

    print(f"\n{len(salvos)} cenas em data/cenas/")


if __name__ == "__main__":
    main()
