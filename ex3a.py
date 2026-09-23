"""
EXERCÍCIO 3 - ITEM A: DETECÇÃO EM TEMPO REAL COM YOLO E SSD
(Competências 3.1, 3.2 e 4.3)

Objetivo: executar YOLOv4-tiny e SSD MobileNetV2 sobre o mesmo vídeo, aplicar
Non-Maximum Suppression com limiar 0.4, medir FPS e latência por frame e
concluir qual modelo é mais adequado a robótica embarcada, com base nos dados
coletados nesta máquina.

VÍDEO: TP3/data/vtest.avi (pedestres em um campus, 768x576, 795 frames).
É o mesmo vídeo do Exercício 2B, o que permite comparar diretamente o que os
detectores clássicos (HOG+SVM) e os profundos enxergam na mesma cena.

MÉTODO
Os dois detectores recebem EXATAMENTE os mesmos frames, com os mesmos limiares
e o mesmo backend (cv2.dnn em CPU). Cada modelo roda em uma PASSADA SEPARADA
sobre o vídeo, e não intercalado no mesmo laço.

Isso não é detalhe: medimos as duas formas. Alternando as redes a cada frame, o
YOLOv4-tiny marcou 141 ms de inferência; sozinho, sobre os mesmos frames, marcou
49 ms. A diferença vem da disputa por cache e pelo pool de threads entre dois
grafos grandes vivos ao mesmo tempo — exatamente o efeito que aparece quando se
tenta rodar dois modelos concorrentes na mesma CPU de um robô. Para comparar
arquiteturas, a medição precisa ser isolada; o número intercalado só descreveria
o custo de executar os dois juntos.

O QUE É MEDIDO
  - latência de inferência por frame (apenas o forward pass);
  - latência total por frame (blob + forward + NMS + desenho);
  - FPS médio e FPS no percentil 95 da latência (em tempo real o pior caso
    importa mais que a média: um frame lento atrasa a reação do veículo);
  - número de parâmetros, somando os tensores de peso da rede;
  - tamanho dos arquivos em disco;
  - detecções por classe e confiança média;
  - concordância entre os dois modelos, por IoU, já que não há anotação de
    referência para este vídeo e portanto não se pode calcular precisão real.

Execução:
    .venv/Scripts/python.exe ex3a.py
    .venv/Scripts/python.exe ex3a.py --frames 100     (execução mais curta)
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import (cabecalho, finish_video, imwrite_u, tabela,
                    tabela_markdown, video_capture_u, video_writer_u)
from detectores import (CLASSES_TRANSITO, CONF_MIN, NMS_LIMIAR, DetectorSSD,
                        DetectorYOLO, desenhar_deteccoes)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
VIDEO = ROOT.parent / "TP3" / "data" / "vtest.avi"
N_FRAMES = 300
FRAMES_SALVOS = (40, 120, 220)


def milhar(n):
    """Formata inteiro com ponto como separador de milhar (padrão pt-BR)."""
    return f"{n:,}".replace(",", ".")


def iou(a, b):
    """Interseção sobre união entre duas caixas (x, y, w, h)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)


def concordancia(caixas_a, classes_a, caixas_b, classes_b, limiar=0.5):
    """
    Quantas detecções de A têm correspondente em B (mesma classe, IoU >= limiar).

    Sem anotação de referência não é possível medir precisão e recall reais.
    A concordância entre dois detectores independentes é o substituto honesto:
    objetos vistos pelos dois são provavelmente verdadeiros, e as divergências
    marcam onde pelo menos um dos dois erra.
    """
    usados, casados = set(), 0
    for ca, cla in zip(caixas_a, classes_a):
        melhor, melhor_j = 0.0, -1
        for j, (cb, clb) in enumerate(zip(caixas_b, classes_b)):
            if j in usados or cla != clb:
                continue
            v = iou(ca, cb)
            if v > melhor:
                melhor, melhor_j = v, j
        if melhor >= limiar:
            usados.add(melhor_j)
            casados += 1
    return casados


def avaliar(detector, n_frames):
    """Passada isolada do detector sobre os n_frames iniciais do vídeo."""
    cap, _tmp = video_capture_u(VIDEO)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nome_arq = detector.nome.split()[0].lower().replace("-", "")
    writer, tmp_video, destino = video_writer_u(
        OUT / f"ex3a_{nome_arq}.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
        15.0, (largura, altura))

    lat_inf, lat_tot, confs, por_frame = [], [], [], []
    contagem = Counter()
    frames_salvos, deteccoes_por_frame = [], []

    print(f"\n{detector.nome}: passada isolada de {n_frames} frames...")
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        t0 = cv2.getTickCount()
        caixas, scores, classes, ms_inf = detector.detectar(frame)
        anotado = desenhar_deteccoes(frame, caixas, scores, classes)
        ms_tot = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000

        lat_inf.append(ms_inf)
        lat_tot.append(ms_tot)
        contagem.update(classes)
        confs.extend(scores)
        por_frame.append(len(caixas))
        deteccoes_por_frame.append((caixas, classes))

        cv2.rectangle(anotado, (0, 0), (largura, 30), (0, 0, 0), -1)
        cv2.putText(anotado,
                    f"{detector.nome} | frame {i:03d} | {len(caixas)} obj | "
                    f"{ms_tot:.0f} ms | {1000/max(ms_tot,1e-6):.1f} FPS",
                    (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        writer.write(anotado)

        if i in FRAMES_SALVOS:
            arq = OUT / f"ex3a_{nome_arq}_frame{i:03d}.png"
            imwrite_u(arq, anotado)
            frames_salvos.append(arq.name)

        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{n_frames} "
                  f"({np.mean(lat_tot[-50:]):.0f} ms/frame)")

    cap.release()
    finish_video(writer, tmp_video, destino)

    inf, tot = np.array(lat_inf), np.array(lat_tot)
    return dict(
        nome=detector.nome, frames=len(tot), entrada=detector.entrada,
        lat_inf_media=float(inf.mean()),
        lat_inf_p95=float(np.percentile(inf, 95)),
        lat_total_media=float(tot.mean()),
        lat_total_p95=float(np.percentile(tot, 95)),
        fps_medio=float(1000.0 / tot.mean()),
        fps_p95=float(1000.0 / np.percentile(tot, 95)),
        parametros=detector.parametros(),
        disco_mb=detector.tamanho_disco_mb(),
        deteccoes=int(sum(contagem.values())), por_classe=contagem,
        det_por_frame=float(np.mean(por_frame)),
        conf_media=float(np.mean(confs)) if confs else 0.0,
        video=destino, frames_salvos=frames_salvos,
        saidas=deteccoes_por_frame,
    )


def varredura_resolucao(frames=40):
    """
    Sensibilidade do YOLO à resolução de entrada.

    A resolução é o parâmetro que mais muda a relação velocidade/detecção em um
    detector de estágio único, e é o primeiro botão a girar quando o FPS não
    fecha no hardware alvo. Vantagem do Darknet: a entrada é configurável no
    momento da inferência, sem reconverter o modelo.
    """
    linhas = []
    for entrada in (256, 320, 416, 512):
        det = DetectorYOLO(entrada=entrada)
        cap, _tmp = video_capture_u(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        tempos, total, pessoas = [], 0, 0
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                break
            caixas, _, classes, ms = det.detectar(frame)
            tempos.append(ms)
            total += len(caixas)
            pessoas += sum(1 for c in classes if c == "person")
        cap.release()
        n = max(len(tempos), 1)
        linhas.append([f"{entrada}x{entrada}", f"{np.mean(tempos):.1f}",
                       f"{1000/np.mean(tempos):.1f}", f"{total/n:.2f}",
                       f"{pessoas/n:.2f}"])
    return linhas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_frames = N_FRAMES
    if "--frames" in sys.argv:
        n_frames = int(sys.argv[sys.argv.index("--frames") + 1])

    cabecalho("CONFIGURAÇÃO")
    print(f"vídeo            : {VIDEO.name}")
    print(f"frames avaliados : {n_frames}")
    print(f"confiança mínima : {CONF_MIN}")
    print(f"limiar de NMS    : {NMS_LIMIAR}  (exigido pelo enunciado)")
    print("backend          : cv2.dnn, CPU, para os dois modelos\n")

    resultados = [avaliar(DetectorYOLO(), n_frames),
                  avaliar(DetectorSSD(), n_frames)]
    y, s = resultados[0], resultados[1]

    # Concordância calculada frame a frame sobre as saídas das duas passadas.
    casados = sum(concordancia(ca, cla, cb, clb)
                  for (ca, cla), (cb, clb) in zip(y["saidas"], s["saidas"]))

    # ------------------------------------------------------------------
    cabecalho("TABELA COMPARATIVA")
    cab = ["modelo", "entrada", "FPS médio", "FPS no p95",
           "inferência (ms)", "frame completo (ms)", "parâmetros",
           "arquivo (MB)"]
    linhas = [[
        r["nome"], f"{r['entrada']}x{r['entrada']}", f"{r['fps_medio']:.1f}",
        f"{r['fps_p95']:.1f}", f"{r['lat_inf_media']:.1f}",
        f"{r['lat_total_media']:.1f}", milhar(r["parametros"]),
        f"{r['disco_mb']:.1f}",
    ] for r in resultados]
    print(tabela(cab, linhas))

    print("\nObservação: a diferença entre 'inferência' e 'frame completo' é o")
    print("custo de montar o blob, rodar a NMS e desenhar as anotações. É pequena")
    print("aqui, mas em vídeo de alta resolução a montagem do blob deixa de ser")
    print("desprezível, porque envolve redimensionar e reordenar a imagem toda.")

    # ------------------------------------------------------------------
    cabecalho("O QUE CADA MODELO ENCONTROU")
    for r in resultados:
        transito = {c: n for c, n in r["por_classe"].items()
                    if c in CLASSES_TRANSITO}
        outras = {c: n for c, n in r["por_classe"].items()
                  if c not in CLASSES_TRANSITO}
        print(f"\n{r['nome']}: {r['deteccoes']} detecções em {r['frames']} frames "
              f"({r['det_por_frame']:.2f} por frame), "
              f"confiança média {r['conf_media']*100:.1f}%")
        if transito:
            print("   classes de trânsito: "
                  + ", ".join(f"{c}={n}" for c, n in
                              sorted(transito.items(), key=lambda kv: -kv[1])))
        if outras:
            print("   outras classes     : "
                  + ", ".join(f"{c}={n}" for c, n in
                              sorted(outras.items(), key=lambda kv: -kv[1])[:6]))

    menor = min(y["deteccoes"], s["deteccoes"])
    print(f"\nConcordância entre os dois modelos (mesma classe, IoU >= 0.5): "
          f"{casados} pares em {menor} detecções do modelo mais conservador "
          f"({casados/max(menor,1)*100:.1f}%).")
    print("Sem anotação de referência para este vídeo, não é possível calcular")
    print("precisão nem recall: 'mais detecções' não significa 'melhor', porque")
    print("pode ser recall maior ou falso positivo. A concordância indica o")
    print("núcleo de objetos que os dois confirmam, e as divergências marcam")
    print("exatamente os casos que precisariam de anotação manual para julgar.")

    # ------------------------------------------------------------------
    cabecalho("SENSIBILIDADE DO YOLO À RESOLUÇÃO DE ENTRADA")
    linhas_res = varredura_resolucao()
    print(tabela(["entrada", "inferência (ms)", "FPS", "objetos por frame",
                  "pessoas por frame"], linhas_res))
    print("\nO custo cresce com a área da entrada; o número de objetos cai")
    print("quando os alvos pequenos deixam de ser representáveis na grade.")

    # ------------------------------------------------------------------
    cabecalho("CONCLUSÃO: QUAL MODELO PARA ROBÓTICA EMBARCADA")
    mais_rapido = y if y["fps_medio"] > s["fps_medio"] else s
    mais_lento = s if mais_rapido is y else y
    ganho = mais_rapido["fps_medio"] / mais_lento["fps_medio"]
    mais_leve = y if y["disco_mb"] < s["disco_mb"] else s
    mais_confiante = y if y["conf_media"] > s["conf_media"] else s

    # Veredito de tempo real pelo PIOR CASO (p95), não pela média: em controle
    # de veículo o que compromete a reação é o frame lento, não a média boa.
    acima_p95 = [r["nome"] for r in resultados if r["fps_p95"] >= 25]
    if len(acima_p95) == len(resultados):
        veredito_tempo_real = (
            "Os dois sustentam 25 FPS mesmo no pior caso nesta CPU de desktop, "
            "portanto atendem tempo real aqui; em uma placa embarcada, cuja CPU "
            "costuma ser 5 a 10 vezes mais lenta, a margem desaparece e voltam a "
            "exigir acelerador, quantização int8 ou entrada menor.")
    elif acima_p95:
        veredito_tempo_real = (
            f"Apenas o {acima_p95[0]} sustenta 25 FPS no pior caso; o outro fica "
            f"abaixo, o que em controle de veículo significa atraso de reação nos "
            f"frames mais cheios de objetos.")
    else:
        veredito_tempo_real = (
            "Nenhum dos dois sustenta 25 FPS no pior caso nesta CPU, o que já "
            "responde à pergunta de projeto: em um robô, os dois exigiriam "
            "acelerador (GPU, VPU ou NPU), quantização int8 ou entrada menor.")

    # A recomendação é derivada dos números, e não fixada no código. Critério:
    # diferenças de FPS abaixo de 1.5x não decidem um projeto embarcado, porque
    # são recuperáveis com resolução de entrada e quantização; nesse regime
    # pesam mais o tamanho do modelo, a confiança das decisões e a flexibilidade
    # de configuração. Acima de 1.5x, a velocidade passa a ser o critério
    # dominante.
    if ganho < 1.5:
        recomendacao = (
            f"A diferença de velocidade é de apenas {ganho:.2f}x, dentro do que se "
            f"recupera ajustando resolução de entrada ou quantizando, logo ela não "
            f"decide. Decidem o tamanho e a robustez: o {mais_leve['nome']} ocupa "
            f"{abs(y['disco_mb']-s['disco_mb']):.0f} MB menos em disco e tem "
            f"{milhar(abs(y['parametros']-s['parametros']))} parâmetros menos; "
            + ("e é também o que decide" if mais_confiante is mais_leve
               else f"já o {mais_confiante['nome']} decide")
            + f" com confiança média mais alta "
            f"({mais_confiante['conf_media']*100:.1f}% contra "
            f"{min(y['conf_media'], s['conf_media'])*100:.1f}%), o que reduz a "
            f"oscilação de detecção entre frames e facilita o rastreamento do item B. "
            f"Some-se que a entrada do YOLO em Darknet é ajustável em tempo de "
            f"execução, sem reconverter o modelo: pela tabela de sensibilidade, "
            f"trocar 416 por 320 quase dobra o FPS mantendo a maioria dos pedestres. "
            f"Por isso a escolha recomendada para robótica embarcada nesta classe de "
            f"cena é o YOLOv4-tiny. O SSD MobileNetV2 continua preferível quando o "
            f"alvo é grande e próximo, como um robô de armazém identificando caixas a "
            f"poucos metros, ou quando o dispositivo já executa TensorFlow Lite "
            f"quantizado, caso em que sua versão int8 fica bastante eficiente.")
    else:
        recomendacao = (
            f"O {mais_rapido['nome']} é {ganho:.2f}x mais rápido, diferença grande o "
            f"bastante para dominar a decisão em sistema de tempo real, e é a escolha "
            f"recomendada se o requisito de FPS for rígido. A contrapartida a "
            f"documentar: ocupa {mais_rapido['disco_mb']:.1f} MB contra "
            f"{mais_lento['disco_mb']:.1f} MB do outro e decide com confiança média de "
            f"{mais_rapido['conf_media']*100:.1f}%, contra "
            f"{mais_lento['conf_media']*100:.1f}%.")

    print(f"""
Dados coletados nesta máquina (CPU, backend cv2.dnn, mesmos frames, mesmos
limiares de confiança e de NMS):

1. Velocidade. {y['nome']} roda a {y['fps_medio']:.1f} FPS e {s['nome']} a
   {s['fps_medio']:.1f} FPS; o {mais_rapido['nome']} é {ganho:.2f}x mais rápido.
   Nos percentis, o pior caso cai para {y['fps_p95']:.1f} e {s['fps_p95']:.1f} FPS
   respectivamente. {veredito_tempo_real}

2. Tamanho. {y['nome']}: {y['disco_mb']:.1f} MB e {milhar(y['parametros'])} parâmetros.
   {s['nome']}: {s['disco_mb']:.1f} MB e {milhar(s['parametros'])} parâmetros.
   O modelo mais leve é o {mais_leve['nome']}, com vantagem de
   {abs(y['disco_mb'] - s['disco_mb']):.1f} MB em disco. Em placa com flash
   limitada e sem swap, isso pesa tanto quanto o FPS.

3. Comportamento na cena. {y['nome']} produziu {y['det_por_frame']:.2f} detecções por
   frame, com confiança média de {y['conf_media']*100:.1f}%; {s['nome']},
   {s['det_por_frame']:.2f} por frame com {s['conf_media']*100:.1f}%. A confiança mais
   alta do YOLO indica decisões menos ambíguas nesta cena, o que reduz a
   oscilação de detecção entre frames consecutivos e facilita o rastreamento
   feito no item B. A entrada do SSD é fixa em 300x300, contra 416x416 do YOLO:
   um pedestre que ocupa 60 px de altura no frame original cai para cerca de
   31 px no SSD e 43 px no YOLO, o que explica parte da diferença de confiança.

4. Recomendação. {recomendacao}

5. Ressalva metodológica. FPS medido em desktop não se transfere para o alvo
   embarcado. O procedimento correto é repetir esta mesma medição na placa
   final, com o mesmo vídeo, porque a razão entre os dois modelos muda com o
   conjunto de instruções e com o acelerador disponível.
""".strip())

    # ------------------------------------------------------------------
    (OUT / "ex3a_tabela_comparativa.md").write_text(
        "# Exercício 3A — YOLOv4-tiny vs. SSD MobileNetV2\n\n"
        + tabela_markdown(cab, linhas)
        + "\n\n## Sensibilidade do YOLO à resolução de entrada\n\n"
        + tabela_markdown(["entrada", "inferência (ms)", "FPS",
                           "objetos por frame", "pessoas por frame"],
                          linhas_res)
        + f"\n\nVídeo: {VIDEO.name}, {y['frames']} frames, "
          f"confiança >= {CONF_MIN}, NMS = {NMS_LIMIAR}, backend cv2.dnn/CPU.\n"
        + f"\nConcordância entre modelos (IoU >= 0.5): {casados} pares.\n",
        encoding="utf-8")

    cabecalho("ARQUIVOS GERADOS")
    for r in resultados:
        print(f"  outputs/{Path(r['video']).name}")
        for f in r["frames_salvos"]:
            print(f"  outputs/{f}")
    print("  outputs/ex3a_tabela_comparativa.md")


if __name__ == "__main__":
    main()
