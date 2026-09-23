# AT — Visão Computacional com OpenCV (DR2)

Pipeline de percepção visual desenvolvido no Assessment da disciplina DR2
(Sistemas Robóticos) do Instituto Infnet: calibração de câmera e realidade
aumentada, classificação com OpenCV DNN, detecção com YOLO e SSD, rastreamento
com ID persistente e segmentação semântica.

Este repositório existe para hospedar o material em **resolução original** —
imagens PNG sem perdas e vídeos na qualidade de gravação. O pacote entregue na
plataforma da faculdade traz os mesmos arquivos recomprimidos, por causa do
limite de 20 MB.

## Relatório

O relatório técnico completo, com as evidências e o relatório integrativo, está
em [`Gabriel_Cruz_DR2_AT.pdf`](Gabriel_Cruz_DR2_AT.pdf). O relatório integrativo
exigido no Exercício 4B também está separado em
[`relatorio_integrativo.md`](relatorio_integrativo.md).

## Exercícios

| Exercício | Arquivos | Resultado principal |
|---|---|---|
| 1 — Calibração e realidade aumentada | `camera_virtual.py`, `ex1a.py`, `ex1b.py` | erro de reprojeção de 0,0202 px; erro de pose de 0,31 mm |
| 2 — OpenCV DNN vs. Keras e pipeline | `preparar_dados_ex2.py`, `ex2a.py`, `pipeline_ex2b.py`, `ex2b.ipynb` | OpenCV DNN 2× mais rápido e com 5,3× menos memória |
| 3 — YOLO, SSD e rastreamento | `preparar_modelos_ex3.py`, `detectores.py`, `ex3a.py`, `ex3b.py` | 24,6 e 31,1 FPS; 10,19 ID switches por minuto |
| 4 — Segmentação semântica e relatório | `preparar_dados_ex4.py`, `ex4a.py`, `ex4b.py` | 93,9% dos pixels como "fundo" no vocabulário VOC |

## Como executar

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

.venv/Scripts/python.exe preparar_dados_ex2.py
.venv/Scripts/python.exe preparar_modelos_ex3.py
.venv/Scripts/python.exe preparar_dados_ex4.py

.venv/Scripts/python.exe ex1a.py      # e assim por diante, de ex1a a ex4b
```

Detalhes de ambiente, ordem de execução e justificativas de cada decisão estão
na seção 1 do relatório.

## Evidências

`outputs/` contém as saídas integrais de terminal (`*_terminal.txt`), os painéis
e gráficos em PNG e os vídeos de demonstração, incluindo o do rastreamento com ID
persistente exigido no Exercício 3B.

## Observação sobre o ambiente

O ambiente exige `opencv-python` **abaixo da versão 5**: a 5.0 removeu
`HOGDescriptor`, `CascadeClassifier` e os rastreadores do módulo principal, que o
Exercício 2B utiliza.
