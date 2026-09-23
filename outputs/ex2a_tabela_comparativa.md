# Exercício 2A — OpenCV DNN vs. Keras (MobileNetV2)

| backend | latência (ms) | memória do processo (MB) | acurácia top-1 | acurácia top-3 | modelo em disco (MB) |
|---|---|---|---|---|---|
| OpenCV DNN (TFLite) | 7.06 | 81.8 | 91.7% (11/12) | 100.0% | 13.35 |
| Keras / TensorFlow (tf.function) | 13.87 | 437.0 | 91.7% (11/12) | 100.0% | 14.05 |

Concordância do top-1: 12/12. Maior diferença de probabilidade: 4.77e-02.
