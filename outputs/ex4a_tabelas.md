# Exercício 4A — segmentação semântica vs. HSV

## Custo por cena

| cena | DeepLabV3 (ms) | FCN (ms) | HSV (ms) | razão rede/HSV |
|---|---|---|---|---|
| campus_gramado | 1002 | 800 | 4.9 | 206x |
| campus_passeio | 964 | 770 | 3.3 | 290x |
| campus_via | 958 | 847 | 2.4 | 404x |
| park_bench | 860 | 674 | 2.3 | 373x |
| school_bus | 829 | 660 | 2.5 | 329x |
| street_sign | 902 | 732 | 3.4 | 269x |
| traffic_light | 925 | 767 | 3.9 | 236x |

## Cobertura do vocabulário

| cena | fundo (%) | classe útil (%) | via pelo HSV (%) |
|---|---|---|---|
| campus_gramado | 96.7 | 3.3 | 31.0 |
| campus_passeio | 98.1 | 1.9 | 31.3 |
| campus_via | 97.6 | 2.4 | 30.8 |
| park_bench | 100.0 | 0.0 | 14.9 |
| school_bus | 64.8 | 35.2 | 7.7 |
| street_sign | 100.0 | 0.0 | 0.8 |
| traffic_light | 100.0 | 0.0 | 3.4 |

## Área por classe (DeepLabV3 e FCN)

| cena | modelo | áreas |
|---|---|---|
| campus_gramado | DeepLabV3-ResNet50 | __background__ 96.7%, person 2.6%, car 0.6% |
| campus_gramado | FCN-ResNet50 | __background__ 97.0%, person 2.6%, car 0.4% |
| campus_passeio | DeepLabV3-ResNet50 | __background__ 98.1%, person 1.3%, car 0.6% |
| campus_passeio | FCN-ResNet50 | __background__ 98.5%, person 1.2%, car 0.3% |
| campus_via | DeepLabV3-ResNet50 | __background__ 97.6%, person 1.9%, car 0.5% |
| campus_via | FCN-ResNet50 | __background__ 97.9%, person 1.9%, car 0.3% |
| park_bench | DeepLabV3-ResNet50 | __background__ 100.0% |
| park_bench | FCN-ResNet50 | __background__ 100.0% |
| school_bus | DeepLabV3-ResNet50 | __background__ 64.8%, bus 34.7%, car 0.5% |
| school_bus | FCN-ResNet50 | __background__ 72.4%, bus 27.0%, car 0.6% |
| street_sign | DeepLabV3-ResNet50 | __background__ 100.0% |
| street_sign | FCN-ResNet50 | __background__ 100.0% |
| traffic_light | DeepLabV3-ResNet50 | __background__ 100.0% |
| traffic_light | FCN-ResNet50 | __background__ 100.0% |
