# Investigação de Funções de Perda na Segmentação de Núcleos em Imagens Histológicas H&E

Trabalho de Conclusão de Curso — Sistemas de Informação, FACOM/UFU
Autor: André Machado Silva
Orientador: Prof. Marcelo Zanchetta do Nascimento

Investigação comparativa de funções de perda para segmentação de núcleos celulares em
imagens histológicas coradas com Hematoxilina e Eosina (H&E), utilizando uma arquitetura
U-Net fixa com o módulo ProtoSeg. O objetivo é avaliar como diferentes funções de perda
afetam o desempenho e a estabilidade do treinamento, mantendo o restante do pipeline
constante.

Este repositório estende o pipeline de [Norm_HE_ProtoSeg](https://github.com/LIPAI-Org/Norm_HE_ProtoSeg),
desenvolvido por Freitas (2025).

---

## Notebooks modificados neste trabalho

| Arquivo | Descrição |
|---|---|
| `Unet_ProtoSeg.ipynb` | **Notebook principal.** Contém as funções de perda implementadas, o treinamento e a avaliação. A seleção da função de perda é feita pela variável `LOSS_FN`; o modo do ProtoSeg, por `protoseg_online`. Foi adaptado para: (i) permitir a troca da função de perda, (ii) usar a função selecionada também no modo sem ProtoSeg, e (iii) padronizar o salvamento por `run_id` (função + modo + semente). |
| `ProtoSeg_Camadas.ipynb` | Análise do SA Score por camada da rede. Adaptado para reconstruir a U-Net e carregar os pesos salvos (`load_weights`), em vez de carregar o modelo completo, ler as imagens em RGB e avaliar os modelos com e sem ProtoSeg sobre o mesmo lote de imagens de teste. |
| `Exportar_Mascaras.ipynb` | Criado neste trabalho. Aplica os pesos de cada execução a todo o conjunto de teste e salva as máscaras preditas publicadas no Zenodo. |

## Notebooks do pipeline base, utilizados na preparação dos dados

| Arquivo | Descrição |
|---|---|
| `Crop_Imagens.ipynb` | Recorte das imagens originais, gerando o conjunto `crop_OralEpitheliumDB`. Executado sem modificações. |
| `RCAug_Colab.ipynb` | Aumento de dados, gerando o conjunto `crop_OralEpitheliumDB_aug` utilizado em todos os experimentos. Executado sem modificações. |

## Demais notebooks do pipeline base

Mantidos no repositório para integridade do pipeline original, **sem alterações e sem uso
direto nos experimentos deste trabalho**: `Teste_Modelos.ipynb`,
`Teste_Unitario_Modelos.ipynb` e `Zoom_Imagem.ipynb`.

## Resultados

| Arquivo | Descrição |
|---|---|
| `resultados_experimentos.csv` | Registro das 54 execuções, com as métricas de avaliação por treino. Base para as tabelas apresentadas no texto do TCC. Duas execuções interrompidas da BCE (semente 42, 17/07/2026) foram removidas. |

As máscaras preditas por cada execução sobre o conjunto de teste estão disponíveis no Zenodo:
[https://doi.org/10.5281/zenodo.23602713](https://doi.org/10.5281/zenodo.23602713)

---

## Funções de perda investigadas

Comparação principal (forma padrão):

- **BCE** — *baseline*
- **Focal** — γ = 2,0; α = 0,25
- **Dice** — suavização (*smooth*) = 1,0
- **Tversky** — α = 0,7; β = 0,3
- **Focal Tversky** — α = 0,7; β = 0,3; expoente = 0,75
- **Combo** — BCE + Dice, pesos 1:1

Combinações e variações de aprimoramento:

- **Tversky+BCE** — pesos 1:1
- **Combo ponderada (região)** — 0,5·BCE + 1,0·Dice
- **Combo ponderada (pixel)** — 1,0·BCE + 0,5·Dice
- **Focal+Dice** — pesos 1:1
- **Focal (α = 0,5)** — variação do parâmetro α

## Protocolo experimental

- 200 épocas por treinamento; três sementes (42, 123 e 2024) para as configurações estáveis
  e semente única (42) para as que colapsaram (Dice, Tversky e Focal Tversky).
- Cada função avaliada com e sem o módulo ProtoSeg, totalizando 54 treinamentos.
- Arquitetura, conjunto de dados e demais hiperparâmetros mantidos fixos.
- Métricas: Dice, IoU, acurácia, precisão, recall, Panoptic Quality (PQ) e AJI.

## Ambiente

Python 3.10, TensorFlow 2.10.1 / Keras 2.10, CUDA 11.2, cuDNN 8.1.
GPU NVIDIA RTX 3050 (6 GB). Execução local via conda.

---

## Observação sobre os dados

O conjunto de dados (OralEpitheliumDB), os pesos treinados e as saídas de treinamento
**não** estão versionados neste repositório (ver `.gitignore`), por questões de tamanho e
de licença de uso do conjunto de dados.
