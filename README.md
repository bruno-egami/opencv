# OpenCV — Fork para Análise Cerâmica

Este repositório é um **fork** da biblioteca original **OpenCV (Open Source Computer Vision Library)** contendo um subprojeto específico para metrologia dimensional e análise de retração de corpos de prova cerâmicos fabricados por manufatura aditiva (FDM/DIW).

O código do pipeline de análise está localizado na pasta [ceramic_analysis/](ceramic_analysis/).

---

## Recursos do Projeto Original (OpenCV)

*   **Homepage:** <https://opencv.org>
*   **Documentação:** <https://docs.opencv.org/4.x/>
*   **Repositório Original:** <https://github.com/opencv/opencv>
*   **Licença:** [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

Para contribuir com a biblioteca base OpenCV, leia as [diretrizes de contribuição](https://github.com/opencv/opencv/wiki/How_to_contribute).

---

## Ceramic Analysis Pipeline (Instruções do Fork)

O subprojeto contido no diretório `ceramic_analysis/` foi desenvolvido para comparar automaticamente as dimensões de peças cerâmicas no estado **úmido** (pós-impressão) e **seco** (pós-secagem), além de permitir a **comparação de fidelidade tridimensional com o modelo CAD de referência (STL e STEP)**, calculando a **retração percentual** e os desvios geométricos.


### Funcionalidades

- 📷 **Conversão RAW** — Arquivos `.NEF` (Nikon) → TIFF 16-bit via `rawpy`
- 🔧 **Calibração de câmera** — Checkerboard 9×6 com `cv2.calibrateCamera`
- 🔍 **Pré-processamento** — Correção de distorção, equalização de histograma, filtro gaussiano
- 🎯 **Segmentação multi-estratégia** — Subtração de fundo, LAB color space, Otsu, adaptativo (com detecção de inversão para materiais claros/escuros)
- 📏 **Metrologia dual-axis** — Grade gravada a laser no MDF → `px_per_mm` independente em X e Y com suporte a fatores de correção multiplicativos (`SCALE_CORRECTION_FACTOR_H/V`) para neutralizar distorções/alongamentos ópticos centrais de lentes de celulares.
- 📐 **Correção de paralaxe** — Retificação física para vistas laterais com grade posicionada atrás da peça
- 🧊 **Dimensões 3D** — Combinação de vistas de cima (X-Y) e lateral (X-Z)
- 🖥️ **Comparação CAD (STL/STEP)** — Suporte nativo a STEP do **Autodesk Inventor** via CadQuery, orientação automática por PCA com fallback para OBB, registro ICP (com suporte a simetria de rotação/translação) e extração de furos/cavidades.
- 🎨 **Mapa de Desvio Visual** — Mapa de calor sobreposto na peça indicando desvios críticos (Vermelho), toleráveis (Amarelo) e ideais (Verde), legenda completa de metrologia.
- 📊 **Exportação CSV** — Medições, retração percentual, desvios CAD e metadados de escala
- 🖼️ **Imagens anotadas** — Contornos e dimensões sobrepostos para validação visual (com filtro inteligente de contorno principal)
- ✅ **Validação ImageJ** — Máscaras binárias salvas em PNG para inspeção independente

### Estrutura da pasta `ceramic_analysis/`

```
ceramic_analysis/
├── pipeline.py               # CLI principal (ponto de entrada)
├── config.py                 # Parâmetros ajustáveis (limite de desvios CAD e fatores de ajuste de escala)
├── cad_compare.py            # Módulo de comparação com modelos CAD (STL/STEP)
├── raw_converter.py          # .NEF → TIFF 16-bit
├── calibrate.py              # Calibração de lente (checkerboard com ajuste fino de grade)
├── preprocessing.py          # Undistort, equalização, filtro
├── segmentation.py           # Segmentação (com auto-inversão e ordenação)
├── metrology.py              # Grade → escala px/mm + paralaxe (orientada à rotação)
├── analysis.py               # Retração, CSV, anotações
├── requirements.txt          # Inclui dependências de CAD (trimesh, cadquery, shapely, scipy)
├── data/                     # Imagens do checkerboard, background e sessões
└── tests/
    └── test_pipeline.py      # 64 testes unitários
```

### Instalação e Preparação

Antes de rodar os comandos, navegue para a pasta `ceramic_analysis/`:

```bash
cd ceramic_analysis
pip install -r requirements.txt
```

### Guia Rápido de Uso

*Sempre execute os comandos CLI de dentro do diretório `ceramic_analysis/`.*

#### 1. Calibração da câmera (executar uma vez)
Coloque 10-15 fotos `.NEF` do checkerboard em `data/calibration/raw/`:
```bash
python pipeline.py calibrate
```
Saída: `output/calibration_params.yaml`. Meta: RMS de reprojeção < 0.5 px.

#### 2. Criar uma sessão de captura
```bash
python pipeline.py create-session --session 20250612
```
Isso cria a estrutura de diretórios em `data/sessions/session_20250612/` incluindo a pasta `cad/`. Copie os arquivos `.NEF` e os arquivos do modelo CAD para suas respectivas pastas.

#### 3. Executar o pipeline completo (com Comparação CAD)
Ao fornecer o modelo CAD, o pipeline rodará automaticamente a comparação geométrica tridimensional na 5ª etapa:
```bash
python pipeline.py full --session 20250612 --cad data/sessions/session_20250612/cad/modelo.stl
```

#### 4. Comandos Individuais e CAD
```bash
# Apenas processar sem análise comparativa
python pipeline.py process --session 20250612 --view top --state wet

# Comparar as fotos processadas de uma sessão com um modelo CAD (STL ou STEP)
python pipeline.py cad-compare --session 20250612 --cad data/sessions/session_20250612/cad/peca.step --view top front

# Rodar cad-compare em lote com limite de tolerância de desvio customizado (ex: 1.5mm)
python pipeline.py cad-compare --session 20250612 --cad data/sessions/session_20250612/cad/modelo.stl --view all --tolerance 1.5
```

#### 5. Ajuste Fino de Escala (Distorção de Celulares)
Caso a lente do celular ou o pós-processamento interno deformem as dimensões centrais da peça em relação à grade externa, ajuste as variáveis em `config.py`:
```python
SCALE_CORRECTION_FACTOR_H = 1.0381  # Ajuste horizontal (ex: 53.0mm -> 51.0mm)
SCALE_CORRECTION_FACTOR_V = 1.0330  # Ajuste vertical (ex: 39.2mm -> 38.0mm)
```
Esses fatores ajustam o fator de escala `px_per_mm` individualmente nos eixos e corrigem as métricas de forma rotacionalmente orientada.

### Testes Unitários

Para garantir a corretude do código e as novas validações matemáticas e de geometria CAD:
```bash
python -m pytest tests/test_pipeline.py -v
```
Todos os 64 testes unitários utilizam dados sintéticos e não exigem conexões físicas ou imagens reais.

---

## Licença

Este é um projeto acadêmico desenvolvido no âmbito do Mestrado em Engenharia de Materiais do IFRS.

Como este subprojeto está hospedado e utiliza a biblioteca [OpenCV](https://github.com/opencv/opencv), ele adere e faz menção à licença original do projeto: a **[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)**. O código do pipeline desenvolvido nesta pasta é livre para uso científico e acadêmico sob os mesmos termos.
