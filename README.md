# OpenCV — Fork para Análise Cerâmica

Este repositório é um **fork** da biblioteca original **OpenCV (Open Source Computer Vision Library)** contendo um subprojeto específico para metrologia dimensional e análise de retração de corpos de prova cerâmicos fabricados por manufatura aditiva (FDM/DIW).

O código do pipeline de análise está localizado na pasta [ceramic_analysis/](file:///d:/GitHub/OpenCV/ceramic_analysis/).

---

## Recursos do Projeto Original (OpenCV)

*   **Homepage:** <https://opencv.org>
*   **Documentação:** <https://docs.opencv.org/4.x/>
*   **Repositório Original:** <https://github.com/opencv/opencv>
*   **Licença:** [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

Para contribuir com a biblioteca base OpenCV, leia as [diretrizes de contribuição](https://github.com/opencv/opencv/wiki/How_to_contribute).

---

## Ceramic Analysis Pipeline (Instruções do Fork)

O subprojeto contido no diretório `ceramic_analysis/` foi desenvolvido para comparar automaticamente as dimensões de peças cerâmicas no estado **úmido** (pós-impressão) e **seco** (pós-secagem), calculando a **retração percentual** em cada dimensão:

$$\Delta L = \frac{L_{\text{úmido}} - L_{\text{seco}}}{L_{\text{úmido}}} \times 100$$

### Funcionalidades

- 📷 **Conversão RAW** — Arquivos `.NEF` (Nikon) → TIFF 16-bit via `rawpy`
- 🔧 **Calibração de câmera** — Checkerboard 9×6 com `cv2.calibrateCamera`
- 🔍 **Pré-processamento** — Correção de distorção, equalização de histograma, filtro gaussiano
- 🎯 **Segmentação multi-estratégia** — Subtração de fundo, LAB color space, Otsu, adaptativo (com detecção de inversão para materiais claros/escuros)
- 📏 **Metrologia dual-axis** — Grade gravada a laser no MDF → `px_per_mm` independente em X e Y
- 📐 **Correção de paralaxe** — Retificação física para vistas laterais com grade posicionada atrás da peça
- 🧊 **Dimensões 3D** — Combinação de vistas de cima (X-Y) e lateral (X-Z)
- 📊 **Exportação CSV** — Medições, retração percentual e metadados de escala
- 🖼️ **Imagens anotadas** — Contornos e dimensões sobrepostos para validação visual
- ✅ **Validação ImageJ** — Máscaras binárias salvas em PNG para inspeção independente

### Estrutura da pasta `ceramic_analysis/`

```
ceramic_analysis/
├── pipeline.py               # CLI principal (ponto de entrada)
├── config.py                 # Parâmetros ajustáveis
├── raw_converter.py          # .NEF → TIFF 16-bit
├── calibrate.py              # Calibração de lente (checkerboard)
├── preprocessing.py          # Undistort, equalização, filtro
├── segmentation.py           # Segmentação (com auto-inversão e ordenação)
├── metrology.py              # Grade → escala px/mm + paralaxe
├── analysis.py               # Retração, CSV, anotações
├── requirements.txt
├── data/                     # Imagens do checkerboard, background e sessões
└── tests/
    └── test_pipeline.py      # 44 testes unitários
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
Isso cria a estrutura de diretórios em `data/sessions/session_20250612/`. Copie os arquivos `.NEF` para as subpastas apropriadas (`background/`, `raw/wet/`, `raw/dry/`).

#### 3. Executar o pipeline completo
```bash
python pipeline.py full --session 20250612
```
Este comando executa a conversão RAW, detecção de escala, pré-processamento, segmentação das peças, metrologia em mm e cálculo comparativo de retração.

#### 4. Comandos Individuais
```bash
# Apenas processar sem análise comparativa
python pipeline.py process --session 20250612 --view top --state wet

# Especificar estratégia de segmentação
python pipeline.py process --session 20250612 --strategy lab

# Comparar úmido vs seco e gerar CSV
python pipeline.py analyze --session 20250612
```

### Testes Unitários

Para garantir a corretude do código e as novas validações matemáticas:
```bash
python -m pytest tests/test_pipeline.py -v
```
Todos os 44 testes unitários utilizam dados sintéticos e não exigem conexões físicas ou imagens reais.

---

## Licença

Este é um projeto acadêmico desenvolvido no âmbito do Mestrado em Engenharia de Materiais do IFRS.

Como este subprojeto está hospedado e utiliza a biblioteca [OpenCV](https://github.com/opencv/opencv), ele adere e faz menção à licença original do projeto: a **[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)**. O código do pipeline desenvolvido nesta pasta é livre para uso científico e acadêmico sob os mesmos termos.
