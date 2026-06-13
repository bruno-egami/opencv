# Ceramic Analysis Pipeline

Pipeline Python/OpenCV para metrologia dimensional e análise de retração de corpos de prova cerâmicos fabricados por manufatura aditiva (FDM/DIW).

## Objetivo

Comparar automaticamente as dimensões de peças cerâmicas no estado **úmido** (pós-impressão) e **seco** (pós-secagem), calculando a **retração percentual** em cada dimensão:

$$\Delta L (\%) = \frac{L_{\text{úmido}} - L_{\text{seco}}}{L_{\text{úmido}}} \times 100$$

## Funcionalidades

- 📷 **Conversão RAW** — Arquivos `.NEF` (Nikon) → TIFF 16-bit via `rawpy`
- 🔧 **Calibração de câmera** — Checkerboard 9×6 com `cv2.calibrateCamera`
- 🔍 **Pré-processamento** — Correção de distorção, equalização de histograma, filtro gaussiano
- 🎯 **Segmentação multi-estratégia** — Subtração de fundo, LAB color space, Otsu, adaptativo
- 📏 **Metrologia dual-axis** — Grade gravada a laser no MDF → `px_per_mm` independente em X e Y
- 📐 **Correção de paralaxe** — Para vistas laterais com grade posicionada atrás da peça
- 🧊 **Dimensões 3D** — Combinação de vistas de cima (X-Y) e lateral (X-Z)
- 📊 **Exportação CSV** — Medições, retração percentual e metadados de escala
- 🖼️ **Imagens anotadas** — Contornos e dimensões sobrepostos para validação visual
- ✅ **Validação ImageJ** — Máscaras binárias salvas em PNG para inspeção independente

## Requisitos

- Python 3.10+
- Câmera (utilizado neste projeto uma câmera DSLR Nikon com lente 55mm (sensor 10MP, 2592×3872))
- Base MDF 300×230mm com grade 20×20mm gravada a laser
- Tripé fixo


### Dependências Python

```bash
pip install -r requirements.txt
```

| Pacote | Versão | Uso |
|---|---|---|
| `opencv-python` | ≥ 4.8.0 | Processamento de imagem, calibração, contornos |
| `numpy` | ≥ 1.24.0 | Operações matriciais |
| `rawpy` | ≥ 0.19.0 | Conversão de arquivos RAW (.NEF) |
| `pyyaml` | ≥ 6.0 | Serialização de parâmetros de calibração |
| `pytest` | ≥ 7.0.0 | Testes unitários |

## Estrutura do Projeto

```
ceramic_analysis/
├── pipeline.py               # CLI principal (ponto de entrada)
├── config.py                 # Parâmetros ajustáveis
├── raw_converter.py          # .NEF → TIFF 16-bit
├── calibrate.py              # Calibração de lente (checkerboard)
├── preprocessing.py          # Undistort, equalização, filtro
├── segmentation.py           # Segmentação e detecção de contornos
├── metrology.py              # Grade → escala px/mm
├── analysis.py               # Retração, CSV, anotações
├── requirements.txt
├── data/
│   ├── calibration/          # Imagens do checkerboard
│   │   ├── raw/              #   .NEF originais
│   │   └── converted/        #   TIFF convertidos
│   ├── sessions/             # Sessões de captura (por data)
│   │   └── session_YYYYMMDD/
│   │       ├── background/   #   Fotos da base vazia
│   │       │   ├── top/      #     Vista de cima
│   │       │   └── side/     #     Vista lateral
│   │       ├── raw/          #   .NEF das peças
│   │       │   ├── wet/      #     Úmidas (top/ e side/)
│   │       │   └── dry/      #     Secas (top/ e side/)
│   │       └── converted/    #   TIFF convertidos
│   ├── undistorted/          # Imagens corrigidas
│   └── masks/                # Máscaras binárias (ImageJ)
├── output/
│   ├── calibration_params.yaml
│   ├── results.csv
│   └── annotated/            # Imagens com contornos e dimensões
└── tests/
    └── test_pipeline.py      # 41 testes unitários
```

## Guia Rápido

### 1. Calibração da câmera (executar uma vez)

Coloque 10-15 fotos `.NEF` do checkerboard em `data/calibration/raw/`:

```bash
python pipeline.py calibrate
```

Saída: `output/calibration_params.yaml` com a matriz intrínseca e coeficientes de distorção.  
Meta: RMS de reprojeção < 0.5 px.

### 2. Criar uma sessão de captura

```bash
python pipeline.py create-session --session 20250612
```

Isso cria a estrutura de diretórios. Copie os arquivos `.NEF` para os subdiretórios apropriados.

### 3. Executar o pipeline completo

```bash
python pipeline.py full --session 20250612
```

Isso executa em sequência:
1. **Conversão** `.NEF` → TIFF 16-bit
2. **Detecção da grade** no background → fator de escala (px/mm)
3. **Pré-processamento** → undistort + equalização + filtro
4. **Segmentação** → contornos + métricas em pixels
5. **Metrologia** → conversão para milímetros
6. **Análise** → retração percentual + exportação CSV

### 4. Comandos individuais

```bash
# Apenas converter RAW
python pipeline.py convert --session 20250612

# Apenas processar (sem análise comparativa)
python pipeline.py process --session 20250612 --view top --state wet

# Escolher estratégia de segmentação
python pipeline.py process --session 20250612 --strategy lab

# Análise comparativa (úmido vs seco)
python pipeline.py analyze --session 20250612

# Modo verboso (debug)
python pipeline.py full --session 20250612 --verbose
```

### 5. Verificar resultados

- **Máscaras** em `data/masks/` → abrir no ImageJ e sobrepor com imagem original
- **Imagens anotadas** em `output/annotated/` → contornos + dimensões em mm
- **CSV** em `output/` → `measurements_*.csv` (medições) e `shrinkage_*.csv` (retração)

## Estratégias de Segmentação

| Estratégia | Quando usar | Comando |
|---|---|---|
| `auto` | Padrão — tenta cada estratégia e seleciona a melhor | `--strategy auto` |
| `background_sub` | Qualquer cor de argila (requer foto da base vazia) | `--strategy background_sub` |
| `lab` | Baixo contraste de luminância (argila marrom sobre MDF) | `--strategy lab` |
| `otsu` | Alto contraste (caulim branco sobre MDF) | `--strategy otsu` |
| `adaptive` | Iluminação desigual na bancada | `--strategy adaptive` |

## Ajuste de Parâmetros

Os parâmetros de segmentação podem ser ajustados em `config.py`:

```python
# Se o fundo ruidoso gera falsos positivos:
BG_SUB_THRESHOLD = 35      # ↑ Aumentar (padrão: 25)

# Se partes da peça são cortadas:
BG_SUB_THRESHOLD = 15      # ↓ Diminuir

# Se peças pequenas não são detectadas:
MIN_CONTOUR_AREA_PX = 2000  # ↓ Diminuir (padrão: 5000)

# Se lacunas nas bordas persistem:
MORPH_KERNEL_SIZE = (9, 9)  # ↑ Aumentar kernel
MORPH_ITERATIONS = 3        # ↑ Aumentar iterações
```

## Testes

```bash
python -m pytest tests/test_pipeline.py -v
```

```
41 passed in 0.56s ✅
```

Os testes usam imagens sintéticas e não dependem de hardware ou imagens reais.

## Setup Físico

### Vista de Cima
```
    [Câmera no tripé — apontando para baixo]
              │
              ▼
    ┌─────────────────────┐
    │   Base MDF          │
    │   ┌───┐             │
    │   │   │ ← Peça      │
    │   └───┘             │
    │   ▦▦▦▦▦▦▦▦▦▦▦▦▦▦▦  │ ← Grade 20×20mm
    └─────────────────────┘
```

### Vista Lateral
```
    [Câmera no tripé — apontando horizontalmente]
              │
              ▼
    ┌──────────────┐
    │ MDF vertical │ ← Backdrop com grade (atrás da peça)
    │ (grade)      │
    │    ┌───┐     │
    │    │   │ ← Peça (sobre mesa)
    │    └───┘     │
    └──────────────┘
         ↕ gap (medir com paquímetro → SIDE_GRID_GAP_MM)
```

## Licença

Projeto acadêmico — Mestrado em Engenharia de Materiais.
