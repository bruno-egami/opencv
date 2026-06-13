# -*- coding: utf-8 -*-
"""
Configuração centralizada do pipeline de análise de corpos de prova cerâmicos.

Todos os parâmetros ajustáveis estão neste arquivo. Modifique conforme
as condições de iluminação, geometria do setup e tipo de argila.
"""

import os

# ══════════════════════════════════════════════════════════════════════════════
# CONVERSÃO RAW (.NEF Nikon)
# ══════════════════════════════════════════════════════════════════════════════
RAW_EXTENSIONS = [".nef", ".NEF"]
USE_CAMERA_WB = True        # Usar white balance registrado pela câmera
OUTPUT_BPS = 16              # Bits por canal no TIFF de saída (16 = máxima dinâmica)

# ══════════════════════════════════════════════════════════════════════════════
# CHECKERBOARD (calibração de distorção da lente — executar 1× por lente)
# ══════════════════════════════════════════════════════════════════════════════
CHECKERBOARD_SIZE = (9, 6)   # (colunas, linhas) de cantos internos
SQUARE_SIZE_MM = 20.0        # Tamanho real de cada quadrado em mm
MIN_CALIBRATION_IMAGES = 3   # Mínimo de imagens com checkerboard detectado

# ══════════════════════════════════════════════════════════════════════════════
# GRADE NO MDF (calibração de escala — por sessão)
# Base: 300 × 230 mm, grade 20 × 20 mm gravada a laser (linhas pretas)
# ══════════════════════════════════════════════════════════════════════════════
MDF_WIDTH_MM = 300.0         # Largura da base MDF em mm
MDF_HEIGHT_MM = 230.0        # Altura da base MDF em mm
GRID_SPACING_MM = 20.0       # Espaçamento da grade em mm

# Cálculo das interseções internas:
# Eixo X: 300 / 20 = 15 células → 16 linhas → 14 interseções internas
# Eixo Y: 230 / 20 = 11.5 → 11 células completas (220mm) + 1 parcial (10mm)
#         → 12 linhas completas → 10 interseções internas
# A célula parcial de 10mm na borda Y é ignorada na calibração.
GRID_INTERSECTIONS_X = 14
GRID_INTERSECTIONS_Y = 10

# ══════════════════════════════════════════════════════════════════════════════
# CORREÇÃO DE PARALAXE (vista lateral)
# Quando a grade vertical está posicionada atrás da peça, há uma diferença
# de profundidade que causa erro de escala. Medir o gap com paquímetro 1×.
# Se a grade estiver no mesmo plano da peça, definir como 0.0.
# ══════════════════════════════════════════════════════════════════════════════
SIDE_GRID_GAP_MM = 6.0       # Distância entre a face da peça e a grade (mm)

# ══════════════════════════════════════════════════════════════════════════════
# PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════════════
GAUSSIAN_KERNEL = (5, 5)     # Kernel do filtro gaussiano (deve ser ímpar × ímpar)
GAUSSIAN_SIGMA = 0           # 0 = sigma auto-calculado a partir do tamanho do kernel
CENTER_TOLERANCE = 0.30      # Fração da imagem considerada "zona central" (30%)
                             # A distorção radial é menor no centro da lente 55mm

# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTAÇÃO
# Ajuste estes parâmetros conforme a iluminação da bancada e o tipo de argila.
# ══════════════════════════════════════════════════════════════════════════════

# Estratégia de segmentação:
#   "background_sub" → Usa foto da base vazia como referência (RECOMENDADO)
#                       Funciona para qualquer cor de argila, inclusive marrom sobre MDF
#   "lab"            → Espaço de cor LAB (canais cromáticos A e B)
#                       Bom para baixo contraste de luminância, médio contraste de cor
#   "otsu"           → Limiar automático de Otsu (distribuição bimodal)
#                       Funciona bem para alto contraste (ex: caulim branco sobre MDF)
#   "adaptive"       → Threshold adaptativo (blocos locais)
#                       Útil quando iluminação não é uniforme na bancada
#   "auto"           → Tenta na ordem: background_sub → lab → otsu
#                       Avalia qualidade da máscara em cada passo
SEGMENTATION_STRATEGY = "auto"

# ── Subtração de fundo ──
# Pixels com |imagem - background| > BG_SUB_THRESHOLD são considerados "objeto".
# ↑ Aumentar (ex: 35-50) se o fundo ruidoso gera falsos positivos
# ↓ Diminuir (ex: 15-20) se partes da peça são cortadas na máscara
BG_SUB_THRESHOLD = 25

# ── Threshold adaptativo ──
# Usado quando SEGMENTATION_STRATEGY = "adaptive"
ADAPTIVE_BLOCK_SIZE = 51     # Tamanho do bloco para cálculo da média local (ímpar)
                             # Maior → transição mais suave; menor → mais sensível
ADAPTIVE_C = 10              # Constante subtraída da média local
                             # Maior → menos pixels classificados como objeto

# ── Morfologia (pós-processamento, aplicada após qualquer estratégia) ──
# Fecha lacunas nas bordas causadas por reflexos na superfície da argila úmida.
# Kernel elíptico produz bordas mais naturais que retangular.
MORPH_KERNEL_SIZE = (7, 7)   # Tamanho do kernel (maior → fecha gaps maiores)
MORPH_ITERATIONS = 2         # Número de iterações do fechamento morfológico

# ── Override manual ──
# Definir um valor 0-255 para forçar threshold fixo, ignorando Otsu/adaptativo.
# Usar None para cálculo automático (recomendado na maioria dos casos).
MANUAL_THRESHOLD = None

# ── Filtragem de contornos ──
# Contornos com área menor que este valor (em pixels²) são descartados como ruído.
# Para câmera de 10MP (2592×3872 ≈ 10M px), 5000 px é ~0.05% da imagem.
# ↑ Aumentar se muitos contornos espúrios forem detectados
# ↓ Diminuir se peças pequenas não forem detectadas
MIN_CONTOUR_AREA_PX = 5000

# ══════════════════════════════════════════════════════════════════════════════
# CAMINHOS (relativos à raiz do projeto ceramic_analysis/)
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "calibration", "raw")
CALIBRATION_CONVERTED_DIR = os.path.join(PROJECT_ROOT, "data", "calibration", "converted")
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "data", "sessions")
UNDISTORTED_DIR = os.path.join(PROJECT_ROOT, "data", "undistorted")
MASKS_DIR = os.path.join(PROJECT_ROOT, "data", "masks")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "output", "calibration_params.yaml")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "output", "results.csv")
ANNOTATED_DIR = os.path.join(PROJECT_ROOT, "output", "annotated")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARAÇÃO COM MODELO CAD
# ══════════════════════════════════════════════════════════════════════════════
CAD_MODEL_PATH = None               # Caminho para o modelo (.stl ou .step/.stp)
                                    # Pode ser sobrescrito via CLI: --cad model.stl

# ── Projeção ──
CAD_PROJECTION_PITCH = 0.1         # Resolução da rasterização (mm/pixel)

# ── Tessellation STEP ──
CAD_STEP_TOLERANCE = 0.05          # Tolerância linear para STEP (mm)
CAD_STEP_ANGULAR_TOLERANCE = 0.1   # Tolerância angular para curvas (rad)

# ── Registro (alinhamento) ──
CAD_REGISTRATION_METHOD = "icp"     # "icp", "centroid", "bbox_center"
CAD_ICP_MAX_ITERATIONS = 50        # Iterações para prismas/cilindros
CAD_ICP_MAX_ITERATIONS_ORGANIC = 100  # Iterações para formas orgânicas
CAD_ICP_TOLERANCE = 0.01           # Convergência em mm

# ── Tolerância de desvio ──
CAD_DEVIATION_TOLERANCE_MM = 1.0   # Limiar para cores no mapa de desvio

# ── Orientação automática ──
CAD_AUTO_ORIENT = True
CAD_SYMMETRY_THRESHOLD = 0.05      # Limiar para simetria axial (5%)

# ── Reamostragem de contorno ──
CAD_RESAMPLE_SPACING_MM = 0.5     # Espaçamento base entre pontos
CAD_RESAMPLE_AUTO_ADJUST = True    # Auto-ajustar pela complexidade

# ── Caminhos ──
CAD_DIR = os.path.join(PROJECT_ROOT, "data", "cad")
CAD_COMPARISON_DIR = os.path.join(OUTPUT_DIR, "cad_comparison")

