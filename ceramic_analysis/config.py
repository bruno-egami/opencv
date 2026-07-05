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
CHECKERBOARD_SIZE = (14, 10)  # (colunas, linhas) de cantos internos
SQUARE_SIZE_MM = 20.0        # Tamanho real de cada quadrado em mm
MIN_CALIBRATION_IMAGES = 3   # Mínimo de imagens com checkerboard detectado

# ══════════════════════════════════════════════════════════════════════════════
# BLOCO PADRÃO DE CALIBRAÇÃO (referência de escala coplanar)
# Bloco posicionado ao lado da peça com face no mesmo plano da face a medir.
# Padrão xadrez assimétrico (9×8 quadrados) impresso na face exposta.
# ══════════════════════════════════════════════════════════════════════════════
CALIB_BLOCK_PATTERN_SIZE = (8, 7)   # (colunas, linhas) de cantos INTERNOS
                                     # Para 9×8 quadrados → 8×7 interseções
CALIB_BLOCK_SQUARE_SIZE_MM = 6.0    # Tamanho de cada quadrado em mm
CALIB_BLOCK_BORDER_MM = 3.0         # Borda branca ao redor do padrão (mm)
# Dimensões externas do bloco (calculadas):
# Largura = 9 × 6 + 2 × 3 = 60 mm
# Altura  = 8 × 6 + 2 × 3 = 54 mm

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
#   "auto"           → Tenta na ordem: background_sub → lab_b → lab → otsu
#                       Avalia qualidade da máscara em cada passo
SEGMENTATION_STRATEGY = "auto"

# ── Tipo de Material ──
# "Argila" (default) ativa a segmentação focada em bordas por conta das manchas
# "Termoplástico" mantém a prioridade no contraste de cores.
MATERIAL_TYPE = "Argila"

# ── Subtração de fundo ──
# Pixels com |imagem - background| > BG_SUB_THRESHOLD são considerados "objeto".
# ↑ Aumentar (ex: 35-50) se o fundo ruidoso gera falsos positivos
# ↓ Diminuir (ex: 15-20) se partes da peça são cortadas na máscara
BG_SUB_THRESHOLD = 25

# Normalizar brilho entre a foto com peça e a base vazia antes de subtrair
BG_SUB_NORMALIZE_BRIGHTNESS = True

# ── Segmentação LAB Canal B (peças amarelas/creme) ──
# Fator multiplicador do desvio padrão para o threshold do canal B.
# Valores menores capturam mais bordas; maiores são mais restritivos.
LAB_B_SIGMA_FACTOR = 2.0

# ── Segmentação baseada em sombra (shadow_band) ──
# delta_low: quanto mais escuro que o MDF o pixel deve ser para ser considerado sombra
SHADOW_BAND_DELTA_LOW = 15
# Erosão física em mm para encolher o contorno em direção à borda real da peça (default: 0.25 mm)
SHADOW_BAND_EROSION_MM = 0.25
# Erosão física assimétrica apenas na base da peça em mm para puxar a borda inferior para cima (default: 0.4 mm)
SHADOW_BAND_BOTTOM_SHIFT_MM = 0.4
# Tamanho da janela de suavização do contorno (média móvel, deve ser ímpar, default: 19)
SHADOW_BAND_SMOOTH_WINDOW = 19
# Habilitar fechamento convexo do contorno (desabilitado para seguir a sinuosidade dos cantos)
SHADOW_BAND_CONVEX_HULL = False
# Limiar mínimo de cinza para ignorar a mesa preta externa (que tem cinza ~40)
SHADOW_BAND_MIN_GRAY = 70

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

# ── Burr Shaver (Abertura Morfológica para raspar rebarbas) ──
BURR_SHAVER_ENABLED = False
BURR_SHAVER_SIZE = 201

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
# Contornos individuais que cobrem mais de 10% da área da imagem são descartados (ex: MDF de fundo)
MAX_CONTOUR_AREA_PROPORTION = 0.10


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


# ══════════════════════════════════════════════════════════════════════════════
# AJUSTE MANUAL INTERATIVO
# ══════════════════════════════════════════════════════════════════════════════
INTERACTIVE_WINDOW_WIDTH = 1200     # Largura da janela de ajuste (px)
INTERACTIVE_VERTEX_RADIUS = 8       # Raio dos círculos dos vértices (px)
INTERACTIVE_SNAP_DISTANCE = 15      # Distância máxima para "grudar" em um vértice (px)
INTERACTIVE_CALIBRATION = True      # Ativar calibração interativa de grade/MDF
INTERACTIVE_ROI_SELECTION = True    # Ativar seleção interativa da Região de Interesse (ROI)

# Os fatores de ajuste fino de escala não são mais necessários na calibração coplanar
# pois a face de calibração está no mesmo plano da peça a ser medida.




