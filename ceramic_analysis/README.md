# Ceramic Analysis Pipeline

Este subprojeto foi desenvolvido para comparar automaticamente as dimensões de peças cerâmicas no estado **seco** e **úmido**, além de permitir a **comparação de fidelidade tridimensional com o modelo CAD de referência (STL e STEP)**, calculando a **retração percentual** e os desvios geométricos.

---

## Funcionalidades

- 📷 **Conversão RAW** — Arquivos `.NEF` (Nikon) → TIFF 16-bit via `rawpy`
- 🔧 **Calibração de câmera** — Checkerboard 9×6 com `cv2.calibrateCamera`
- 🔍 **Pré-processamento** — Correção de distorção, equalização de histograma, filtro gaussiano
- 🎯 **Segmentação multi-estratégia** — Subtração de fundo, LAB color space, Otsu, adaptativo (com detecção de inversão para materiais claros/escuros)
- 📏 **Metrologia Coplanar por Bloco Padrão** — Calibração por imagem utilizando um bloco físico padrão assimétrico de `9×8` quadrados (`6mm` de lado e `3mm` de borda branca, dimensões externas `60×54mm`) posicionado coplanar à face da peça cerâmica. Isso elimina erros de escala causados por profundidade e paralaxe.
- 📐 **Interface Gráfica Interativa** — Ajuste manual fino das interseções do bloco de calibração e do contorno da peça segmentada:
  - **Proporção Dinâmica (Aspect Ratio)**: Garante que maximizar ou redimensionar a janela OpenCV não cause distorções geométricas na imagem da peça.
  - **Filtro de Contraste Integrado**: Tecla **`[C]`** ativa/desativa em tempo real um realce de contraste (CLAHE no espaço LAB) para auxiliar na visualização e refinamento das bordas.
- 🧊 **Dimensões 3D** — Combinação de vistas de cima (X-Y) e lateral (X-Z)
- 🖥️ **Comparação CAD (STL/STEP)** — Suporte nativo a STEP do **Autodesk Inventor** via CadQuery, orientação automática por PCA com fallback para OBB, registro ICP (com suporte a simetria de rotação/translação) e extração de furos/cavidades.
- 🎨 **Mapa de Desvio Visual** — Mapa de calor sobreposto na peça indicando desvios críticos (Vermelho), toleráveis (Amarelo) e ideais (Verde), legenda completa de metrologia.
- 📊 **Exportação CSV** — Medições, retração percentual, desvios CAD e metadados de escala
- 🖼️ **Imagens Anotadas com Cotas** — Desenho de cotas gráficas de engenharia (linhas de cota, extensão, ticks de 45° e o valor medido em mm) diretamente sobre a imagem final no perímetro do bounding box da peça, garantindo saídas autoexplicativas e profissionais.
- ✅ **Validação ImageJ** — Máscaras binárias salvas em PNG para inspeção independente

---

## Estrutura do Subprojeto

```
ceramic_analysis/
├── pipeline.py               # CLI principal (ponto de entrada)
├── config.py                 # Parâmetros ajustáveis (padrão do bloco e limites de tolerância CAD)
├── cad_compare.py            # Módulo de comparação com modelos CAD (STL/STEP)
├── raw_converter.py          # .NEF → TIFF 16-bit
├── calibrate.py              # Calibração de lente (checkerboard original)
├── preprocessing.py          # Undistort, equalização, filtro
├── segmentation.py           # Segmentação (com auto-inversão e ordenação)
├── metrology.py              # Detecção e calibração de escala via bloco padrão coplanar
├── analysis.py               # Retração, CSV, anotações de cotas e metadados
├── requirements.txt          # Dependências do projeto (opencv, cadquery, shapely, scipy, trimesh)
├── data/                     # Imagens do checkerboard, background e sessões
└── tests/
    └── test_pipeline.py      # 61 testes unitários
```

---

## Instalação e Preparação

Navegue para esta pasta e instale os pacotes necessários:

```bash
cd ceramic_analysis
pip install -r requirements.txt
```

---

## Guia Rápido de Uso

*Sempre execute os comandos CLI de dentro deste diretório.*

### 1. Calibração da câmera (executar uma vez)
Coloque 10-15 fotos `.NEF` do checkerboard de calibração em `data/calibration/raw/`:
```bash
python pipeline.py calibrate
```
Saída: `output/calibration_params.yaml`. Meta: RMS de reprojeção < 0.5 px.

### 2. Criar uma sessão de captura
```bash
python pipeline.py create-session --session 20250612
```
Isso cria a estrutura de diretórios em `data/sessions/session_20250612/` incluindo a pasta `cad/`. Copie os arquivos `.NEF` e os arquivos do modelo CAD para suas respectivas pastas.

### 3. Executar o pipeline completo (com Comparação CAD)
Ao fornecer o modelo CAD, o pipeline rodará automaticamente a comparação geométrica tridimensional na 5ª etapa:
```bash
python pipeline.py full --session 20250612 --cad data/sessions/session_20250612/cad/modelo.stl
```

### 4. Comandos Individuais e CAD
```bash
# Apenas processar sem análise comparativa
python pipeline.py process --session 20250612 --view top --state wet

# Comparar as fotos processadas de uma sessão com um modelo CAD (STL ou STEP)
python pipeline.py cad-compare --session 20250612 --cad data/sessions/session_20250612/cad/peca.step --view top front

# Rodar cad-compare em lote com limite de tolerância de desvio customizado (ex: 1.5mm)
python pipeline.py cad-compare --session 20250612 --cad data/sessions/session_20250612/cad/modelo.stl --view all --tolerance 1.5
```

---

## Testes Unitários

Para garantir a corretude do código e as novas validações matemáticas e de geometria CAD:
```bash
python -m pytest tests/test_pipeline.py -v
```
Todos os 61 testes unitários utilizam dados sintéticos e passam sem exigir imagens reais ou conexões físicas.
