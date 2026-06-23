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
- 🎯 **Segmentação multi-estratégia** — Subtração de fundo, LAB color space, Otsu e extração de bordas paramétricas (`segment_edges`) para isolar peças complexas que sujam a base (ex: pó de **Argila** e Caulim sobre o MDF).
- 📏 **Metrologia Coplanar por Bloco Padrão** — Calibração por imagem utilizando um bloco físico padrão assimétrico de `9×8` quadrados (`6mm` de lado e `3mm` de borda branca, dimensões externas `60×54mm`) posicionado coplanar à face da peça cerâmica. Isso elimina erros de escala causados por profundidade e paralaxe.
- 📐 **Interface Gráfica Interativa** — Ajuste manual fino das interseções do bloco de calibração e do contorno da peça segmentada:
  - **Seletor de Material**: Seleção entre "Argila" e "Termoplástico" para calibrar automaticamente a sensibilidade da segmentação contra manchas na base.
  - **Proporção Dinâmica (Aspect Ratio)**: Garante que maximizar ou redimensionar a janela OpenCV não cause distorções geométricas.
  - **Filtro de Contraste Integrado**: Tecla **`[C]`** ativa/desativa um realce de contraste (CLAHE no espaço LAB) para auxiliar na visualização.
- 🧊 **Dimensões 3D** — Combinação de vistas de cima (X-Y) e lateral (X-Z)
- 🖥️ **Comparação CAD (STL/STEP)** — Suporte nativo a STEP do **Autodesk Inventor** via CadQuery, orientação automática por PCA com fallback para OBB, registro ICP (com suporte a simetria de rotação/translação) e extração de furos/cavidades.
- 🎨 **Mapa de Desvio Visual** — Mapa de calor sobreposto na peça indicando desvios críticos (Vermelho), toleráveis (Amarelo) e ideais (Verde), legenda completa de metrologia.
- 📊 **Exportação CSV** — Medições completas, retração percentual, desvios CAD, metadados de escala e cotas transversais relativas extraídas a **20%, 50% e 80%** de comprimento e largura.
- 🖼️ **Imagens Anotadas com Cotas** — Desenho de cotas gráficas de engenharia (linhas de cota, extensão, ticks de 45° e o valor medido em mm) diretamente sobre a imagem final no perímetro do bounding box da peça e marcações nas seções de 20%, 50% e 80%, garantindo saídas autoexplicativas e profissionais.
- 📄 **Relatório HTML Automático** — Geração de relatórios visuais completos contendo imagens comparativas, metadados, medições de retração e mapas de desvio CAD integrados em uma página interativa.
- ✅ **Validação ImageJ** — Máscaras binárias salvas em PNG para inspeção independente

### Estrutura da pasta `ceramic_analysis/`

```
ceramic_analysis/
├── gui.py                    # Interface Gráfica Interativa Principal
├── pipeline.py               # Motor CLI (usado pelo GUI)
├── config.py                 # Parâmetros ajustáveis (padrão do bloco e limites de tolerância CAD)
├── cad_compare.py            # Módulo de comparação com modelos CAD (STL/STEP)
├── raw_converter.py          # .NEF → TIFF 16-bit
├── calibrate.py              # Calibração de lente (checkerboard original)
├── preprocessing.py          # Undistort, equalização, filtro
├── segmentation.py           # Segmentação (com auto-inversão e ordenação)
├── metrology.py              # Detecção e calibração de escala via bloco padrão coplanar
├── analysis.py               # Retração, CSV, anotações de cotas e metadados
├── generate_report.py        # Geração do relatório visual em HTML da sessão
├── requirements.txt          # Dependências do projeto (opencv, cadquery, shapely, scipy, trimesh)
├── data/                     # Imagens do checkerboard, background e sessões
└── tests/
    └── test_pipeline.py      # 61 testes unitários
```

### Instalação e Preparação

Antes de rodar o aplicativo, navegue para a pasta do subprojeto e instale os pacotes:

```bash
cd ceramic_analysis
pip install -r requirements.txt
```

### Guia Rápido de Uso

A ferramenta agora opera através de uma **Interface Gráfica de Usuário (GUI)** moderna e intuitiva, que gerencia todo o fluxo de trabalho sem necessidade de comandos de terminal.

Para iniciar o aplicativo, certifique-se de estar na pasta do projeto e execute:
```bash
python gui.py
```

#### Fluxo de Trabalho na GUI:

1. **Calibração da Lente**
   - Coloque de 10 a 15 fotos `.NEF` do checkerboard em `data/calibration/raw/`.
   - Clique em **"Calibrar Lente"** no painel de ferramentas. O resultado será salvo em `output/calibration_params.yaml`.

2. **Criação de Sessão**
   - Digite o nome da sessão no campo de texto (ex: `caulim_cilindro`).
   - Clique em **"Criar Nova Sessão"**. O aplicativo criará automaticamente a estrutura de pastas correta em `data/sessions/session_caulim_cilindro/`.
   - Arraste suas fotos `.NEF` secas/úmidas e os arquivos CAD (se houver) para dentro das novas pastas geradas.

3. **Configuração de Processamento**
   - **Seletor de Material**: Escolha entre "Termoplástico" e "Argila" para adaptar o algoritmo matemático de isolamento das bordas.
   - Selecione opcionalmente um modelo 3D (STL ou STEP) através do botão de busca, caso queira comparar a contração dimensional contra o CAD original.

4. **Processamento e Análise**
   - **"Converter RAW"**: Transforma os `.NEF` em `.TIFF` brutos.
   - **"Processar"**: Ativa a janela interativa onde você marcará com o mouse o centro da peça e os cantos do bloco padrão de calibração. A régua de contraste pode ser ativada na tecla `[C]`.
   - **"Analisar Retração" / "Comparar com CAD"**: Comparam o estado seco vs úmido e geram os relatórios e os mapas de calor de desvios.
   - **"Rodar Completo (Automático)"**: Executa todas as etapas acima em sequência contínua com base nos arquivos disponíveis na sessão.

5. **Visualizar Relatórios**
   - Clique em **"Gerar Relatório HTML"** para consolidar imagens visuais e planilhas em uma página de navegador pronta para publicação acadêmica.

### Testes Unitários

Para garantir a corretude do código e as novas validações matemáticas e de geometria CAD:
```bash
python -m pytest tests/test_pipeline.py -v
```
Todos os 61 testes unitários utilizam dados sintéticos e passam sem exigir imagens reais ou conexões físicas.

---

## Licença

Este é um projeto acadêmico desenvolvido no âmbito do Mestrado em Engenharia de Materiais do IFRS.

Como este subprojeto está hospedado e utiliza a biblioteca [OpenCV](https://github.com/opencv/opencv), ele adere e faz menção à licença original do projeto: a **[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)**. O código do pipeline desenvolvido nesta pasta é livre para uso científico e acadêmico sob os mesmos termos.
