# -*- coding: utf-8 -*-
"""
Módulo de calibração da câmera via padrão de checkerboard (tabuleiro de xadrez).

Utiliza cv2.calibrateCamera para calcular a matriz intrínseca da câmera e os
coeficientes de distorção da lente (Nikon 55mm). Os parâmetros são salvos em
um arquivo YAML para uso posterior na correção de distorção (undistort).

A calibração precisa ser executada apenas uma vez (ou quando a lente mudar).

Uso:
    from calibrate import run_calibration, load_calibration

    # Calibrar a partir de imagens do checkerboard
    mtx, dist, rms = run_calibration("data/calibration/converted/")

    # Carregar calibração salva
    mtx, dist = load_calibration("output/calibration_params.yaml")
"""

import os
import logging
from pathlib import Path
from glob import glob

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Exceção para erros durante a calibração."""
    pass


def find_checkerboard(image_path: str, pattern_size: tuple) -> tuple:
    """
    Detecta o padrão de checkerboard em uma imagem.

    Args:
        image_path: Caminho da imagem (TIFF ou outro formato suportado).
        pattern_size: Tupla (colunas, linhas) de cantos internos. Ex: (9, 6).

    Returns:
        Tupla (found, corners, gray):
            - found: True se o checkerboard foi detectado.
            - corners: Array Nx1x2 com as coordenadas dos cantos em pixels,
                       refinadas a nível subpixel. None se não detectado.
            - gray: Imagem em escala de cinza (para verificação de tamanho).

    Raises:
        FileNotFoundError: Se a imagem não existir.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext in ('.jpg', '.jpeg', '.png'):
        img = cv2.imread(image_path)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        logger.error(f"Falha ao carregar imagem: {image_path}")
        return False, None, None

    # Se 16-bit, converter para 8-bit para detecção do checkerboard
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Converter para escala de cinza
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Definir os tamanhos a testar: o padrão e o transposto (caso a câmera esteja rotacionada)
    cols, rows = pattern_size
    transposed_size = (rows, cols)
    
    # Flags com e sem FAST_CHECK
    flags_fast = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    flags_no_fast = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    
    found = False
    corners = None
    
    # Testar diferentes resoluções (escalas) para lidar com imagens de alta resolução
    # Escalas menores (0.1, 0.15, 0.25) são essenciais para imagens de 60MP+ e rodam instantaneamente
    for scale in [0.1, 0.15, 0.25, 0.5, 1.0]:
        if scale == 1.0:
            gray_sc = gray
        else:
            gray_sc = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        # Evitar erros de assert do OpenCV (adaptiveThreshold) se a imagem for pequena demais
        min_dim = max(cols, rows) * 6
        if gray_sc.shape[1] < min_dim or gray_sc.shape[0] < min_dim:
            continue
            
        # 1. Tentar padrão original
        # Tentar com FAST_CHECK
        found_orig, corners_sc = cv2.findChessboardCorners(gray_sc, pattern_size, flags=flags_fast)
        if not found_orig:
            # Tentar sem FAST_CHECK
            found_orig, corners_sc = cv2.findChessboardCorners(gray_sc, pattern_size, flags=flags_no_fast)
            
        if found_orig:
            corners = corners_sc / scale if scale != 1.0 else corners_sc
            found = True
            break
            
        # 2. Se falhar, tentar padrão transposto (rotacionado)
        if transposed_size != pattern_size:
            found_trans, corners_sc = cv2.findChessboardCorners(gray_sc, transposed_size, flags=flags_fast)
            if not found_trans:
                found_trans, corners_sc = cv2.findChessboardCorners(gray_sc, transposed_size, flags=flags_no_fast)
                
            if found_trans:
                # Transpor os cantos para bater com a ordem do pattern_size original
                # corners_sc tem formato (cols * rows, 1, 2) na orientação transposta
                # Reshaping para (cols, rows, 2)
                c_grid = corners_sc.reshape(cols, rows, 2)
                # Transpor dimensões 0 e 1 -> (rows, cols, 2)
                c_transposed = c_grid.transpose(1, 0, 2)
                # Voltar para o formato linear (cols * rows, 1, 2)
                corners_orig_sc = c_transposed.reshape(-1, 1, 2)
                
                corners = corners_orig_sc / scale if scale != 1.0 else corners_orig_sc
                found = True
                break

    # --- INÍCIO DO AJUSTE MANUAL INTERATIVO ---
    import sys
    is_testing = "pytest" in sys.modules
    use_interactive = getattr(config, "INTERACTIVE_CALIBRATION", False) and not is_testing

    if use_interactive:
        import interactive
        cols, rows = pattern_size
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()

        initial_points = None
        if found:
            corners_reshaped = corners.reshape(-1, 2)
            tl = corners_reshaped[0]
            tr = corners_reshaped[cols - 1]
            br = corners_reshaped[cols * rows - 1]
            bl = corners_reshaped[cols * (rows - 1)]
            initial_corners = np.array([tl, tr, br, bl], dtype=np.float32)
            
            selected = interactive.select_checkerboard_corners_manually(
                img_color,
                window_title=f"Validar 4 Cantos do Checkerboard - {os.path.basename(image_path)}",
                initial_corners=initial_corners
            )
            if selected is not None:
                src_pts = np.array([
                    [0, 0],
                    [cols - 1, 0],
                    [cols - 1, rows - 1],
                    [0, rows - 1]
                ], dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, selected)
                grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
                ideal_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
                projected_grid = cv2.perspectiveTransform(ideal_grid.reshape(-1, 1, 2), H)
                initial_points = projected_grid.reshape(-1, 2)
            else:
                initial_points = corners_reshaped.copy()
        else:
            selected = interactive.select_checkerboard_corners_manually(
                img_color,
                window_title=f"Detecao Falhou: Calibracao Manual do Checkerboard - {os.path.basename(image_path)}"
            )
            if selected is not None:
                src_pts = np.array([
                    [0, 0],
                    [cols - 1, 0],
                    [cols - 1, rows - 1],
                    [0, rows - 1]
                ], dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, selected)
                grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
                ideal_grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
                projected_grid = cv2.perspectiveTransform(ideal_grid.reshape(-1, 1, 2), H)
                initial_points = projected_grid.reshape(-1, 2)

        if initial_points is not None:
            tuned_points, confirmed = interactive.fine_tune_checkerboard_grid(
                img_color,
                initial_points,
                pattern_size=pattern_size,
                window_title=f"Ajuste Fino da Malha do Checkerboard - {os.path.basename(image_path)}"
            )
            if confirmed:
                corners = tuned_points.reshape(-1, 1, 2)
                found = True
            elif found:
                # Se cancelou o ajuste fino, mas a detecção automática funcionou, mantemos os originais
                pass
            else:
                corners = None
                found = False
    # --- FIM DO AJUSTE MANUAL INTERATIVO ---

    if found:
        # Refinamento subpixel dos cantos para maior precisão
        # Janela se adapta à resolução da imagem (ex: 64MP precisa de janela maior)
        win_size = max(11, int(gray.shape[1] / 150))
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,   # Máximo de iterações
            0.001  # Precisão desejada (pixels)
        )
        cv2.cornerSubPix(gray, corners, (win_size, win_size), (-1, -1), criteria)

    return found, corners, gray


def run_calibration(
    images_dir: str = None,
    output_path: str = None,
    pattern_size: tuple = None,
    square_size_mm: float = None,
    debug_dir: str = None
) -> tuple:
    """
    Executa a calibração da câmera a partir de imagens do checkerboard.

    Args:
        images_dir: Diretório com imagens do checkerboard (TIFF/PNG/JPG).
                    Default: config.CALIBRATION_CONVERTED_DIR
        output_path: Caminho do arquivo YAML de saída.
                     Default: config.CALIBRATION_FILE
        pattern_size: (colunas, linhas) de cantos internos.
                      Default: config.CHECKERBOARD_SIZE
        square_size_mm: Tamanho de cada quadrado em mm.
                        Default: config.SQUARE_SIZE_MM
        debug_dir: Se definido, salva imagens com cantos desenhados para verificação.

    Returns:
        Tupla (camera_matrix, dist_coefs, rms):
            - camera_matrix: Matriz intrínseca 3x3.
            - dist_coefs: Coeficientes de distorção (k1, k2, p1, p2, k3).
            - rms: Erro RMS de reprojeção em pixels (meta: < 0.5).

    Raises:
        CalibrationError: Se poucas imagens tiverem checkerboard detectado.
    """
    images_dir = images_dir or config.CALIBRATION_CONVERTED_DIR
    output_path = output_path or config.CALIBRATION_FILE
    pattern_size = pattern_size or config.CHECKERBOARD_SIZE
    square_size_mm = square_size_mm or config.SQUARE_SIZE_MM

    images_dir = Path(images_dir)
    output_path = Path(output_path)

    # Encontrar todas as imagens no diretório
    image_extensions = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))
    image_files.sort()

    if not image_files:
        raise CalibrationError(
            f"Nenhuma imagem encontrada em {images_dir}\n"
            f"Extensões procuradas: {image_extensions}"
        )

    logger.info(
        f"Calibração: {len(image_files)} imagens encontradas em {images_dir}"
    )
    logger.info(f"Checkerboard: {pattern_size[0]}×{pattern_size[1]} cantos, "
                f"quadrado = {square_size_mm}mm")

    # Pontos 3D do checkerboard no sistema de coordenadas do tabuleiro
    # (Z = 0, pois o tabuleiro é plano)
    pattern_points = np.zeros(
        (pattern_size[0] * pattern_size[1], 3), np.float32
    )
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size_mm

    obj_points = []   # Pontos 3D no mundo real
    img_points = []   # Pontos 2D na imagem
    image_size = None

    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        logger.info(f"  Processando {img_path.name}...")

        try:
            found, corners, gray = find_checkerboard(
                str(img_path), pattern_size
            )
        except Exception as e:
            logger.warning(f"  ✗ Erro ao processar {img_path.name}: {e}")
            continue

        if gray is None:
            continue

        # Verificar consistência de tamanho
        current_size = (gray.shape[1], gray.shape[0])  # (width, height)
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            logger.warning(
                f"  ✗ Tamanho inconsistente em {img_path.name}: "
                f"{current_size} (esperado {image_size}). Pulando."
            )
            continue

        if found:
            obj_points.append(pattern_points)
            img_points.append(corners.reshape(-1, 2))
            logger.info(f"  ✓ Checkerboard detectado em {img_path.name}")

            # Salvar imagem de debug com cantos desenhados
            if debug_dir:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                debug_path = debug_dir / f"{img_path.stem}_corners.png"
                cv2.imwrite(str(debug_path), vis)
        else:
            logger.warning(
                f"  ✗ Checkerboard NÃO detectado em {img_path.name}. "
                f"Verifique se o tabuleiro {pattern_size[0]}×{pattern_size[1]} "
                f"está inteiramente visível e sem oclusão."
            )

    # Verificar mínimo de imagens válidas
    if len(obj_points) < config.MIN_CALIBRATION_IMAGES:
        raise CalibrationError(
            f"Apenas {len(obj_points)} imagens com checkerboard detectado "
            f"(mínimo: {config.MIN_CALIBRATION_IMAGES}).\n"
            f"Sugestões:\n"
            f"  - Verifique se o checkerboard tem {pattern_size[0]}×{pattern_size[1]} "
            f"cantos internos\n"
            f"  - Certifique-se de que o tabuleiro está totalmente visível\n"
            f"  - Melhore a iluminação (sem reflexos no tabuleiro)\n"
            f"  - Capture mais imagens em diferentes ângulos"
        )

    logger.info(
        f"Calibrando (Primeira Passada) com {len(obj_points)}/{len(image_files)} imagens válidas..."
    )

    # Usar flags estáveis (Fix K2, Fix K3, Zero Tangential) para evitar overfitting em celulares/lentes planas
    calib_flags = cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST

    # Executar calibração da primeira passada
    rms_init, camera_matrix_init, dist_coefs_init, rvecs_init, tvecs_init = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=calib_flags
    )

    # Filtrar imagens com erros de reprojeção individuais altos (ex: devido a OIS/tremores do celular)
    filtered_obj_points = []
    filtered_img_points = []
    
    for i in range(len(obj_points)):
        img_pts_proj, _ = cv2.projectPoints(
            obj_points[i], rvecs_init[i], tvecs_init[i], camera_matrix_init, dist_coefs_init
        )
        img_pts_proj = img_pts_proj.reshape(-1, 2)
        img_pts_actual = img_points[i].reshape(-1, 2)
        err = np.linalg.norm(img_pts_actual - img_pts_proj, axis=1)
        rms_img = np.sqrt(np.mean(err**2))
        
        if rms_img < 2.0:
            filtered_obj_points.append(obj_points[i])
            filtered_img_points.append(img_points[i])
            logger.info(f"  [MANTER] Imagem {i+1}: RMS={rms_img:.4f} px")
        else:
            logger.warning(f"  [DESCARTAR] Imagem {i+1}: RMS={rms_img:.4f} px (alto erro)")

    # Se tivermos imagens suficientes após o filtro, calibrar novamente
    if len(filtered_obj_points) >= config.MIN_CALIBRATION_IMAGES:
        logger.info(f"Refinando calibração com {len(filtered_obj_points)} imagens selecionadas...")
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
            filtered_obj_points, filtered_img_points, image_size, None, None, flags=calib_flags
        )
    else:
        logger.warning("Imagens selecionadas insuficientes para refinamento. Usando todas as imagens.")
        rms, camera_matrix, dist_coefs, rvecs, tvecs = rms_init, camera_matrix_init, dist_coefs_init, rvecs_init, tvecs_init

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTADO DA CALIBRAÇÃO")
    logger.info(f"{'='*60}")
    logger.info(f"RMS de reprojeção: {rms:.4f} px")
    if rms < 0.5:
        logger.info(f"  ✓ Excelente (< 0.5 px)")
    elif rms < 1.0:
        logger.info(f"  ⚠ Aceitável (< 1.0 px), mas considere recalibrar")
    else:
        logger.warning(f"  ✗ Alto (> 1.0 px) — recalibre com melhores imagens")

    logger.info(f"\nMatriz da câmera:\n{camera_matrix}")
    logger.info(f"\nCoeficientes de distorção: {dist_coefs.ravel()}")
    logger.info(
        f"\nDistância focal: fx={camera_matrix[0,0]:.1f} px, "
        f"fy={camera_matrix[1,1]:.1f} px"
    )
    logger.info(
        f"Centro óptico: cx={camera_matrix[0,2]:.1f} px, "
        f"cy={camera_matrix[1,2]:.1f} px"
    )

    # Salvar parâmetros em YAML
    save_calibration(camera_matrix, dist_coefs, rms, image_size, output_path)

    return camera_matrix, dist_coefs, rms


def save_calibration(
    camera_matrix: np.ndarray,
    dist_coefs: np.ndarray,
    rms: float,
    image_size: tuple,
    output_path: str
):
    """
    Salva os parâmetros de calibração em arquivo YAML usando cv2.FileStorage.

    Args:
        camera_matrix: Matriz intrínseca 3x3.
        dist_coefs: Coeficientes de distorção.
        rms: Erro RMS de reprojeção.
        image_size: Tupla (width, height) da imagem.
        output_path: Caminho do arquivo YAML de saída.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(str(output_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coefs", dist_coefs)
    fs.write("rms", rms)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.release()

    logger.info(f"\nParâmetros salvos em: {output_path}")


def load_calibration(yaml_path: str = None) -> tuple:
    """
    Carrega parâmetros de calibração de um arquivo YAML.

    Args:
        yaml_path: Caminho do YAML. Default: config.CALIBRATION_FILE

    Returns:
        Tupla (camera_matrix, dist_coefs).

    Raises:
        FileNotFoundError: Se o arquivo YAML não existir.
        CalibrationError: Se o arquivo estiver corrompido.
    """
    yaml_path = yaml_path or config.CALIBRATION_FILE

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Arquivo de calibração não encontrado: {yaml_path}\n"
            f"Execute primeiro: python pipeline.py calibrate"
        )

    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coefs = fs.getNode("dist_coefs").mat()

    fs.release()

    if camera_matrix is None or dist_coefs is None:
        raise CalibrationError(
            f"Arquivo de calibração corrompido: {yaml_path}\n"
            f"Execute novamente: python pipeline.py calibrate"
        )

    logger.info(f"Calibração carregada de {yaml_path}")
    return camera_matrix, dist_coefs
