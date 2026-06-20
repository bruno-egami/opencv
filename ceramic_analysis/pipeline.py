# -*- coding: utf-8 -*-
"""
Script principal do pipeline de análise de corpos de prova cerâmicos.

Orquestra todos os módulos do pipeline via interface de linha de comando (CLI).

Uso:
    python pipeline.py convert    --session SESSION_ID
    python pipeline.py calibrate
    python pipeline.py process    --session SESSION_ID --view top
    python pipeline.py analyze    --session SESSION_ID
    python pipeline.py full       --session SESSION_ID

Exemplos:
    # Converter todos os .NEF de uma sessão para TIFF
    python pipeline.py convert --session 20250612

    # Calibrar a lente (executar uma vez)
    python pipeline.py calibrate

    # Processar peças úmidas e secas (vista de cima)
    python pipeline.py process --session 20250612 --view top --state both

    # Comparar úmido vs seco e gerar CSV
    python pipeline.py analyze --session 20250612

    # Pipeline completo (converter + processar + analisar)
    python pipeline.py full --session 20250612
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from glob import glob

import cv2
import numpy as np

# Adicionar diretório do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import raw_converter
import calibrate as calibrate_module
import preprocessing
import segmentation
import interactive
import metrology
import analysis
import cad_compare

logger = logging.getLogger("ceramic_analysis")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    """Configura o sistema de logging."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def get_session_dir(session_id: str) -> Path:
    """Retorna o diretório de uma sessão, criando se necessário."""
    session_dir = Path(config.SESSIONS_DIR) / f"session_{session_id}"
    return session_dir


def create_session_structure(session_id: str) -> Path:
    """Cria a estrutura de diretórios de uma sessão."""
    session_dir = get_session_dir(session_id)

    dirs = [
        session_dir / "cad",
        session_dir / "background" / "top",
        session_dir / "background" / "side",
        session_dir / "raw" / "wet" / "top",
        session_dir / "raw" / "wet" / "side",
        session_dir / "raw" / "dry" / "top",
        session_dir / "raw" / "dry" / "side",
        session_dir / "converted" / "wet" / "top",
        session_dir / "converted" / "wet" / "side",
        session_dir / "converted" / "dry" / "top",
        session_dir / "converted" / "dry" / "side",
        session_dir / "converted" / "background" / "top",
        session_dir / "converted" / "background" / "side",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Estrutura da sessão criada: {session_dir}")
    return session_dir


def find_images(directory: str, extensions: list = None) -> list:
    """Encontra todas as imagens em um diretório."""
    if extensions is None:
        extensions = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg", "*.bmp"]

    directory = Path(directory)
    if not directory.exists():
        return []

    files = []
    for ext in extensions:
        files.extend(directory.glob(ext))

    return sorted(files)


def find_raw_files(directory: str) -> list:
    """Encontra todos os arquivos RAW em um diretório."""
    directory = Path(directory)
    if not directory.exists():
        return []

    files = []
    for ext in config.RAW_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))

    return sorted(files)


def extract_sample_id(filename: str) -> str:
    """Extrai o sample_id do nome do arquivo (stem sem sufixos de processamento)."""
    stem = Path(filename).stem
    # Remover sufixos comuns de processamento
    for suffix in ["_undistorted", "_mask", "_annotated"]:
        stem = stem.replace(suffix, "")
    return stem


# ──────────────────────────────────────────────────────────────────────────────
# Comandos do pipeline
# ──────────────────────────────────────────────────────────────────────────────

def cmd_convert(args):
    """Converte arquivos .NEF para TIFF 16-bit."""
    session_dir = get_session_dir(args.session)

    if not session_dir.exists():
        create_session_structure(args.session)
        logger.info(
            f"Sessão '{args.session}' criada. Coloque os arquivos .NEF em:\n"
            f"  Background top:  {session_dir / 'raw' / 'background' / 'top'}\n"
            f"  Background side: {session_dir / 'raw' / 'background' / 'side'}\n"
            f"  Peças úmidas:    {session_dir / 'raw' / 'wet' / 'top|side'}\n"
            f"  Peças secas:     {session_dir / 'raw' / 'dry' / 'top|side'}\n"
            f"Depois execute novamente: python pipeline.py convert --session {args.session}"
        )
        return

    total_converted = 0

    # Converter cada subdiretório com .NEF
    raw_dirs = [
        ("background/top", "converted/background/top"),
        ("background/side", "converted/background/side"),
        ("wet/top", "converted/wet/top"),
        ("wet/side", "converted/wet/side"),
        ("dry/top", "converted/dry/top"),
        ("dry/side", "converted/dry/side"),
    ]

    for raw_sub, conv_sub in raw_dirs:
        # Ajusta caminhos para background (no nível da sessão) e peças (sob raw/)
        possible_raw_dirs = [
            session_dir / "raw" / raw_sub,
            session_dir / raw_sub,
        ]

        for raw_dir in possible_raw_dirs:
            raw_files = find_raw_files(str(raw_dir))
            if raw_files:
                conv_dir = session_dir / conv_sub
                logger.info(f"\n{'─'*50}")
                logger.info(f"Convertendo {raw_sub}: {len(raw_files)} arquivo(s)")
                converted = raw_converter.convert_batch(str(raw_dir), str(conv_dir))
                total_converted += len(converted)
                break

    if total_converted == 0:
        logger.warning(
            f"Nenhum arquivo .NEF encontrado na sessão '{args.session}'.\n"
            f"Verifique se os arquivos estão nos diretórios corretos."
        )
    else:
        logger.info(f"\n{'='*50}")
        logger.info(f"Total convertido: {total_converted} arquivo(s)")


def cmd_calibrate(args):
    """Executa a calibração da câmera via checkerboard."""
    # Verificar se há imagens de calibração
    cal_dir = Path(config.CALIBRATION_CONVERTED_DIR)

    if not cal_dir.exists() or not find_images(str(cal_dir)):
        # Tentar converter RAW de calibração primeiro
        raw_cal_dir = Path(config.CALIBRATION_RAW_DIR)
        if raw_cal_dir.exists() and find_raw_files(str(raw_cal_dir)):
            logger.info("Convertendo imagens RAW de calibração...")
            raw_converter.convert_batch(
                str(raw_cal_dir), str(cal_dir)
            )
        else:
            logger.error(
                f"Nenhuma imagem de calibração encontrada.\n"
                f"Coloque imagens do checkerboard em:\n"
                f"  RAW: {config.CALIBRATION_RAW_DIR}\n"
                f"  Ou convertidas: {config.CALIBRATION_CONVERTED_DIR}"
            )
            return

    debug_dir = str(Path(config.OUTPUT_DIR) / "calibration_debug") if args.verbose else None

    try:
        mtx, dist, rms = calibrate_module.run_calibration(
            images_dir=str(cal_dir),
            debug_dir=debug_dir
        )
        logger.info(f"\n✓ Calibração concluída com sucesso (RMS: {rms:.4f} px)")
    except calibrate_module.CalibrationError as e:
        logger.error(f"Falha na calibração: {e}")


def cmd_process(args):
    """Processa imagens de uma sessão (pré-processamento + segmentação + metrologia)."""
    session_dir = get_session_dir(args.session)

    if not session_dir.exists():
        logger.error(f"Sessão não encontrada: {session_dir}")
        return

    views = []
    if args.view in ("top", "both"):
        views.append("top")
    if args.view in ("side", "both"):
        views.append("side")

    states = []
    if args.state in ("wet", "both"):
        states.append("wet")
    if args.state in ("dry", "both"):
        states.append("dry")

    all_results = []

    for view in views:
        logger.info(f"\n{'═'*60}")
        logger.info(f"PROCESSANDO VISTA: {view.upper()}")
        logger.info(f"{'═'*60}")

        # Carregar e processar background para esta vista
        bg_dir = session_dir / "converted" / "background" / view
        bg_images = find_images(str(bg_dir))

        background = None
        scale = None
        H = None
        grid_mask = None

        if bg_images:
            bg_path = bg_images[0]  # Usar primeira imagem de background
            logger.info(f"Background: {bg_path.name}")

            ext = Path(bg_path).suffix.lower()
            flags = cv2.IMREAD_UNCHANGED if ext in [".tiff", ".tif"] else cv2.IMREAD_COLOR
            bg_img = cv2.imread(str(bg_path), flags)
            if bg_img is not None:
                if bg_img.dtype == np.uint16:
                    bg_img = preprocessing.normalize_16bit_to_8bit(bg_img)

                # Aplicar undistort ao background também
                try:
                    mtx, dist = calibrate_module.load_calibration()
                    bg_img = preprocessing.undistort_image(bg_img, mtx, dist)
                except FileNotFoundError:
                    pass

                background = bg_img

                # O background é apenas carregado e undistorted. A calibração de escala
                # agora é feita diretamente nas fotos da peça usando o bloco padrão coplanar.
                pass
        else:
            logger.warning(
                f"Nenhuma imagem de background encontrada para vista '{view}'.\n"
                f"  Esperado em: {bg_dir}\n"
                f"  A subtração de fundo não será utilizada."
            )

        # Processar cada estado (wet/dry)
        for state in states:
            logger.info(f"\n{'─'*50}")
            logger.info(f"Estado: {state.upper()} | Vista: {view.upper()}")
            logger.info(f"{'─'*50}")

            img_dir = session_dir / "converted" / state / view
            images = find_images(str(img_dir))

            if not images:
                logger.warning(f"Nenhuma imagem encontrada em: {img_dir}")
                continue

            for img_path in images:
                logger.info(f"\n  Processando: {img_path.name}")
                sample_id = extract_sample_id(img_path.name)
                corners = None

                try:
                    # Desativar equalização se usar estratégia que depende de cor/luminância original
                    equalize = args.strategy in ["otsu", "adaptive"]
                    # 1. Pré-processamento
                    gray, color = preprocessing.preprocess(
                        str(img_path),
                        save_undistorted=True,
                        equalize=equalize
                    )

                    # 1b. Calibração de escala via bloco padrão coplanar na própria imagem
                    try:
                        auto_corners = metrology.detect_calibration_block(color)
                        
                        import sys
                        is_testing = "pytest" in sys.modules
                        logger.info(
                            f"  [Escala] INTERACTIVE_CALIBRATION={getattr(config, 'INTERACTIVE_CALIBRATION', True)}, "
                            f"is_testing={is_testing}, auto_corners_found={auto_corners is not None}"
                        )
                        if getattr(config, "INTERACTIVE_CALIBRATION", True) and not is_testing:
                            corners = interactive.validate_calibration_block_grid(
                                color, auto_corners,
                                pattern_size=config.CALIB_BLOCK_PATTERN_SIZE,
                                cache_key=img_path.name
                            )
                            if corners is None:
                                raise metrology.MetrologyError("Calibração de bloco rejeitada/cancelada pelo usuário.")
                        else:
                            if auto_corners is None:
                                raise metrology.MetrologyError("Bloco de calibração não detectado automaticamente.")
                            corners = auto_corners
                        
                        scale = metrology.calibrate_scale_from_block(corners)
                    except metrology.MetrologyError as e:
                        logger.warning(f"Calibração de escala falhou para {img_path.name}: {e}")
                        logger.warning("Usando pixels como unidade padrão (1.0 px/mm)")
                        scale = {
                            "px_per_mm_h": 1.0, "px_per_mm_v": 1.0,
                            "anisotropy": 0.0, "linearity_h": 0.0,
                            "linearity_v": 0.0, "n_points": 0,
                            "view_mode": view
                        }

                    # 2. Coleta de seed points (peça + MDF) para segmentação
                    seed_point = None
                    mdf_point = None
                    if getattr(config, "INTERACTIVE_CALIBRATION", True) and not is_testing:
                        logger.info(f"  [Seed] Solicitando identificação da peça e fundo...")
                        seed_point, mdf_point = interactive.get_seed_points(
                            color,
                            window_title=f"Identificar Peca e Fundo - {img_path.name}"
                        )
                        if seed_point is not None and mdf_point is not None:
                            logger.info(f"  [Seed] Peça=({seed_point[0]},{seed_point[1]}), MDF=({mdf_point[0]},{mdf_point[1]})")
                        else:
                            logger.warning("  [Seed] Seed points não fornecidos. Segmentação sem seeds.")

                    # 3. Segmentação
                    seg_results = segmentation.segment(
                        gray, color, img_path.stem,
                        background=background,
                        strategy=args.strategy,
                        calibration_corners=corners,
                        seed_point=seed_point,
                        mdf_point=mdf_point
                    )

                    # Ajuste manual interativo do contorno principal
                    import sys
                    is_testing = "pytest" in sys.modules
                    logger.info(
                        f"  [Contorno] INTERACTIVE_CALIBRATION={getattr(config, 'INTERACTIVE_CALIBRATION', True)}, "
                        f"is_testing={is_testing}"
                    )
                    if len(seg_results) > 0 and getattr(config, "INTERACTIVE_CALIBRATION", True) and not is_testing:
                        primary_metrics = seg_results[0]
                        adjusted_contour, was_adjusted = interactive.adjust_contour(
                            color,
                            primary_metrics["contour"],
                            window_title=f"Ajuste Manual - {img_path.name}"
                        )
                        if was_adjusted:
                            logger.info(f"  → Contorno ajustado manualmente para {img_path.name}")
                            # Recalcular métricas para o contorno ajustado
                            new_metrics = segmentation.extract_contour_metrics(adjusted_contour)
                            new_metrics["contour_index"] = primary_metrics.get("contour_index", 0)
                            new_metrics["image_name"] = primary_metrics.get("image_name", img_path.stem)
                            seg_results[0] = new_metrics

                    # 3. Converter para mm e coletar resultados
                    for metrics_px in seg_results:
                        metrics_mm = metrology.convert_measurements(metrics_px, scale)
                        
                        # Se houver múltiplos contornos, adicionar sufixo para evitar colisão de chaves
                        if len(seg_results) > 1:
                            metrics_mm["sample_id"] = f"{sample_id}_{metrics_px['contour_index']}"
                        else:
                            metrics_mm["sample_id"] = sample_id
                            
                        metrics_mm["session"] = args.session
                        metrics_mm["state"] = state
                        metrics_mm["view_mode"] = view
                        metrics_mm["source_file"] = img_path.name

                        all_results.append(metrics_mm)

                        # 4. Anotar imagem (apenas para o contorno principal de maior score)
                        if not args.no_annotate and metrics_px == seg_results[0]:
                            ann_dir = Path(config.OUTPUT_DIR) / args.session / "annotated" / view / state
                            ann_path = ann_dir / f"{img_path.stem}_annotated.png"
                            draw_ellipse = metrics_mm.get("circularity", 0) > 0.80
                            analysis.annotate_image(
                                color, metrics_mm, str(ann_path),
                                scale=scale,
                                draw_ellipse=draw_ellipse
                            )

                except (segmentation.SegmentationError, Exception) as e:
                    logger.error(f"  ✗ Falha ao processar {img_path.name}: {e}")
                    continue

    # Exportar CSV com medições individuais
    if all_results:
        csv_dir = Path(config.OUTPUT_DIR) / args.session
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"measurements_{args.session}.csv"
        analysis.export_csv(all_results, str(csv_path))
        logger.info(f"\n✓ {len(all_results)} medição(ões) salvas em {csv_path}")

        try:
            import generate_report
            generate_report.generate_report(args.session)
        except Exception as report_err:
            logger.warning(f"Não foi possível gerar o relatório HTML automaticamente: {report_err}")

    return all_results


def cmd_analyze(args):
    """Compara peças úmidas vs secas e calcula retração."""
    session_dir = get_session_dir(args.session)

    if not session_dir.exists():
        logger.error(f"Sessão não encontrada: {session_dir}")
        return

    # Primeiro, processar se ainda não foi feito
    measurements_csv = Path(config.OUTPUT_DIR) / args.session / f"measurements_{args.session}.csv"

    if not measurements_csv.exists():
        logger.info("Medições não encontradas. Executando processamento primeiro...")
        results = cmd_process(args)
    else:
        # Carregar do CSV
        import csv
        results = []
        with open(measurements_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Converter valores numéricos
                for key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
                results.append(row)

    if not results:
        logger.error("Nenhuma medição disponível para análise.")
        return

    # Separar por estado e vista
    views = set(r.get("view_mode", "top") for r in results)

    all_comparisons = []

    for view in views:
        view_results = [r for r in results if r.get("view_mode") == view]

        wet_results = [r for r in view_results if r.get("state") == "wet"]
        dry_results = [r for r in view_results if r.get("state") == "dry"]

        if wet_results and dry_results:
            logger.info(f"\n{'═'*60}")
            logger.info(f"ANÁLISE DE RETRAÇÃO — Vista: {view.upper()}")
            logger.info(f"{'═'*60}")

            comparisons = analysis.compare_specimens(wet_results, dry_results)
            all_comparisons.extend(comparisons)
        else:
            logger.info(
                f"Vista '{view}': {len(wet_results)} úmida(s), "
                f"{len(dry_results)} seca(s)"
            )

    # Combinar vistas se ambas disponíveis
    top_results = [r for r in results if r.get("view_mode") == "top"]
    side_results = [r for r in results if r.get("view_mode") == "side"]

    if top_results and side_results:
        logger.info(f"\n{'═'*60}")
        logger.info("COMBINANDO VISTAS → DIMENSÕES 3D")
        logger.info(f"{'═'*60}")
        combined = analysis.combine_views(top_results, side_results)
        if combined:
            combined_csv = Path(config.OUTPUT_DIR) / args.session / f"combined_3d_{args.session}.csv"
            analysis.export_csv(combined, str(combined_csv))

    # Exportar retração
    if all_comparisons:
        shrinkage_csv = Path(config.OUTPUT_DIR) / args.session / f"shrinkage_{args.session}.csv"
        analysis.export_csv(
            all_comparisons, str(shrinkage_csv),
            columns=analysis.CSV_SHRINKAGE_COLUMNS
        )
        logger.info(f"\n✓ Retração exportada para {shrinkage_csv}")

        # Resumo final
        logger.info(f"\n{'═'*60}")
        logger.info("RESUMO DA RETRAÇÃO")
        logger.info(f"{'═'*60}")
        for comp in all_comparisons:
            sid = comp.get("sample_id", "?")
            sw = comp.get("shrinkage_width_pct", 0)
            sh = comp.get("shrinkage_height_pct", 0)
            sa = comp.get("shrinkage_area_pct", 0)
            logger.info(
                f"  {sid}: ΔLargura={sw:.2f}%  ΔAltura={sh:.2f}%  ΔÁrea={sa:.2f}%"
            )

        try:
            import generate_report
            generate_report.generate_report(args.session)
        except Exception as report_err:
            logger.warning(f"Não foi possível gerar o relatório HTML automaticamente: {report_err}")


def cmd_cad_compare(args):
    """Compara as peças segmentadas da sessão com um modelo CAD de referência."""
    session_dir = get_session_dir(args.session)

    if not session_dir.exists():
        logger.error(f"Sessão não encontrada: {session_dir}")
        return

    # 1. Carrega o modelo CAD
    logger.info(f"Carregando modelo CAD: {args.cad}")
    mesh = cad_compare.load_cad_model(args.cad)

    # 2. Detecta orientação e alinha
    logger.info("Detectando orientação automática do CAD...")
    orientation = cad_compare.detect_orientation(mesh)
    aligned_mesh = cad_compare.align_mesh(mesh, orientation)
    logger.info(f"  Classe da Forma: {orientation['shape_class'].upper()}")
    logger.info(f"  Dimensões (largura, profundidade, altura): {orientation['extents_mm']} mm")
    logger.info(f"  Simetria detectada: {orientation['is_symmetric']} (eixo: {orientation['symmetry_axis']})")

    # 3. Expandir vistas a processar
    views = ["top", "front", "back", "left", "right"] if "all" in args.view else args.view

    # 4. Extrai medições executando o processamento
    logger.info("Processando imagens da sessão para extrair contornos das fotos...")
    proc_args = argparse.Namespace(
        session=args.session,
        view="both",  # Processa ambas as vistas (top e side)
        state=args.state,
        strategy=getattr(args, "strategy", "auto"),
        perspective_correction=getattr(args, "perspective_correction", False),
        no_annotate=getattr(args, "no_annotate", False)
    )
    measurements = cmd_process(proc_args)

    if not measurements:
        logger.error("Nenhuma medição física encontrada na sessão para comparação.")
        return

    all_results = []
    tolerance_mm = args.tolerance if args.tolerance is not None else config.CAD_DEVIATION_TOLERANCE_MM

    # Mapeamento de vistas CAD para as pastas das fotos
    CAD_TO_PHOTO_VIEW = {
        "top": "top",
        "front": "side",
        "back": "side",
        "left": "side",
        "right": "side",
    }

    for view in views:
        logger.info(f"\nComparando vista CAD: {view.upper()}")

        # Projetar CAD
        logger.info("  Projetando silhueta CAD em 2D...")
        cad_ext_mm, cad_holes_mm, cad_mask, cad_bbox = cad_compare.project_to_2d(
            aligned_mesh, view=view
        )

        # Complexidade e reamostragem do CAD
        complexity = cad_compare.compute_shape_complexity(cad_ext_mm)
        spacing = config.CAD_RESAMPLE_SPACING_MM
        if config.CAD_RESAMPLE_AUTO_ADJUST and complexity > 20:
            spacing = max(0.2, spacing * (20.0 / complexity))

        cad_ext_mm = cad_compare.resample_contour(cad_ext_mm, spacing)

        # Filtrar medições da foto correspondentes a esta vista
        photo_view = CAD_TO_PHOTO_VIEW[view]
        matching_measurements = [m for m in measurements if m["view_mode"] == photo_view]

        if not matching_measurements:
            logger.warning(f"  Nenhuma foto de vista '{photo_view}' disponível para comparação com a vista CAD '{view}'")
            continue

        for m in matching_measurements:
            state = m["state"]
            sample_id = m["sample_id"]
            logger.info(f"  Comparando com amostra '{sample_id}' ({state})")

            # Carregar a imagem física
            img_path = session_dir / "converted" / state / photo_view / m["source_file"]
            ext = Path(img_path).suffix.lower()
            flags = cv2.IMREAD_UNCHANGED if ext in [".tiff", ".tif"] else cv2.IMREAD_COLOR
            photo_image = cv2.imread(str(img_path), flags)
            if photo_image is None:
                logger.error(f"  Não foi possível ler imagem: {img_path}")
                continue

            if photo_image.dtype == np.uint16:
                photo_image = preprocessing.normalize_16bit_to_8bit(photo_image)

            # Undistort
            try:
                mtx, dist = calibrate_module.load_calibration()
                photo_image = preprocessing.undistort_image(photo_image, mtx, dist)
            except FileNotFoundError:
                pass

            # Extrair contorno da foto em mm
            photo_contour_px = np.squeeze(m["contour"])
            if len(photo_contour_px.shape) != 2:
                logger.error(f"  Contorno da amostra {sample_id} está no formato incorreto.")
                continue

            px_h = m["px_per_mm_h"]
            px_v = m["px_per_mm_v"]

            photo_contour_mm = np.zeros_like(photo_contour_px, dtype=np.float64)
            photo_contour_mm[:, 0] = photo_contour_px[:, 0] / px_h
            photo_contour_mm[:, 1] = photo_contour_px[:, 1] / px_v

            # Reamostrar contorno da foto
            photo_contour_mm = cad_compare.resample_contour(photo_contour_mm, spacing)

            # Registrar contornos (alinhamento CAD -> Foto)
            reg_method = args.registration if args.registration is not None else config.CAD_REGISTRATION_METHOD
            cad_aligned_mm, transform = cad_compare.register_contours(
                cad_ext_mm, photo_contour_mm,
                method=reg_method,
                shape_class=orientation["shape_class"]
            )

            # Comparar
            # Para simplificar, assumimos que os furos da foto coincidem com a presença no CAD
            metrics = cad_compare.compare_contours(
                cad_aligned_mm, photo_contour_mm, cad_bbox,
                cad_holes=cad_holes_mm,
                shape_class=orientation["shape_class"],
                transform=transform
            )

            # Converter contornos alinhados de volta para pixel para desenho
            cad_contour_px = np.zeros_like(cad_aligned_mm)
            cad_contour_px[:, 0] = cad_aligned_mm[:, 0] * px_h
            cad_contour_px[:, 1] = cad_aligned_mm[:, 1] * px_v

            cad_holes_px = []
            if cad_holes_mm:
                for hole_mm in cad_holes_mm:
                    hole_aligned_mm = cad_compare.apply_registration(hole_mm, transform)
                    hole_px = np.zeros_like(hole_aligned_mm)
                    hole_px[:, 0] = hole_aligned_mm[:, 0] * px_h
                    hole_px[:, 1] = hole_aligned_mm[:, 1] * px_v
                    cad_holes_px.append(hole_px)

            # Salvar imagem de desvio
            out_dir = Path(config.OUTPUT_DIR) / args.session / "cad_comparison"
            out_path = out_dir / f"{sample_id}_{state}_{view}_deviation.png"
            cad_compare.generate_deviation_map(
                photo_image, cad_contour_px, photo_contour_px,
                metrics["per_point_distances_mm"],
                str(out_path),
                px_per_mm_h=px_h,
                px_per_mm_v=px_v,
                tolerance_mm=tolerance_mm,
                cad_holes_px=cad_holes_px,
                metrics=metrics
            )

            # Adicionar aos resultados globais
            res_dict = {
                "sample_id": sample_id,
                "session": args.session,
                "cad_view": view,
                "photo_view": photo_view,
                "state": state,
                "cad_model": os.path.basename(args.cad),
                "shape_class": orientation["shape_class"],
                "shape_complexity": metrics["shape_complexity"],
                "cad_bbox_w_mm": metrics["cad_bbox_w_mm"],
                "cad_bbox_h_mm": metrics["cad_bbox_h_mm"],
                "cad_area_mm2": metrics["cad_area_mm2"],
                "measured_bbox_w_mm": metrics["measured_bbox_w_mm"],
                "measured_bbox_h_mm": metrics["measured_bbox_h_mm"],
                "measured_area_mm2": metrics["measured_area_mm2"],
                "bbox_w_deviation_mm": metrics["bbox_w_deviation_mm"],
                "bbox_h_deviation_mm": metrics["bbox_h_deviation_mm"],
                "bbox_w_deviation_pct": metrics["bbox_w_deviation_pct"],
                "bbox_h_deviation_pct": metrics["bbox_h_deviation_pct"],
                "area_deviation_pct": metrics["area_deviation_pct"],
                "hausdorff_mm": metrics["hausdorff_mm"],
                "mean_deviation_mm": metrics["mean_deviation_mm"],
                "deviation_std_mm": metrics["deviation_std_mm"],
                "deviation_p95_mm": metrics["deviation_p95_mm"],
                "iou": metrics["iou"],
                "registration_method": transform["method_used"],
                "registration_rotation_deg": transform["rotation_deg"],
                "registration_rms_mm": transform["rms_error_mm"],
                "auto_oriented": config.CAD_AUTO_ORIENT,
                "cad_extents_mm": f"{orientation['extents_mm'][0]:.2f}x{orientation['extents_mm'][1]:.2f}x{orientation['extents_mm'][2]:.2f}"
            }

            if "diameter_cad_mm" in metrics:
                res_dict.update({
                    "diameter_cad_mm": metrics["diameter_cad_mm"],
                    "diameter_photo_mm": metrics["diameter_photo_mm"],
                    "diameter_deviation_mm": metrics["diameter_deviation_mm"],
                    "diameter_deviation_pct": metrics["diameter_deviation_pct"],
                    "concentricity_mm": metrics["concentricity_mm"]
                })

            res_dict.update({
                "n_holes_cad": metrics["n_holes_cad"],
                "n_holes_photo": metrics["n_holes_photo"],
                "holes_matched": 1 if metrics["holes_matched"] else 0,
                "holes_iou": metrics.get("holes_iou", metrics["iou"])
            })

            all_results.append(res_dict)

    if all_results:
        csv_dir = Path(config.OUTPUT_DIR) / args.session
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"cad_comparison_{args.session}.csv"
        analysis.export_csv(all_results, str(csv_path), columns=analysis.CSV_CAD_COLUMNS)
        logger.info(f"\n✓ {len(all_results)} resultado(s) da comparacao CAD salvo(s) em {csv_path}")

        try:
            import generate_report
            generate_report.generate_report(args.session)
        except Exception as report_err:
            logger.warning(f"Não foi possível gerar o relatório HTML automaticamente: {report_err}")


def cmd_full(args):
    """Executa o pipeline completo."""
    logger.info(f"{'═'*60}")
    logger.info(f"PIPELINE COMPLETO — Sessão: {args.session}")
    logger.info(f"{'═'*60}")

    # 1. Converter RAW
    logger.info(f"\n{'═'*60}")
    logger.info("ETAPA 1/5: Conversão RAW → TIFF")
    logger.info(f"{'═'*60}")
    cmd_convert(args)

    # 2. Calibrar (se ainda não calibrado)
    cal_file = Path(config.CALIBRATION_FILE)
    if not cal_file.exists():
        logger.info(f"\n{'═'*60}")
        logger.info("ETAPA 2/5: Calibração da Lente")
        logger.info(f"{'═'*60}")
        cmd_calibrate(args)
    else:
        logger.info(f"\n  Calibração existente: {cal_file}")

    # 3. Processar
    logger.info(f"\n{'═'*60}")
    logger.info("ETAPA 3/5: Processamento (pré-proc + segmentação + metrologia)")
    logger.info(f"{'═'*60}")
    cmd_process(args)

    # 4. Analisar
    logger.info(f"\n{'═'*60}")
    logger.info("ETAPA 4/5: Análise Comparativa de Retração")
    logger.info(f"{'═'*60}")
    cmd_analyze(args)

    # 5. Comparação CAD (Opcional)
    if hasattr(args, "cad") and args.cad:
        logger.info(f"\n{'═'*60}")
        logger.info("ETAPA 5/5: Comparação de Peças com Modelo CAD")
        logger.info(f"{'═'*60}")
        cmd_cad_compare(args)

    logger.info(f"\n{'═'*60}")
    logger.info("PIPELINE CONCLUÍDO")
    logger.info(f"{'═'*60}")
    logger.info(f"  Resultados em: {config.OUTPUT_DIR}")
    logger.info(f"  Máscaras em:   {config.MASKS_DIR}")
    logger.info(f"  Anotações em:  {config.ANNOTATED_DIR}")
    if hasattr(args, "cad") and args.cad:
        logger.info(f"  Desvios CAD em: {Path(config.OUTPUT_DIR) / 'cad_comparison'}")

    try:
        import generate_report
        generate_report.generate_report(args.session)
    except Exception as report_err:
        logger.warning(f"Não foi possível gerar o relatório HTML automaticamente: {report_err}")


def cmd_create_session(args):
    """Cria a estrutura de diretórios para uma nova sessão."""
    session_dir = create_session_structure(args.session)
    logger.info(
        f"\nSessão '{args.session}' criada em:\n"
        f"  {session_dir}\n\n"
        f"Próximos passos:\n"
        f"  1. Copie os .NEF para os diretórios apropriados\n"
        f"  2. Se desejar usar comparação CAD, copie o modelo para {session_dir / 'cad'}\n"
        f"  3. Execute: python pipeline.py full --session {args.session} [--cad {session_dir / 'cad' / 'modelo.stl'}]"
    )


def cmd_report(args):
    """Gera o relatório HTML da sessão."""
    import generate_report
    generate_report.generate_report(args.session)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    """Constrói o parser de argumentos da CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline de análise de corpos de prova cerâmicos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python pipeline.py create-session --session 20250612
  python pipeline.py convert --session 20250612
  python pipeline.py calibrate
  python pipeline.py process --session 20250612 --view both --state both
  python pipeline.py analyze --session 20250612
  python pipeline.py full --session 20250612
        """
    )

    # Argumentos globais
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Modo detalhado com logging de debug"
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")

    # create-session
    p_session = subparsers.add_parser(
        "create-session", help="Cria estrutura de diretórios para nova sessão"
    )
    p_session.add_argument("--session", required=True, help="ID da sessão (ex: 20250612)")

    # convert
    p_convert = subparsers.add_parser("convert", help="Converte .NEF para TIFF 16-bit")
    p_convert.add_argument("--session", required=True, help="ID da sessão")

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Calibra a câmera via checkerboard")

    # process
    p_proc = subparsers.add_parser("process", help="Processa imagens de uma sessão")
    p_proc.add_argument("--session", required=True, help="ID da sessão")
    p_proc.add_argument(
        "--view", choices=["top", "side", "both"], default="both",
        help="Vista(s) a processar (default: both)"
    )
    p_proc.add_argument(
        "--state", choices=["wet", "dry", "both"], default="both",
        help="Estado(s) a processar (default: both)"
    )
    p_proc.add_argument(
        "--strategy",
        choices=["grabcut_seeded", "background_sub", "lab", "otsu", "adaptive", "yellow", "auto"],
        default="auto",
        help="Estratégia de segmentação (default: auto)"
    )
# --perspective-correction removed (no longer needed)
    p_proc.add_argument(
        "--no-annotate", action="store_true",
        help="Pular geração de imagens anotadas"
    )

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Compara úmido vs seco e calcula retração")
    p_analyze.add_argument("--session", required=True, help="ID da sessão")
    # Adicionar args que cmd_analyze pode precisar via cmd_process
    p_analyze.add_argument("--view", choices=["top", "side", "both"], default="both")
    p_analyze.add_argument("--state", choices=["wet", "dry", "both"], default="both")
    p_analyze.add_argument("--strategy", default="auto")
# --perspective-correction removed
    p_analyze.add_argument("--no-annotate", action="store_true", default=False)

    # full
    p_full = subparsers.add_parser("full", help="Pipeline completo")
    p_full.add_argument("--session", required=True, help="ID da sessão")
    p_full.add_argument("--view", choices=["top", "side", "both"], default="both")
    p_full.add_argument("--state", choices=["wet", "dry", "both"], default="both")
    p_full.add_argument(
        "--strategy",
        choices=["grabcut_seeded", "background_sub", "lab", "otsu", "adaptive", "yellow", "auto"],
        default="auto"
    )
# --perspective-correction removed
    p_full.add_argument("--no-annotate", action="store_true", default=False)
    p_full.add_argument("--cad", default=None, help="Caminho para o modelo CAD (.stl/.step/.stp) para comparacao")
    p_full.add_argument(
        "--view-cad", nargs="+", dest="view_cad",
        choices=["top", "front", "back", "left", "right", "all"],
        default=["top", "front"],
        help="Vista(s) do CAD a comparar (default: top front)"
    )
    p_full.add_argument("--registration", choices=["icp", "centroid", "bbox_center"], default=None)
    p_full.add_argument("--tolerance", type=float, default=None, help="Tolerancia limite de desvio (mm)")

    # cad-compare
    p_cad = subparsers.add_parser("cad-compare", help="Compara pecas com modelo CAD")
    p_cad.add_argument("--session", required=True, help="ID da sessão")
    p_cad.add_argument("--cad", required=True, help="Caminho para o modelo CAD (.stl/.step/.stp)")
    p_cad.add_argument(
        "--view", nargs="+",
        choices=["top", "front", "back", "left", "right", "all"],
        default=["top", "front"],
        help="Vista(s) a comparar (default: top front)"
    )
    p_cad.add_argument("--state", choices=["wet", "dry", "both"], default="both")
    p_cad.add_argument(
        "--strategy",
        choices=["grabcut_seeded", "background_sub", "lab", "otsu", "adaptive", "yellow", "auto"],
        default="auto"
    )
# --perspective-correction removed
    p_cad.add_argument("--registration", choices=["icp", "centroid", "bbox_center"], default=None)
    p_cad.add_argument("--tolerance", type=float, default=None, help="Tolerancia limite de desvio (mm)")
    p_cad.add_argument("--no-annotate", action="store_true", default=False, help="Nao gerar imagens anotadas com cotas")

    # report
    p_rep = subparsers.add_parser("report", help="Gera relatório HTML da sessão")
    p_rep.add_argument("--session", required=True, help="ID da sessão")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.verbose)

    commands = {
        "create-session": cmd_create_session,
        "convert": cmd_convert,
        "calibrate": cmd_calibrate,
        "process": cmd_process,
        "analyze": cmd_analyze,
        "cad-compare": cmd_cad_compare,
        "report": cmd_report,
        "full": cmd_full,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        try:
            cmd_func(args)
        except KeyboardInterrupt:
            logger.info("\nOperação cancelada pelo usuário.")
        except Exception as e:
            logger.error(f"Erro fatal: {e}", exc_info=args.verbose)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
