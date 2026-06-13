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
import metrology
import analysis

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
        raw_dir = session_dir / "raw" / raw_sub if "background" not in raw_sub else session_dir / raw_sub
        # Adjust path: background is at session level, not under raw
        # Actually let's check both locations for flexibility
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

        if bg_images:
            bg_path = bg_images[0]  # Usar primeira imagem de background
            logger.info(f"Background: {bg_path.name}")

            bg_img = cv2.imread(str(bg_path), cv2.IMREAD_UNCHANGED)
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

                # Detectar grade e calibrar escala
                try:
                    camera_matrix = None
                    try:
                        camera_matrix, _ = calibrate_module.load_calibration()
                    except FileNotFoundError:
                        pass

                    grid_points = metrology.detect_grid(bg_img)
                    scale = metrology.calibrate_scale(
                        grid_points, view_mode=view,
                        camera_matrix=camera_matrix
                    )

                    # Correção de perspectiva se solicitada
                    if args.perspective_correction:
                        logger.info("Aplicando correção de perspectiva...")
                        bg_img, H = metrology.correct_perspective(bg_img, grid_points)
                        background = bg_img

                except metrology.MetrologyError as e:
                    logger.warning(f"Falha na detecção da grade: {e}")
                    logger.warning("Tentando calibração manual...")
                    try:
                        scale = metrology.manual_scale_calibration(bg_img, view)
                    except metrology.MetrologyError:
                        logger.error("Calibração de escala falhou. Usando pixels.")
                        scale = {
                            "px_per_mm_h": 1.0, "px_per_mm_v": 1.0,
                            "anisotropy": 0.0, "linearity_h": 0.0,
                            "linearity_v": 0.0, "n_points": 0,
                            "view_mode": view
                        }
        else:
            logger.warning(
                f"Nenhuma imagem de background encontrada para vista '{view}'.\n"
                f"  Esperado em: {bg_dir}\n"
                f"  A subtração de fundo não será utilizada."
            )
            scale = {
                "px_per_mm_h": 1.0, "px_per_mm_v": 1.0,
                "anisotropy": 0.0, "linearity_h": 0.0,
                "linearity_v": 0.0, "n_points": 0,
                "view_mode": view
            }

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

                try:
                    # 1. Pré-processamento
                    gray, color = preprocessing.preprocess(
                        str(img_path),
                        save_undistorted=True
                    )

                    # 2. Segmentação
                    seg_results = segmentation.segment(
                        gray, color, img_path.stem,
                        background=background,
                        strategy=args.strategy
                    )

                    # 3. Converter para mm e coletar resultados
                    for metrics_px in seg_results:
                        metrics_mm = metrology.convert_measurements(metrics_px, scale)
                        metrics_mm["sample_id"] = sample_id
                        metrics_mm["session"] = args.session
                        metrics_mm["state"] = state
                        metrics_mm["view_mode"] = view
                        metrics_mm["source_file"] = img_path.name

                        all_results.append(metrics_mm)

                        # 4. Anotar imagem
                        if not args.no_annotate:
                            ann_dir = Path(config.ANNOTATED_DIR) / args.session / view / state
                            ann_path = ann_dir / f"{img_path.stem}_annotated.png"
                            draw_ellipse = metrics_mm.get("circularity", 0) > 0.7
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
        csv_path = Path(config.OUTPUT_DIR) / f"measurements_{args.session}.csv"
        analysis.export_csv(all_results, str(csv_path))
        logger.info(f"\n✓ {len(all_results)} medição(ões) salvas em {csv_path}")

    return all_results


def cmd_analyze(args):
    """Compara peças úmidas vs secas e calcula retração."""
    session_dir = get_session_dir(args.session)

    if not session_dir.exists():
        logger.error(f"Sessão não encontrada: {session_dir}")
        return

    # Primeiro, processar se ainda não foi feito
    measurements_csv = Path(config.OUTPUT_DIR) / f"measurements_{args.session}.csv"

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
            combined_csv = Path(config.OUTPUT_DIR) / f"combined_3d_{args.session}.csv"
            analysis.export_csv(combined, str(combined_csv))

    # Exportar retração
    if all_comparisons:
        shrinkage_csv = Path(config.OUTPUT_DIR) / f"shrinkage_{args.session}.csv"
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


def cmd_full(args):
    """Executa o pipeline completo."""
    logger.info(f"{'═'*60}")
    logger.info(f"PIPELINE COMPLETO — Sessão: {args.session}")
    logger.info(f"{'═'*60}")

    # 1. Converter RAW
    logger.info(f"\n{'═'*60}")
    logger.info("ETAPA 1/4: Conversão RAW → TIFF")
    logger.info(f"{'═'*60}")
    cmd_convert(args)

    # 2. Calibrar (se ainda não calibrado)
    cal_file = Path(config.CALIBRATION_FILE)
    if not cal_file.exists():
        logger.info(f"\n{'═'*60}")
        logger.info("ETAPA 2/4: Calibração da Lente")
        logger.info(f"{'═'*60}")
        cmd_calibrate(args)
    else:
        logger.info(f"\n  Calibração existente: {cal_file}")

    # 3. Processar
    logger.info(f"\n{'═'*60}")
    logger.info("ETAPA 3/4: Processamento (pré-proc + segmentação + metrologia)")
    logger.info(f"{'═'*60}")
    cmd_process(args)

    # 4. Analisar
    logger.info(f"\n{'═'*60}")
    logger.info("ETAPA 4/4: Análise Comparativa")
    logger.info(f"{'═'*60}")
    cmd_analyze(args)

    logger.info(f"\n{'═'*60}")
    logger.info("PIPELINE CONCLUÍDO")
    logger.info(f"{'═'*60}")
    logger.info(f"  Resultados em: {config.OUTPUT_DIR}")
    logger.info(f"  Máscaras em:   {config.MASKS_DIR}")
    logger.info(f"  Anotações em:  {config.ANNOTATED_DIR}")


def cmd_create_session(args):
    """Cria a estrutura de diretórios para uma nova sessão."""
    session_dir = create_session_structure(args.session)
    logger.info(
        f"\nSessão '{args.session}' criada em:\n"
        f"  {session_dir}\n\n"
        f"Próximos passos:\n"
        f"  1. Copie os .NEF para os diretórios apropriados\n"
        f"  2. Execute: python pipeline.py full --session {args.session}"
    )


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
        choices=["background_sub", "lab", "otsu", "adaptive", "auto"],
        default="auto",
        help="Estratégia de segmentação (default: auto)"
    )
    p_proc.add_argument(
        "--perspective-correction", action="store_true",
        help="Aplicar correção de perspectiva via homografia"
    )
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
    p_analyze.add_argument("--perspective-correction", action="store_true")
    p_analyze.add_argument("--no-annotate", action="store_true", default=False)

    # full
    p_full = subparsers.add_parser("full", help="Pipeline completo")
    p_full.add_argument("--session", required=True, help="ID da sessão")
    p_full.add_argument("--view", choices=["top", "side", "both"], default="both")
    p_full.add_argument("--state", choices=["wet", "dry", "both"], default="both")
    p_full.add_argument(
        "--strategy",
        choices=["background_sub", "lab", "otsu", "adaptive", "auto"],
        default="auto"
    )
    p_full.add_argument("--perspective-correction", action="store_true")
    p_full.add_argument("--no-annotate", action="store_true", default=False)

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
