# -*- coding: utf-8 -*-
"""
Gerador de relatório HTML para o pipeline de análise de corpos de prova cerâmicos.
"""

import os
import csv
import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Metrologia Cerâmica - Sessão {session}</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --accent-blue: #3b82f6;
            --border-color: #475569;
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
            background-attachment: fixed;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            margin-bottom: 3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 1.5rem;
        }}

        .logo-section h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .logo-section p {{
            color: var(--text-secondary);
            font-size: 1rem;
        }}

        .session-badge {{
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.4);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: #60a5fa;
        }}

        /* Grid Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}

        .card {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .card-title {{
            font-size: 0.9rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            font-weight: 600;
            margin-bottom: 1rem;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .card-value {{
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: var(--text-primary);
        }}

        .card-sub {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }}

        /* Image Comparer Section */
        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.75rem;
            margin-bottom: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .section-title::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: var(--glass-border);
        }}

        .tabs-container {{
            margin-bottom: 3rem;
        }}

        .tab-buttons {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}

        .tab-btn {{
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.2s ease;
            position: relative;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
        }}

        .tab-btn.active {{
            color: #60a5fa;
            background: rgba(59, 130, 246, 0.1);
        }}

        .tab-btn.active::after {{
            content: '';
            position: absolute;
            bottom: -9px;
            left: 0;
            width: 100%;
            height: 3px;
            background-color: #3b82f6;
            border-radius: 3px;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .comparison-layout {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 2rem;
        }}

        .img-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--glass-border);
        }}

        .img-card h3 {{
            padding: 1rem;
            font-size: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid var(--glass-border);
        }}

        .img-container {{
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background-color: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}

        .img-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        /* Table */
        .table-container {{
            overflow-x: auto;
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            background: var(--glass-bg);
            margin-bottom: 3rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}

        th, td {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--glass-border);
        }}

        th {{
            background: rgba(0, 0, 0, 0.2);
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .badge-metric {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 500;
        }}

        .badge-green {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-blue {{
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}

        .footer {{
            text-align: center;
            margin-top: 5rem;
            padding: 2rem 0;
            color: var(--text-secondary);
            font-size: 0.85rem;
            border-top: 1px solid var(--glass-border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-section">
                <h1>Metrologia Cerâmica</h1>
                <p>Relatório Dimensional & Comparação CAD 3D</p>
            </div>
            <div>
                <span class="session-badge">Sessão: {session}</span>
                <span class="session-badge" style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; margin-left: 0.5rem;">Estado: {state_name}</span>
            </div>
        </header>

        <!-- Resumo das Principais Métricas -->
        <section>
            <h2 class="section-title">Resumo Metrológico</h2>
            <div class="metrics-grid">
                <div class="card">
                    <div class="card-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
                        Dimensões da Peça
                    </div>
                    <div class="card-value">{dim_top_len:.2f} × {dim_top_width:.2f} mm</div>
                    <div class="card-sub">Espessura (Vista Lateral): {thickness:.2f} mm</div>
                </div>

                <div class="card">
                    <div class="card-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                        Média de Escala
                    </div>
                    <div class="card-value">{scale_h:.2f} px/mm</div>
                    <div class="card-sub">Anisotropia: {anisotropy:.1%} ({scale_v:.2f} px/mm vertical)</div>
                </div>

                <div class="card">
                    <div class="card-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                        Precisão CAD (IoU)
                    </div>
                    <div class="card-value">{iou:.1%}</div>
                    <div class="card-sub">Desvio Médio CAD: {mean_dev:.2f} mm</div>
                </div>
            </div>
        </section>

        <!-- Comparador de Imagens -->
        <section class="tabs-container">
            <h2 class="section-title">Análise Visual e Desvios</h2>
            <div class="tab-buttons">
                <button class="tab-btn active" onclick="openTab('tab-top')">Vista Superior (Top)</button>
                <button class="tab-btn" onclick="openTab('tab-side')">Vista Lateral (Side)</button>
            </div>

            <!-- Tab Vista Superior -->
            <div id="tab-top" class="tab-content active">
                <div class="comparison-layout">
                    <div class="img-card">
                        <h3>Imagem Anotada (Contorno e Medição)</h3>
                        <div class="img-container">
                            <img src="{annotated_top}" alt="Vista Superior Anotada" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Imagem+N%C3%A3o+Encontrada'">
                        </div>
                    </div>
                    <div class="img-card">
                        <h3>Mapa de Calor de Desvios (Vs Modelo CAD)</h3>
                        <div class="img-container">
                            <img src="{deviation_top}" alt="Mapa de Desvio Top" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Mapa+CAD+N%C3%A3o+Encontrada'">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab Vista Lateral -->
            <div id="tab-side" class="tab-content">
                <div class="comparison-layout">
                    <div class="img-card">
                        <h3>Imagem Anotada (Espessura)</h3>
                        <div class="img-container">
                            <img src="{annotated_side}" alt="Vista Lateral Anotada" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Imagem+N%C3%A3o+Encontrada'">
                        </div>
                    </div>
                    <div class="img-card">
                        <h3>Mapa de Calor de Desvios (Vista Frontal CAD)</h3>
                        <div class="img-container">
                            <img src="{deviation_side}" alt="Mapa de Desvio Side" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Mapa+CAD+N%C3%A3o+Encontrada'">
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Tabela Completa de Medições -->
        <section>
            <h2 class="section-title">Dados Detalhados de Medição</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Vista</th>
                            <th>Métrica</th>
                            <th>Valor Medido</th>
                            <th>Valor CAD</th>
                            <th>Desvio</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Comprimento da Peça (Maior)</td>
                            <td>{dim_top_len:.2f} mm</td>
                            <td>{nominal_len:.2f} mm</td>
                            <td><strong style="color: {dev_len_color};">{dev_len:+.2f} mm</strong> ({dev_len_pct:+.1f}%)</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Largura da Peça (Menor)</td>
                            <td>{dim_top_width:.2f} mm</td>
                            <td>{nominal_width:.2f} mm</td>
                            <td><strong style="color: {dev_width_color};">{dev_width:+.2f} mm</strong> ({dev_width_pct:+.1f}%)</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Área Projetada</td>
                            <td>{area_top:.2f} mm²</td>
                            <td>{nominal_area:.2f} mm²</td>
                            <td>{dev_area_pct:+.2f}%</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>Espessura (Altura)</td>
                            <td>{thickness:.2f} mm</td>
                            <td>{nominal_thick:.2f} mm</td>
                            <td><strong style="color: {dev_thick_color};">{dev_thick:+.2f} mm</strong> ({dev_thick_pct:+.1f}%)</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>Largura (Perfil)</td>
                            <td>{dim_side_w:.2f} mm</td>
                            <td>{nominal_side_w:.2f} mm</td>
                            <td>{dev_side_w:+.2f} mm</td>
                        </tr>
                        {angle_rows_html}
                    </tbody>
                </table>
            </div>
        </section>

        <footer class="footer">
            <p>Gerado automaticamente pelo Pipeline de Metrologia OpenCV - Mestrado em Engenharia de Materiais (IFRS)</p>
            <p style="margin-top: 0.5rem; opacity: 0.5;">Data de geração: 2026</p>
        </footer>
    </div>

    <script>
        function openTab(tabId) {{
            // Esconder todos os conteúdos
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));

            // Remover classe active de todos os botões
            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => btn.classList.remove('active'));

            // Mostrar a aba selecionada e adicionar active no botão
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
        }}
    </script>
</body>
</html>
"""

def read_csv_data(session_id: str) -> tuple:
    output_session_dir = Path(config.OUTPUT_DIR) / session_id
    measurements_path = output_session_dir / f"measurements_{session_id}.csv"
    cad_path = output_session_dir / f"cad_comparison_{session_id}.csv"

    measurements = []
    cad_comparisons = []

    if measurements_path.exists():
        with open(measurements_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            measurements = list(reader)

    if cad_path.exists():
        with open(cad_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cad_comparisons = list(reader)

    return measurements, cad_comparisons

def generate_report(session_id: str):
    measurements, cad_comparisons = read_csv_data(session_id)

    if not measurements:
        print("Nenhuma medição encontrada para gerar o relatório.")
        return

    # Determinar qual estado mostrar (preferir 'dry' se houver, senão 'wet')
    available_states = set(m.get("state") for m in measurements)
    state_to_use = "dry" if "dry" in available_states else ("wet" if "wet" in available_states else "dry")
    state_name = "Seco" if state_to_use == "dry" else "Úmido"

    # Buscar dados do CSV
    top_meas = next((m for m in measurements if m.get("view_mode") == "top" and m.get("state") == state_to_use), {})
    if not top_meas:
        top_meas = next((m for m in measurements if m.get("view_mode") == "top"), {})
        
    side_meas = next((m for m in measurements if m.get("view_mode") == "side" and m.get("state") == state_to_use), {})
    if not side_meas:
        side_meas = next((m for m in measurements if m.get("view_mode") == "side"), {})

    # Determinar estados efetivos encontrados
    state_top = top_meas.get("state", state_to_use) if top_meas else state_to_use
    state_side = side_meas.get("state", state_to_use) if side_meas else state_to_use

    top_cad = next((c for c in cad_comparisons if c.get("cad_view") == "top" and c.get("state") == state_top), {})
    if not top_cad:
        top_cad = next((c for c in cad_comparisons if c.get("cad_view") == "top"), {})
        
    front_cad = next((c for c in cad_comparisons if c.get("cad_view") == "front" and c.get("state") == state_side), {})
    if not front_cad:
        front_cad = next((c for c in cad_comparisons if c.get("cad_view") == "front"), {})

    # Valores padrão se não encontrados
    dim_top_w = float(top_meas.get("min_rect_w_mm", 0.0)) if top_meas else 0.0
    dim_top_h = float(top_meas.get("min_rect_h_mm", 0.0)) if top_meas else 0.0
    area_top = float(top_meas.get("area_mm2", 0.0)) if top_meas else 0.0
    
    thickness = float(side_meas.get("bbox_h_mm", 0.0)) if side_meas else 0.0
    dim_side_w = float(side_meas.get("min_rect_w_mm", 0.0)) if side_meas else 0.0

    scale_h = float(top_meas.get("px_per_mm_h", 1.0)) if top_meas else 1.0
    scale_v = float(top_meas.get("px_per_mm_v", 1.0)) if top_meas else 1.0
    anisotropy = float(top_meas.get("anisotropy", 0.0)) if top_meas else 0.0

    iou = float(top_cad.get("iou", 0.0)) if top_cad else 0.0
    mean_dev = float(top_cad.get("mean_deviation_mm", 0.0)) if top_cad else 0.0

    # Valores padrão para nominal (se não houver comparação CAD)
    nominal_len = 56.10
    nominal_width = 38.00
    nominal_thick = 0.00
    nominal_area = 2131.8
    nominal_side_w = 38.00

    if top_cad:
        cad_w = float(top_cad.get("cad_bbox_w_mm", 0.0))
        cad_h = float(top_cad.get("cad_bbox_h_mm", 0.0))
        if cad_w > 0 and cad_h > 0:
            nominal_len = max(cad_w, cad_h)
            nominal_width = min(cad_w, cad_h)
            nominal_area = float(top_cad.get("cad_area_mm2", nominal_area))
            nominal_side_w = nominal_width
        
        # Tentar extrair espessura nominal do cad_extents_mm (formato WxHxD)
        extents_str = top_cad.get("cad_extents_mm")
        if extents_str:
            try:
                parts = extents_str.replace("×", "x").split("x")
                if len(parts) == 3:
                    nominal_thick = float(parts[2])
            except Exception as e:
                logger.warning(f"Erro ao parsear cad_extents_mm: {e}")

    if front_cad:
        cad_thick = float(front_cad.get("cad_bbox_h_mm", 0.0))
        if cad_thick > 0:
            nominal_thick = cad_thick

    # Ordenar dimensões para que o maior valor medido seja comparado ao comprimento nominal
    # e o menor valor medido seja comparado à largura nominal.
    if dim_top_w > 0 or dim_top_h > 0:
        measured_dims = sorted([dim_top_w, dim_top_h], reverse=True)
        dim_top_len = measured_dims[0]
        dim_top_width = measured_dims[1]
    else:
        dim_top_len = 0.0
        dim_top_width = 0.0

    # Devs
    dev_len = dim_top_len - nominal_len if dim_top_len > 0 else 0.0
    dev_len_pct = (dev_len / nominal_len) * 100 if dim_top_len > 0 and nominal_len > 0 else 0.0
    dev_width = dim_top_width - nominal_width if dim_top_width > 0 else 0.0
    dev_width_pct = (dev_width / nominal_width) * 100 if dim_top_width > 0 and nominal_width > 0 else 0.0
    
    dev_thick = thickness - nominal_thick if thickness > 0 else 0.0
    dev_thick_pct = (dev_thick / nominal_thick) * 100 if thickness > 0 and nominal_thick > 0 else 0.0
    dev_area_pct = ((area_top - nominal_area) / nominal_area) * 100 if area_top > 0 and nominal_area > 0 else 0.0
    
    dev_side_w = dim_side_w - nominal_side_w if dim_side_w > 0 else 0.0

    # Se a vista lateral não foi processada, forçar valores nominais e desvios para 0.0
    if thickness == 0.0:
        nominal_thick = 0.0
        dev_thick = 0.0
        dev_thick_pct = 0.0
    if dim_side_w == 0.0:
        nominal_side_w = 0.0
        dev_side_w = 0.0

    # Ângulos internos dos vértices (se disponíveis)
    angle_0 = float(top_meas.get("corner_angle_0", 0.0)) if top_meas else 0.0
    angle_1 = float(top_meas.get("corner_angle_1", 0.0)) if top_meas else 0.0
    angle_2 = float(top_meas.get("corner_angle_2", 0.0)) if top_meas else 0.0
    angle_3 = float(top_meas.get("corner_angle_3", 0.0)) if top_meas else 0.0

    angle_rows_html = ""
    if angle_0 > 0:
        angle_rows_html = f"""
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Ângulo Vértice 1 (Sup-Esq)</td>
                            <td>{angle_0:.2f}°</td>
                            <td>90.00°</td>
                            <td>{angle_0 - 90.00:+.2f}°</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Ângulo Vértice 2 (Sup-Dir)</td>
                            <td>{angle_1:.2f}°</td>
                            <td>90.00°</td>
                            <td>{angle_1 - 90.00:+.2f}°</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Ângulo Vértice 3 (Inf-Dir)</td>
                            <td>{angle_2:.2f}°</td>
                            <td>90.00°</td>
                            <td>{angle_2 - 90.00:+.2f}°</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Ângulo Vértice 4 (Inf-Esq)</td>
                            <td>{angle_3:.2f}°</td>
                            <td>90.00°</td>
                            <td>{angle_3 - 90.00:+.2f}°</td>
                        </tr>
        """

    def get_color(val):
        return "#10b981" if abs(val) < 1.5 else "#ef4444"

    dev_len_color = get_color(dev_len)
    dev_width_color = get_color(dev_width)
    dev_thick_color = get_color(dev_thick)

    # Imagens (Caminhos relativos para o HTML carregar localmente)
    source_file_top = top_meas.get('source_file', '') if top_meas else ''
    source_file_side = side_meas.get('source_file', '') if side_meas else ''
    
    stem_top = Path(source_file_top).stem if source_file_top else ''
    stem_side = Path(source_file_side).stem if source_file_side else ''
    
    annotated_top = f"annotated/top/{state_top}/{stem_top}_annotated.png" if stem_top else ""
    annotated_side = f"annotated/side/{state_side}/{stem_side}_annotated.png" if stem_side else ""
    
    deviation_top = f"cad_comparison/{top_meas.get('sample_id', '')}_{state_top}_top_deviation.png" if (top_meas and top_cad) else ""
    deviation_side = f"cad_comparison/{side_meas.get('sample_id', '')}_{state_side}_front_deviation.png" if (side_meas and front_cad) else ""

    html_content = HTML_TEMPLATE.format(
        session=session_id,
        state_name=state_name,
        dim_top_len=dim_top_len,
        dim_top_width=dim_top_width,
        area_top=area_top,
        thickness=thickness,
        dim_side_w=dim_side_w,
        scale_h=scale_h,
        scale_v=scale_v,
        anisotropy=anisotropy,
        iou=iou,
        mean_dev=mean_dev,
        dev_len=dev_len,
        dev_len_pct=dev_len_pct,
        dev_width=dev_width,
        dev_width_pct=dev_width_pct,
        dev_thick=dev_thick,
        dev_thick_pct=dev_thick_pct,
        dev_area_pct=dev_area_pct,
        dev_side_w=dev_side_w,
        dev_len_color=dev_len_color,
        dev_width_color=dev_width_color,
        dev_thick_color=dev_thick_color,
        annotated_top=annotated_top,
        annotated_side=annotated_side,
        deviation_top=deviation_top,
        deviation_side=deviation_side,
        nominal_len=nominal_len,
        nominal_width=nominal_width,
        nominal_area=nominal_area,
        nominal_thick=nominal_thick,
        nominal_side_w=nominal_side_w,
        angle_rows_html=angle_rows_html
    )

    output_session_dir = Path(config.OUTPUT_DIR) / session_id
    output_session_dir.mkdir(parents=True, exist_ok=True)
    output_report_path = output_session_dir / f"report_{session_id}.html"
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Relatório gerado com sucesso em: {output_report_path.resolve()}")

if __name__ == "__main__":
    import sys
    sess = sys.argv[1] if len(sys.argv) > 1 else "16-06"
    generate_report(sess)
