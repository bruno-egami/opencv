# -*- coding: utf-8 -*-
"""
Gerador de relatório HTML para o pipeline de análise de corpos de prova cerâmicos.
"""

import os
import csv
import json
import logging
import webbrowser
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') # Backend non-interactive
import matplotlib.pyplot as plt

import config

logger = logging.getLogger(__name__)

def _aggregate(measurements_list, key):
    vals = []
    for m in measurements_list:
        val_str = m.get(key, 0.0)
        if val_str == "":
            val_str = 0.0
        val = float(val_str)
        if val > 0:
            vals.append(val)
    if not vals:
        return 0.0, 0.0, 0
    return np.mean(vals), np.std(vals), len(vals)

def _format_stat(mean, std, n, unit="mm"):
    if n > 1:
        return f"{mean:.2f} &plusmn; {std:.2f} {unit}" if unit else f"{mean:.2f} &plusmn; {std:.2f}"
    elif n == 1:
        return f"{mean:.2f} {unit}" if unit else f"{mean:.2f}"
    else:
        return f"0.00 {unit}" if unit else "0.00"


def format_cad_details_html(cad_c, view):
    """Gera um bloco de texto HTML completo com todas as estatísticas de desvio do CAD."""
    try:
        tolerance_mm = config.TOLERANCE_LIMIT
    except AttributeError:
        tolerance_mm = 1.00
        
    mean_dev = float(cad_c.get("mean_deviation_mm", 0))
    dev_std = float(cad_c.get("deviation_std_mm", 0))
    hausdorff = float(cad_c.get("hausdorff_mm", 0))
    iou = float(cad_c.get("iou", 0)) * 100
    complexity = float(cad_c.get("shape_complexity", 0))
    
    html = f"""
    <div style="padding: 1.2rem; border-top: 1px solid var(--border-color); font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6; background: rgba(0,0,0,0.15);">
        <span style="color: {'#60a5fa' if view == 'top' else '#34d399'}; font-weight: 600; display: block; margin-bottom: 0.75rem; font-family: 'Outfit', sans-serif; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em;">Dados de Desvio do Modelo CAD ({'Vista Superior' if view == 'top' else 'Vista Lateral'})</span>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem 1.5rem; margin-bottom: 0.75rem;">
            <div>• Desvio Médio Local: <strong style="color: #f8fafc;">{mean_dev:.3f} mm</strong> (std: {dev_std:.3f} mm)</div>
            <div>• Desvio Máx (Hausdorff): <strong style="color: #f8fafc;">{hausdorff:.3f} mm</strong></div>
            <div>• Tolerância Limite: <strong style="color: #f8fafc;">{tolerance_mm:.2f} mm</strong></div>
            <div>• Coincidência (IoU): <strong style="color: #34d399;">{iou:.1f}%</strong></div>
            <div>• Complexidade da Forma: <strong style="color: #f8fafc;">{complexity:.2f}</strong></div>
        </div>
        <div style="border-top: 1px dashed rgba(255,255,255,0.1); padding-top: 0.75rem; margin-top: 0.5rem; font-size: 0.85rem; line-height: 1.5;">
    """
    
    # Se houver diâmetro
    if "diameter_photo_mm" in cad_c and cad_c.get("diameter_photo_mm") != "":
        dia_cad = float(cad_c.get("diameter_cad_mm", 0))
        dia_photo = float(cad_c.get("diameter_photo_mm", 0))
        dia_dev = float(cad_c.get("diameter_deviation_mm", 0))
        dia_dev_pct = float(cad_c.get("diameter_deviation_pct", 0))
        concentricity = float(cad_c.get("concentricity_mm", 0))
        
        html += f"""
            • Diâmetro (Real vs CAD): <strong style="color: #f8fafc;">{dia_photo:.2f} / {dia_cad:.2f} mm</strong> (Desvio: {dia_dev:+.3f} mm, {dia_dev_pct:+.2f}%)<br>
            • Concentricidade: <strong style="color: #f8fafc;">{concentricity:.3f} mm</strong><br>
        """
    else:
        w_label = "Largura (X)"
        h_label = "Profundidade (Y)" if view == "top" else "Altura (Z)"
        
        cad_w = float(cad_c.get('cad_bbox_w_mm', 0.0) or 0.0)
        cad_h = float(cad_c.get('cad_bbox_h_mm', 0.0) or 0.0)
        meas_w = float(cad_c.get('measured_bbox_w_mm', 0.0) or 0.0)
        meas_h = float(cad_c.get('measured_bbox_h_mm', 0.0) or 0.0)
        
        dev_w = float(cad_c.get('bbox_w_deviation_mm', 0.0) or 0.0)
        dev_w_pct = float(cad_c.get('bbox_w_deviation_pct', 0.0) or 0.0)
        dev_h = float(cad_c.get('bbox_h_deviation_mm', 0.0) or 0.0)
        dev_h_pct = float(cad_c.get('bbox_h_deviation_pct', 0.0) or 0.0)
        
        cad_area = float(cad_c.get('cad_area_mm2', 0.0) or 0.0)
        meas_area = float(cad_c.get('measured_area_mm2', 0.0) or 0.0)
        dev_area_pct = float(cad_c.get('area_deviation_pct', 0.0) or 0.0)
        
        html += f"""
            • {w_label}: CAD: <strong>{cad_w:.2f} mm</strong> | Real: <strong>{meas_w:.2f} mm</strong> | Desvio: <strong style="color: {'#10b981' if dev_w >= 0 else '#ef4444'};">{dev_w:+.2f} mm</strong> ({dev_w_pct:+.1f}%)<br>
            • {h_label}: CAD: <strong>{cad_h:.2f} mm</strong> | Real: <strong>{meas_h:.2f} mm</strong> | Desvio: <strong style="color: {'#10b981' if dev_h >= 0 else '#ef4444'};">{dev_h:+.2f} mm</strong> ({dev_h_pct:+.1f}%)<br>
            • Área Projetada: CAD: <strong>{cad_area:.1f} mm²</strong> | Real: <strong>{meas_area:.1f} mm²</strong> | Desvio: <strong style="color: {'#10b981' if dev_area_pct >= 0 else '#ef4444'};">{dev_area_pct:+.1f}%</strong><br>
        """
        
    if "n_holes_photo" in cad_c and cad_c.get("n_holes_photo") != "" and int(cad_c.get("n_holes_cad", 0) or 0) > 0:
        n_holes_photo = int(cad_c.get("n_holes_photo", 0) or 0)
        n_holes_cad = int(cad_c.get("n_holes_cad", 0) or 0)
        holes_iou = float(cad_c.get("holes_iou", 0.0) or 0.0)
        html += f"""
            • Furos (Foto vs CAD): <strong style="color: #f8fafc;">{n_holes_photo} / {n_holes_cad}</strong> (IoU Furos: {holes_iou:.3f})<br>
        """
        
    html += """
        </div>
    </div>
    """
    return html


def generate_profile_plot(measurements_list, output_path, title, y_label, prefix="cross_width_", nominal_value=None):
    plt.figure(figsize=(10, 5))
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.set_facecolor('#1e293b')
    plt.gcf().patch.set_facecolor('#0f172a')
    
    percentages = list(range(0, 101, 5))
    has_data = False
    
    for i, meas in enumerate(measurements_list):
        vals = []
        valid_pcts = []
        for p in percentages:
            key = f"{prefix}{p}pct_mm"
            val_str = meas.get(key, 0.0)
            if val_str == "":
                val_str = 0.0
            val = float(val_str)
            
            # Forçar os extremos de 0% e 100% a serem plotados (mesmo se forem 0.0)
            if p in [0, 100]:
                vals.append(val)
                valid_pcts.append(p)
            elif val > 0:
                vals.append(val)
                valid_pcts.append(p)
        if vals:
            has_data = True
            label = meas.get("sample_id", f"Peça {i+1}")
            plt.plot(valid_pcts, vals, marker='o', linestyle='-', linewidth=2, markersize=4, label=label)
            
    if not has_data:
        plt.close()
        return None
        
    # Desenhar linha nominal de referência do CAD
    if nominal_value is not None and nominal_value > 0:
        plt.axhline(y=nominal_value, color='#ef4444', linestyle='--', linewidth=2, label=f"Nominal CAD ({nominal_value:.2f} mm)")

    plt.title(title, color='#f8fafc', pad=15)
    plt.xlabel('Posição ao longo da peça (%)', color='#94a3b8')
    plt.ylabel(y_label, color='#94a3b8')
    plt.grid(color='#334155', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc')
    plt.tick_params(colors='#94a3b8')
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['bottom'].set_color('#475569')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    return output_path

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
                <div class="card" style="grid-column: span 2;">
                    <div class="card-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
                        Dimensões da Peça (C × L × E)
                    </div>
                    <div class="card-value" style="font-size: 1.25rem; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;">
                        <span>{dim_top_len}</span>
                        <span style="color: var(--text-secondary);">×</span>
                        <span>{dim_top_width}</span>
                        <span style="color: var(--text-secondary);">×</span>
                        <span>{thickness}</span>
                    </div>
                </div>

                {cad_card_html}
            </div>
            <div style="margin-top: 3rem; color: var(--text-secondary); font-size: 0.85rem; text-align: center; border-top: 1px solid var(--glass-border); padding-top: 1rem;">
                Média de Escala: {scale_h:.2f} px/mm (Anisotropia: {anisotropy:.1%})
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
                    {annotated_top_cards_html}
                    {deviation_top_cards_html}
                </div>
                {profile_top_html}
            </div>

            <!-- Tab Vista Lateral -->
            <div id="tab-side" class="tab-content">
                <div class="comparison-layout">
                    {annotated_side_cards_html}
                    {deviation_side_cards_html}
                </div>
                {profile_side_html}
            </div>
        </section>

        <!-- Tabela Completa de Medições -->
        <section>
            <h2 class="section-title">Dados Detalhados de Medição</h2>
            
            <div style="background-color: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); padding: 1.5rem; margin-bottom: 2rem; border-radius: 12px; backdrop-filter: blur(12px);">
                <h3 style="font-size: 1.05rem; color: #60a5fa; margin-bottom: 0.75rem; font-family: 'Outfit', sans-serif; display: flex; align-items: center; gap: 0.5rem; font-weight: 600;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                    Metodologia de Medição e Perfis Dimensionais
                </h3>
                <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 0.75rem;">
                    <strong>Dimensões Globais:</strong> Os valores principais de <em>Comprimento</em> e <em>Largura da Peça</em> são obtidos pelo Retângulo de Área Mínima (<em>Minimum Area Bounding Box</em>) do contorno. Isso permite medir o tamanho total real independente de rotações da amostra na imagem.
                </p>
                <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.6;">
                    <strong>Geração dos Perfis (0% a 100%):</strong> O sistema traça uma linha média longitudinal ("espinha dorsal") ao longo do eixo principal da peça. Em intervalos regulares de 5%, o algoritmo projeta uma linha vetorial perpendicular a esse eixo. Os pontos onde essa linha intercepta o contorno externo são detectados, e a largura/comprimento local é calculado como a distância direta entre os dois pontos extremos de interseção. Nos limites de 0% e 100%, se a linha de varredura passar por fora da ponta arredondada da peça devido ao acabamento geométrico, o valor é plotado como 0.0 mm para uniformidade gráfica.
                </p>
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Vista</th>
                            <th>Métrica</th>
                            <th>Valor Medido</th>
                            {cad_th_html}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Comprimento da Peça (Lado Maior)</td>
                            <td>{dim_top_len}</td>
                            {cad_len_td_html}
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Largura da Peça (Lado Menor)</td>
                            <td>{dim_top_width}</td>
                            {cad_width_td_html}
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>Área Projetada</td>
                            <td>{area_top}</td>
                            {cad_area_td_html}
                        </tr>
                        {side_rows_html}
                        {angle_rows_html}
                        {cross_section_rows_html}
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

def generate_report(session_id: str, open_browser: bool = False):
    measurements, cad_comparisons = read_csv_data(session_id)

    if not measurements:
        print("Nenhuma medição encontrada para gerar o relatório.")
        return

    # Determinar qual estado mostrar (preferir 'dry' se houver, senão 'wet')
    available_states = set(m.get("state") for m in measurements)
    state_to_use = "dry" if "dry" in available_states else ("wet" if "wet" in available_states else "dry")
    state_name = f"Seco (n={len([m for m in measurements if m.get('view_mode')=='top' and m.get('state')==state_to_use])})" if state_to_use == "dry" else f"Úmido (n={len([m for m in measurements if m.get('view_mode')=='top' and m.get('state')==state_to_use])})"

    # Buscar dados do CSV
    top_meas_list = [m for m in measurements if m.get("view_mode") == "top" and m.get("state") == state_to_use]
    if not top_meas_list:
        top_meas_list = [m for m in measurements if m.get("view_mode") == "top"]
        
    side_meas_list = [m for m in measurements if m.get("view_mode") == "side" and m.get("state") == state_to_use]
    if not side_meas_list:
        side_meas_list = [m for m in measurements if m.get("view_mode") == "side"]

    top_meas = top_meas_list[0] if top_meas_list else {}
    side_meas = side_meas_list[0] if side_meas_list else {}

    # Determinar estados efetivos encontrados
    state_top = top_meas.get("state", state_to_use) if top_meas else state_to_use
    state_side = side_meas.get("state", state_to_use) if side_meas else state_to_use

    top_cad_list = [c for c in cad_comparisons if c.get("cad_view") == "top" and c.get("state") == state_top]
    if not top_cad_list:
        top_cad_list = [c for c in cad_comparisons if c.get("cad_view") == "top"]
        
    front_cad_list = [c for c in cad_comparisons if c.get("cad_view") == "front" and c.get("state") == state_side]
    if not front_cad_list:
        front_cad_list = [c for c in cad_comparisons if c.get("cad_view") == "front"]

    top_cad = top_cad_list[0] if top_cad_list else {}
    front_cad = front_cad_list[0] if front_cad_list else {}

    # Agregação
    # Para cada peça, o comprimento é o maior lado do retângulo e a largura é o menor lado
    top_lengths = []
    top_widths = []
    for m in top_meas_list:
        w_val = m.get("min_rect_w_mm", 0.0)
        h_val = m.get("min_rect_h_mm", 0.0)
        w = float(w_val) if w_val and w_val != "" else 0.0
        h = float(h_val) if h_val and h_val != "" else 0.0
        if w > 0 and h > 0:
            top_lengths.append(max(w, h))
            top_widths.append(min(w, h))
            
    if top_lengths:
        mean_dim_top_len = np.mean(top_lengths)
        std_dim_top_len = np.std(top_lengths)
        n_top = len(top_lengths)
    else:
        mean_dim_top_len = std_dim_top_len = 0.0
        n_top = 0
        
    if top_widths:
        mean_dim_top_width = np.mean(top_widths)
        std_dim_top_width = np.std(top_widths)
    else:
        mean_dim_top_width = std_dim_top_width = 0.0

    mean_area_top, std_area_top, _ = _aggregate(top_meas_list, "area_mm2")
    
    mean_thick, std_thick, n_side = _aggregate(side_meas_list, "bbox_h_mm")
    
    # Comprimento na vista lateral é o maior lado do min_rect
    side_lengths = []
    for m in side_meas_list:
        w_val = m.get("min_rect_w_mm", 0.0)
        h_val = m.get("min_rect_h_mm", 0.0)
        w = float(w_val) if w_val and w_val != "" else 0.0
        h = float(h_val) if h_val and h_val != "" else 0.0
        if w > 0 and h > 0:
            side_lengths.append(max(w, h))
            
    if side_lengths:
        mean_dim_side_w = np.mean(side_lengths)
        std_dim_side_w = np.std(side_lengths)
    else:
        mean_dim_side_w = std_dim_side_w = 0.0

    scale_h = float(top_meas.get("px_per_mm_h", 1.0)) if top_meas else 1.0
    scale_v = float(top_meas.get("px_per_mm_v", 1.0)) if top_meas else 1.0
    anisotropy = float(top_meas.get("anisotropy", 0.0)) if top_meas else 0.0

    mean_iou, _, _ = _aggregate(top_cad_list, "iou")
    iou = mean_iou
    mean_dev_cad, _, _ = _aggregate(top_cad_list, "mean_deviation_mm")
    mean_dev = mean_dev_cad

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
            nominal_side_w = nominal_len
        
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

    # As dimensões já foram ordenadas por peça individualmente acima
    pass

    dim_top_len_str = _format_stat(mean_dim_top_len, std_dim_top_len, n_top, "mm")
    dim_top_width_str = _format_stat(mean_dim_top_width, std_dim_top_width, n_top, "mm")
    area_top_str = _format_stat(mean_area_top, std_area_top, n_top, "mm²")
    thickness_str = _format_stat(mean_thick, std_thick, n_side, "mm")
    dim_side_w_str = _format_stat(mean_dim_side_w, std_dim_side_w, n_side, "mm")

    # Devs
    dev_len = mean_dim_top_len - nominal_len if mean_dim_top_len > 0 else 0.0
    dev_len_pct = (dev_len / nominal_len) * 100 if mean_dim_top_len > 0 and nominal_len > 0 else 0.0
    dev_width = mean_dim_top_width - nominal_width if mean_dim_top_width > 0 else 0.0
    dev_width_pct = (dev_width / nominal_width) * 100 if mean_dim_top_width > 0 and nominal_width > 0 else 0.0
    
    dev_thick = mean_thick - nominal_thick if mean_thick > 0 else 0.0
    dev_thick_pct = (dev_thick / nominal_thick) * 100 if mean_thick > 0 and nominal_thick > 0 else 0.0
    dev_area_pct = ((mean_area_top - nominal_area) / nominal_area) * 100 if mean_area_top > 0 and nominal_area > 0 else 0.0
    
    dev_side_w = mean_dim_side_w - nominal_side_w if mean_dim_side_w > 0 else 0.0

    # Determinar se existe comparação CAD
    has_cad = bool(top_cad) or bool(front_cad)

    # Ângulos internos dos vértices (médias)
    mean_angle_0, _, _ = _aggregate(top_meas_list, "corner_angle_0")
    mean_angle_1, _, _ = _aggregate(top_meas_list, "corner_angle_1")
    mean_angle_2, _, _ = _aggregate(top_meas_list, "corner_angle_2")
    mean_angle_3, _, _ = _aggregate(top_meas_list, "corner_angle_3")
    angle_0 = mean_angle_0
    angle_1 = mean_angle_1
    angle_2 = mean_angle_2
    angle_3 = mean_angle_3

    def get_color(val):
        return "#10b981" if abs(val) < 1.5 else "#ef4444"

    dev_len_color = get_color(dev_len)
    dev_width_color = get_color(dev_width)
    dev_thick_color = get_color(dev_thick)

    # --- Construir HTML condicional: CAD card ---
    if has_cad:
        cad_card_html = f"""<div class="card">
                    <div class="card-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                        Precisão CAD (IoU)
                    </div>
                    <div class="card-value">{iou:.1%}</div>
                    <div class="card-sub">Desvio Médio CAD: {mean_dev:.2f} mm</div>
                </div>"""
    else:
        cad_card_html = ""

    # --- Construir HTML condicional: colunas CAD na tabela ---
    if has_cad:
        cad_th_html = "<th>Valor CAD</th>\n                            <th>Desvio</th>"
        cad_len_td_html = f'<td>{nominal_len:.2f} mm</td>\n                            <td><strong style="color: {dev_len_color};">{dev_len:+.2f} mm</strong> ({dev_len_pct:+.1f}%)</td>'
        cad_width_td_html = f'<td>{nominal_width:.2f} mm</td>\n                            <td><strong style="color: {dev_width_color};">{dev_width:+.2f} mm</strong> ({dev_width_pct:+.1f}%)</td>'
        cad_area_td_html = f'<td>{nominal_area:.2f} mm²</td>\n                            <td>{dev_area_pct:+.2f}%</td>'
    else:
        cad_th_html = ""
        cad_len_td_html = ""
        cad_width_td_html = ""
        cad_area_td_html = ""

    # --- Construir HTML condicional: linhas da vista lateral ---
    side_rows_html = ""
    if mean_thick > 0.0 or mean_dim_side_w > 0.0:
        if has_cad:
            side_rows_html = f"""
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>Espessura (Altura)</td>
                            <td>{thickness_str}</td>
                            <td>{nominal_thick:.2f} mm</td>
                            <td><strong style="color: {dev_thick_color};">{dev_thick:+.2f} mm</strong> ({dev_thick_pct:+.1f}%)</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>Comprimento (Perfil)</td>
                            <td>{dim_side_w_str}</td>
                            <td>{nominal_side_w:.2f} mm</td>
                            <td>{dev_side_w:+.2f} mm</td>
                        </tr>"""
        else:
            side_rows_html = f"""
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>Espessura (Altura)</td>
                            <td>{thickness_str}</td>
                        </tr>
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>Comprimento (Perfil)</td>
                            <td>{dim_side_w_str}</td>
                        </tr>"""

    # --- Construir HTML: ângulos ---
    angle_rows_html = ""
    if angle_0 > 0:
        angle_labels = [
            ("Ângulo Vértice 1 (Sup-Esq)", angle_0),
            ("Ângulo Vértice 2 (Sup-Dir)", angle_1),
            ("Ângulo Vértice 3 (Inf-Dir)", angle_2),
            ("Ângulo Vértice 4 (Inf-Esq)", angle_3),
        ]
        for label, angle_val in angle_labels:
            dev_str = f"{angle_val - 90.00:+.2f}°"
            if has_cad:
                angle_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>{label}</td>
                            <td>{angle_val:.2f}°</td>
                            <td>90.00°</td>
                            <td>{dev_str}</td>
                        </tr>"""
            else:
                angle_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>{label}</td>
                            <td>{angle_val:.2f}° ({dev_str} vs 90°)</td>
                        </tr>"""

    mean_s_angle_0, _, _ = _aggregate(side_meas_list, "corner_angle_0")
    mean_s_angle_1, _, _ = _aggregate(side_meas_list, "corner_angle_1")
    mean_s_angle_2, _, _ = _aggregate(side_meas_list, "corner_angle_2")
    mean_s_angle_3, _, _ = _aggregate(side_meas_list, "corner_angle_3")
    s_angle_0 = mean_s_angle_0
    
    if s_angle_0 > 0:
        angle_labels = [
            ("Ângulo Vértice 1 (Sup-Esq)", mean_s_angle_0),
            ("Ângulo Vértice 2 (Sup-Dir)", mean_s_angle_1),
            ("Ângulo Vértice 3 (Inf-Dir)", mean_s_angle_2),
            ("Ângulo Vértice 4 (Inf-Esq)", mean_s_angle_3),
        ]
        for label, angle_val in angle_labels:
            dev_str = f"{angle_val - 90.00:+.2f}°"
            if has_cad:
                angle_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>{label}</td>
                            <td>{angle_val:.2f}°</td>
                            <td>90.00°</td>
                            <td>{dev_str}</td>
                        </tr>"""
            else:
                angle_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>{label}</td>
                            <td>{angle_val:.2f}° ({dev_str} vs 90°)</td>
                        </tr>"""

    # --- Construir HTML: seções transversais ---
    cross_section_rows_html = ""
    cross_keys = []
    
    # Adicionar raio de quina
    mean_corner, std_corner, n_corner = _aggregate(top_meas_list, "corner_radius_mm")
    if mean_corner > 0:
        val_str = _format_stat(mean_corner, std_corner, n_corner, "mm")
        dev_str = f"{mean_corner - 1.50:+.2f} mm"
        cross_section_rows_html += f"""
            <tr>
                <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                <td>Raio de Quina Médio</td>
                <td>{val_str}</td>
                <td>1.50 mm</td>
                <td>{dev_str}</td>
            </tr>"""

    for pct in [20, 50, 90]:
        cross_keys.append((f"Largura a {pct}%", f"cross_width_{pct}pct_mm"))
    for pct in [20, 50, 90]:
        cross_keys.append((f"Comprimento a {pct}%", f"cross_length_{pct}pct_mm"))

    has_cross = any(top_meas.get(k, 0) for _, k in cross_keys) if top_meas else False
    if has_cross:
        for label, key in cross_keys:
            mean_val, std_val, n_cross = _aggregate(top_meas_list, key)
            if mean_val > 0:
                val_str = _format_stat(mean_val, std_val, n_cross, "mm")
                if has_cad:
                    cad_key = f"cad_{key}"
                    cad_val_str = top_cad.get(cad_key, "")
                    cad_val = float(cad_val_str) if cad_val_str and cad_val_str != "" else 0.0
                    
                    if cad_val > 0:
                        dev = mean_val - cad_val
                        dev_pct = (dev / cad_val) * 100
                        color_dev = get_color(dev)
                        cad_cell = f"{cad_val:.2f} mm"
                        dev_cell = f'<strong style="color: {color_dev};">{dev:+.2f} mm</strong> ({dev_pct:+.1f}%)'
                    else:
                        cad_cell = "—"
                        dev_cell = "—"
                        
                    cross_section_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>{label}</td>
                            <td>{val_str}</td>
                            <td>{cad_cell}</td>
                            <td>{dev_cell}</td>
                        </tr>"""
                else:
                    cross_section_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-blue">Superior (Top)</span></td>
                            <td>{label}</td>
                            <td>{val_str}</td>
                        </tr>"""
                        
    has_cross_side = any(side_meas.get(k, 0) for _, k in cross_keys) if side_meas else False
    if has_cross_side:
        for label, key in cross_keys:
            side_label = label.replace("Largura", "Altura")
            mean_val, std_val, n_cross = _aggregate(side_meas_list, key)
            if mean_val > 0:
                val_str = _format_stat(mean_val, std_val, n_cross, "mm")
                if has_cad:
                    cad_key = f"cad_{key}"
                    cad_val_str = front_cad.get(cad_key, "")
                    cad_val = float(cad_val_str) if cad_val_str and cad_val_str != "" else 0.0
                    
                    if cad_val > 0:
                        dev = mean_val - cad_val
                        dev_pct = (dev / cad_val) * 100
                        color_dev = get_color(dev)
                        cad_cell = f"{cad_val:.2f} mm"
                        dev_cell = f'<strong style="color: {color_dev};">{dev:+.2f} mm</strong> ({dev_pct:+.1f}%)'
                    else:
                        cad_cell = "—"
                        dev_cell = "—"
                        
                    cross_section_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>{side_label}</td>
                            <td>{val_str}</td>
                            <td>{cad_cell}</td>
                            <td>{dev_cell}</td>
                        </tr>"""
                else:
                    cross_section_rows_html += f"""
                        <tr>
                            <td><span class="badge-metric badge-green">Lateral (Side)</span></td>
                            <td>{side_label}</td>
                            <td>{val_str}</td>
                        </tr>"""
 
    # (Color definitions moved up)
 
    # Imagens (Caminhos relativos para o HTML carregar localmente)
    annotated_top_cards_html = ""
    unique_top_sources = list(dict.fromkeys(m.get("source_file", "") for m in top_meas_list if m.get("source_file", "")))
    for src in unique_top_sources:
        stem = Path(src).stem
        img_path = f"annotated/top/{state_top}/{stem}_annotated.png"
        annotated_top_cards_html += f"""
                    <div class="img-card">
                        <h3>Imagem Anotada ({session_id})</h3>
                        <div class="img-container">
                            <img src="{img_path}" alt="Vista Superior Anotada" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Imagem+N%C3%A3o+Encontrada'">
                        </div>
                    </div>"""
    
    annotated_side_cards_html = ""
    unique_side_sources = list(dict.fromkeys(m.get("source_file", "") for m in side_meas_list if m.get("source_file", "")))
    for src in unique_side_sources:
        stem = Path(src).stem
        img_path = f"annotated/side/{state_side}/{stem}_annotated.png"
        annotated_side_cards_html += f"""
                    <div class="img-card">
                        <h3>Imagem Anotada ({session_id})</h3>
                        <div class="img-container">
                            <img src="{img_path}" alt="Vista Lateral Anotada" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Imagem+N%C3%A3o+Encontrada'">
                        </div>
                    </div>"""
    
    deviation_top_cards_html = ""
    for cad_c in top_cad_list:
        sid = cad_c.get("sample_id", "")
        dev_map = f"cad_comparison/{sid}_{state_top}_top_deviation.png"
        details_panel = format_cad_details_html(cad_c, "top")
        deviation_top_cards_html += f"""
                    <div class="img-card">
                        <h3>Mapa de Calor de Desvios ({sid})</h3>
                        <div class="img-container">
                            <img src="{dev_map}" alt="Mapa CAD" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Mapa+CAD+N%C3%A3o+Encontrado'">
                        </div>
                        {details_panel}
                    </div>"""

    deviation_side_cards_html = ""
    for cad_c in front_cad_list:
        sid = cad_c.get("sample_id", "")
        dev_map = f"cad_comparison/{sid}_{state_side}_front_deviation.png"
        details_panel = format_cad_details_html(cad_c, "front")
        deviation_side_cards_html += f"""
                    <div class="img-card">
                        <h3>Mapa de Calor de Desvios Frontal ({sid})</h3>
                        <div class="img-container">
                            <img src="{dev_map}" alt="Mapa CAD" onerror="this.src='https://placehold.co/600x450/1e293b/f8fafc?text=Mapa+CAD+N%C3%A3o+Encontrado'">
                        </div>
                        {details_panel}
                    </div>"""
                    
    # --- Gerar Gráficos de Perfil ---
    output_session_dir = Path(config.OUTPUT_DIR) / session_id
    output_session_dir.mkdir(parents=True, exist_ok=True)
    
    profile_top_html = ""
    if top_meas_list:
        plot_top_path = output_session_dir / f"profile_top_{session_id}.png"
        res_top = generate_profile_plot(top_meas_list, str(plot_top_path), "Perfil de Variação de Largura (Vista Superior)", "Largura (mm)", prefix="cross_width_", nominal_value=nominal_width)
        
        plot_top_len_path = output_session_dir / f"profile_top_len_{session_id}.png"
        res_top_len = generate_profile_plot(top_meas_list, str(plot_top_len_path), "Perfil de Variação de Comprimento (Vista Superior)", "Comprimento (mm)", prefix="cross_length_", nominal_value=nominal_len)
        
        if res_top:
            profile_top_html += f"""
                <div class="img-card" style="margin-top: 2rem;">
                    <h3>Perfil Dimensional (Largura vs Comprimento)</h3>
                    <div class="img-container">
                        <img src="{plot_top_path.name}" alt="Gráfico de Perfil Superior">
                    </div>
                </div>
            """
        if res_top_len:
            profile_top_html += f"""
                <div class="img-card" style="margin-top: 2rem;">
                    <h3>Perfil Dimensional (Comprimento vs Largura)</h3>
                    <div class="img-container">
                        <img src="{plot_top_len_path.name}" alt="Gráfico de Perfil Superior (Comprimento)">
                    </div>
                </div>
            """
            
    profile_side_html = ""
    if side_meas_list:
        plot_side_path = output_session_dir / f"profile_side_{session_id}.png"
        res_side = generate_profile_plot(side_meas_list, str(plot_side_path), "Perfil de Variação de Espessura (Vista Lateral)", "Espessura (mm)", prefix="cross_width_", nominal_value=nominal_thick)
        
        plot_side_len_path = output_session_dir / f"profile_side_len_{session_id}.png"
        res_side_len = generate_profile_plot(side_meas_list, str(plot_side_len_path), "Perfil de Variação de Comprimento/Largura (Vista Lateral)", "Dimensão (mm)", prefix="cross_length_", nominal_value=nominal_side_w)
        
        if res_side:
            profile_side_html += f"""
                <div class="img-card" style="margin-top: 2rem;">
                    <h3>Perfil Dimensional (Espessura)</h3>
                    <div class="img-container">
                        <img src="{plot_side_path.name}" alt="Gráfico de Perfil Lateral (Espessura)">
                    </div>
                </div>
            """
        if res_side_len:
            profile_side_html += f"""
                <div class="img-card" style="margin-top: 2rem;">
                    <h3>Perfil Dimensional (Comprimento/Largura)</h3>
                    <div class="img-container">
                        <img src="{plot_side_len_path.name}" alt="Gráfico de Perfil Lateral (Outra Dimensão)">
                    </div>
                </div>
            """

    html_content = HTML_TEMPLATE.format(
        session=session_id,
        state_name=state_name,
        dim_top_len=dim_top_len_str,
        dim_top_width=dim_top_width_str,
        area_top=area_top_str,
        thickness=thickness_str,
        dim_side_w=dim_side_w_str,
        scale_h=scale_h,
        scale_v=scale_v,
        anisotropy=anisotropy,
        annotated_top_cards_html=annotated_top_cards_html,
        annotated_side_cards_html=annotated_side_cards_html,
        profile_top_html=profile_top_html,
        profile_side_html=profile_side_html,
        deviation_top_cards_html=deviation_top_cards_html,
        deviation_side_cards_html=deviation_side_cards_html,
        cad_card_html=cad_card_html,
        cad_th_html=cad_th_html,
        cad_len_td_html=cad_len_td_html,
        cad_width_td_html=cad_width_td_html,
        cad_area_td_html=cad_area_td_html,
        side_rows_html=side_rows_html,
        angle_rows_html=angle_rows_html,
        cross_section_rows_html=cross_section_rows_html,
    )

    output_session_dir = Path(config.OUTPUT_DIR) / session_id
    output_session_dir.mkdir(parents=True, exist_ok=True)
    output_report_path = output_session_dir / f"report_{session_id}.html"
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Relatório gerado com sucesso em: {output_report_path.resolve()}")
    
    if open_browser:
        try:
            webbrowser.open(output_report_path.resolve().as_uri())
        except Exception as e:
            logger.warning(f"Não foi possível abrir o navegador: {e}")

if __name__ == "__main__":
    import sys
    sess = sys.argv[1] if len(sys.argv) > 1 else "16-06"
    generate_report(sess)

