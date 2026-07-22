#!/usr/bin/env python3
import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

def main():
    print("=== DEM 3D Viewer Asset Generator ===")
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, "dem_generator", "dem_new_12k.png")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run the DEM generator first.")
        sys.exit(1)
        
    output_rgb_path = os.path.join(current_dir, "dem_1024_rgb16.png")
    output_texture_path = os.path.join(current_dir, "dem_1024_texture.png")
    output_html_path = os.path.join(current_dir, "dem_viewer_3d.html")
    
    # Target size for the web assets (1024x1024 is highly detailed but lightweight enough)
    target_size = 1024
    
    # 1. Load and process heightmap
    print(f"Loading heightmap {input_path}...")
    img = Image.open(input_path)
    print(f"Original size: {img.size}, format: {img.format}, mode: {img.mode}")
    
    # Downsample to target size using bilinear interpolation
    print(f"Downsampling to {target_size}x{target_size}...")
    img_resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    data_resized = np.array(img_resized, dtype=np.float32)
    
    # Min/Max in raw values and meters
    h_min_raw = data_resized.min()
    h_max_raw = data_resized.max()
    h_min_m = h_min_raw / 100.0
    h_max_m = h_max_raw / 100.0
    print(f"Elevation range: {h_min_m:.2f}m to {h_max_m:.2f}m (raw: {h_min_raw:.1f} to {h_max_raw:.1f})")
    
    # 2. Save 16-bit RGB encoded heightmap
    # Red channel = height % 256
    # Green channel = height // 256
    # Blue channel = 0
    print("Encoding 16-bit heightmap to RGB PNG...")
    data_clipped = np.clip(data_resized, 0, 65535).astype(np.uint16)
    r = (data_clipped % 256).astype(np.uint8)
    g = ((data_clipped // 256) % 256).astype(np.uint8)
    b = np.zeros_like(r)
    
    rgb_data = np.dstack((r, g, b))
    img_rgb = Image.fromarray(rgb_data, mode="RGB")
    img_rgb.save(output_rgb_path)
    print(f"Saved RGB heightmap to: {output_rgb_path}")
    
    # 2.5. Parse OSM way coordinates to build polygon mask for texture coloring
    osm_path = os.path.join(project_root, "osm_generator", "outputs", "zoning_map.osm")
    if not os.path.exists(osm_path):
        osm_path = os.path.join(current_dir, "map.osm")
        
    ways_data = []
    if os.path.exists(osm_path):
        print(f"Found OSM file at: {osm_path}. Parsing features...")
        try:
            tree = ET.parse(osm_path)
            root = tree.getroot()
            
            nodes = {}
            for node in root.findall("node"):
                nid = node.get("id")
                lat = float(node.get("lat"))
                lon = float(node.get("lon"))
                nodes[nid] = (lat, lon)
                
            for way in root.findall("way"):
                wid = way.get("id")
                tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
                refs = [nd.get("ref") for nd in way.findall("nd")]
                coords = [nodes[ref] for ref in refs if ref in nodes]
                
                if coords:
                    ways_data.append({
                        "id": wid,
                        "tags": tags,
                        "coords": coords
                    })
            print(f"Parsed {len(ways_data)} ways from OSM.")
        except Exception as e:
            print(f"Warning: Failed to parse OSM: {e}")
    else:
        print("No OSM file found. Skipping feature overlays.")
        
    osm_data_json = json.dumps(ways_data)
    
    # Generate the base color image (1024x1024)
    # Default background is a clean light gray
    bg_color = (204, 204, 204) # #CCCCCC
    base_color_img = Image.new("RGB", (target_size, target_size), bg_color)
    draw = ImageDraw.Draw(base_color_img)
    
    if ways_data:
        min_lon = 37.715173389134144
        max_lon = 37.82446513086585
        min_lat = 47.57967188702157
        max_lat = 47.653344312978426
        
        # Color definitions for tags
        # (tag_key, tag_value, hex_color)
        color_rules = [
            ("natural", "wood", "#22C55E"),      # Forest green
            ("landuse", "forest", "#22C55E"),
            ("landuse", "farmyard", "#EC4899"),   # Pink for farmyard
            ("landuse", "farmland", "#86EFAC"),   # Light green for farmland
            ("natural", "water", "#2563EB"),     # Water blue
            ("water", None, "#2563EB"),
            ("highway", None, "#4B5563"),        # Road gray
        ]
        
        def hex_to_rgb(hex_str):
            h = hex_str.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            
        for way in ways_data:
            tags = way.get("tags", {})
            coords = way.get("coords", [])
            if len(coords) < 2:
                continue
                
            # Find matching color rule
            match_color = None
            for key, val, hex_color in color_rules:
                if key in tags:
                    if val is None or tags[key] == val:
                        match_color = hex_to_rgb(hex_color)
                        break
                        
            if match_color is None:
                continue
                
            # Convert coords to pixel coordinates on the 1024x1024 texture.
            # The DEM is 12288px total but only the central 8192px correspond to the OSM area.
            # Ratio = 8192/12288 = 2/3. In the 1024px texture that central band is:
            #   osm_px = 1024 * (8192/12288) = 682.67px, offset = (1024 - 682.67) / 2 = 170.67px
            osm_band = target_size * (8192 / 12288)  # ~682.67px
            osm_offset = (target_size - osm_band) / 2  # ~170.67px
            poly_points = []
            for lat, lon in coords:
                u = (lon - min_lon) / (max_lon - min_lon)
                v = (max_lat - lat) / (max_lat - min_lat)
                px = osm_offset + u * osm_band
                py = osm_offset + v * osm_band
                poly_points.append((px, py))
                
            # Check if closed way (polygon) or open way (line)
            is_closed = len(coords) > 2 and coords[0] == coords[-1]
            
            if is_closed:
                draw.polygon(poly_points, fill=match_color)
            else:
                draw.line(poly_points, fill=match_color, width=4)
                
    # 3. Generate a beautiful custom terrain texture with shaded relief
    print("Generating shaded relief terrain texture (OSM colors shaded, rest grayscale)...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
        
        # Create shaded relief
        ls = LightSource(azdeg=315, altdeg=45)
        # Convert base color image to float [0, 1] array
        rgb_input = np.array(base_color_img, dtype=np.float32) / 255.0
        # Apply shading directly on the custom colored image using shade_rgb
        shaded = ls.shade_rgb(rgb_input, elevation=data_resized / 100.0, blend_mode='overlay', vert_exag=1.5, dx=8.0, dy=8.0)
        
        # Convert float [0, 1] to uint8 [0, 255] and save
        texture_data = (shaded * 255).astype(np.uint8)
        img_texture = Image.fromarray(texture_data)
        img_texture.save(output_texture_path)
        print(f"Saved shaded terrain texture to: {output_texture_path}")
    except Exception as e:
        print(f"Warning: Could not generate shaded relief using matplotlib: {e}")
        # Fallback: just save the base_color_img directly!
        base_color_img.save(output_texture_path)
        print(f"Saved fallback flat colored texture to: {output_texture_path}")
        
    # 4. Generate HTML interactive 3D Viewer
    print("Writing HTML interactive 3D viewer...")
    
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizador 3D - DEM Matopiba</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <!-- Three.js and OrbitControls via CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <style>
        :root {{
            --bg-color: #0d0e12;
            --panel-bg: rgba(18, 20, 26, 0.75);
            --panel-border: rgba(255, 255, 255, 0.1);
            --accent-color: #4f46e5;
            --accent-hover: #6366f1;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }}

        #canvas-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }}

        /* HUD overlay */
        .hud-panel {{
            position: absolute;
            z-index: 10;
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .hud-panel:hover {{
            border-color: rgba(255, 255, 255, 0.18);
        }}

        /* Header Panel */
        #header-panel {{
            top: 20px;
            left: 20px;
            max-width: 400px;
        }}

        h1 {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
            background: linear-gradient(135deg, #fff 30%, var(--text-muted) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 16px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.02);
        }}

        .stat-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        .stat-value {{
            font-size: 16px;
            font-weight: 600;
        }}

        /* Control Panel */
        #control-panel {{
            top: 20px;
            right: 20px;
            width: 320px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}

        .control-group {{
            margin-bottom: 20px;
        }}

        .control-group:last-child {{
            margin-bottom: 0;
        }}

        .group-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .control-row {{
            margin-bottom: 12px;
        }}

        label {{
            display: block;
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}

        /* Inputs and Sliders */
        .slider-container {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        input[type="range"] {{
            flex: 1;
            -webkit-appearance: none;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-color);
            cursor: pointer;
            transition: background 0.2s;
        }}

        input[type="range"]::-webkit-slider-thumb:hover {{
            background: var(--accent-hover);
        }}

        .slider-value {{
            font-size: 12px;
            width: 35px;
            text-align: right;
            font-weight: 600;
        }}

        /* Buttons and Selectors */
        .btn-toggle-group {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
            background: rgba(0, 0, 0, 0.2);
            padding: 4px;
            border-radius: 8px;
        }}

        .btn-toggle {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 4px;
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
            font-size: 11px;
            font-weight: 600;
            transition: all 0.2s;
        }}

        .btn-toggle.active {{
            background: var(--accent-color);
            color: #fff;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4);
        }}

        .btn-toggle:hover:not(.active) {{
            color: var(--text-color);
            background: rgba(255, 255, 255, 0.05);
        }}

        .btn-primary {{
            background: var(--accent-color);
            border: none;
            color: #fff;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 13px;
            font-weight: 600;
            width: 100%;
            transition: all 0.2s;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }}

        .btn-primary:hover {{
            background: var(--accent-hover);
            box-shadow: 0 4px 16px rgba(79, 70, 229, 0.4);
        }}

        /* Switch checkbox */
        .switch-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }}

        .switch {{
            position: relative;
            display: inline-block;
            width: 40px;
            height: 20px;
        }}

        .switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}

        .switch-slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.1);
            transition: .3s;
            border-radius: 20px;
        }}

        .switch-slider:before {{
            position: absolute;
            content: "";
            height: 14px;
            width: 14px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }}

        input:checked + .switch-slider {{
            background-color: var(--accent-color);
        }}

        input:checked + .switch-slider:before {{
            transform: translateX(20px);
        }}

        /* Probe Panel */
        #probe-panel {{
            bottom: 20px;
            left: 20px;
            width: 320px;
            display: none;
        }}

        .probe-value {{
            font-family: monospace;
            font-size: 13px;
        }}

        /* Help overlay */
        #help-btn {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            color: var(--text-color);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 10;
            font-weight: bold;
            font-size: 18px;
            backdrop-filter: blur(12px);
        }}

        #help-modal {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) scale(0.9);
            z-index: 100;
            width: 90%;
            max-width: 450px;
            background: rgba(18, 20, 26, 0.95);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 32px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(20px);
            opacity: 0;
            pointer-events: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        #help-modal.active {{
            opacity: 1;
            pointer-events: auto;
            transform: translate(-50%, -50%) scale(1);
        }}

        .modal-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 16px;
        }}

        .modal-body p {{
            font-size: 14px;
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 16px;
        }}

        .modal-controls {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 12px;
            margin-top: 16px;
            font-size: 13px;
        }}

        .control-key {{
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-weight: bold;
            text-align: center;
        }}

        .close-modal {{
            margin-top: 24px;
        }}

        #loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--bg-color);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: opacity 0.5s ease;
        }}

        .spinner {{
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: var(--accent-color);
            animation: spin 1s ease-in-out infinite;
            margin-bottom: 20px;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        #loading-text {{
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 1px;
            color: var(--text-color);
        }}

        #loading-progress {{
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 8px;
        }}
    </style>
</head>
<body>

    <div id="loading-overlay">
        <div class="spinner"></div>
        <div id="loading-text">CARGANDO ELEVACIÓN 3D...</div>
        <div id="loading-progress">Inicializando WebGL</div>
    </div>

    <div id="canvas-container"></div>

    <!-- Header / Info Panel -->
    <div id="header-panel" class="hud-panel">
        <div class="subtitle">Farming Simulator 25</div>
        <h1>DEM Matopiba 3D</h1>
        <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">Procedural Heightmap & Layout Viewer</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Tamaño Real</div>
                <div class="stat-value">8.19 × 8.19 km</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Rango Alturas</div>
                <div class="stat-value">{h_min_m:.1f}m - {h_max_m:.1f}m</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Área Jugable</div>
                <div class="stat-value">8.19 × 8.19 km</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Vértices 3D</div>
                <div class="stat-value" id="mesh-vertices">262,144</div>
            </div>
        </div>
    </div>

    <!-- Controls Panel -->
    <div id="control-panel" class="hud-panel">
        <div class="control-group">
            <div class="group-title">Visualización</div>
            <div class="control-row">
                <label>Modo de Superficie</label>
                <div class="btn-toggle-group">
                    <button class="btn-toggle active" onclick="setRenderMode('texture')">Textura</button>
                    <button class="btn-toggle" onclick="setRenderMode('elevation')">Elevación</button>
                    <button class="btn-toggle" onclick="setRenderMode('wireframe')">Malla</button>
                </div>
            </div>
            <div class="control-row">
                <label>Resolución de Malla (Vértices)</label>
                <div class="btn-toggle-group" style="grid-template-columns: repeat(3, 1fr);">
                    <button class="btn-toggle" onclick="changeMeshResolution(256)">256²</button>
                    <button class="btn-toggle active" onclick="changeMeshResolution(512)">512²</button>
                    <button class="btn-toggle" onclick="changeMeshResolution(1024)">1024²</button>
                </div>
            </div>
        </div>

        <div class="control-group">
            <div class="group-title">Parámetros del Relieve</div>
            <div class="control-row">
                <label>Exageración Vertical</label>
                <div class="slider-container">
                    <input type="range" id="exaggeration-slider" min="0.1" max="5.0" step="0.1" value="1.5" oninput="updateExaggeration(this.value)">
                    <div class="slider-value" id="exaggeration-val">1.5x</div>
                </div>
            </div>
        </div>

        <div class="control-group">
            <div class="group-title">Límites & Guías</div>
            
            <div class="switch-container">
                <span style="font-size: 13px;">Límite Jugable (8km)</span>
                <label class="switch">
                    <input type="checkbox" id="toggle-playable" onchange="togglePlayableBox(this.checked)">
                    <span class="switch-slider"></span>
                </label>
            </div>
        </div>

        <div class="control-group">
            <div class="group-title">Iluminación (Sol)</div>
            <div class="control-row">
                <label>Dirección del Sol (Ángulo)</label>
                <div class="slider-container">
                    <input type="range" id="sun-angle" min="0" max="360" value="135" oninput="updateSunAngle(this.value)">
                    <div class="slider-value" id="sun-angle-val">135°</div>
                </div>
            </div>
            <div class="control-row">
                <label>Altitud del Sol</label>
                <div class="slider-container">
                    <input type="range" id="sun-alt" min="10" max="90" value="45" oninput="updateSunAltitude(this.value)">
                    <div class="slider-value" id="sun-alt-val">45°</div>
                </div>
            </div>
        </div>

        <button class="btn-primary" onclick="resetCamera()">Restablecer Cámara</button>
    </div>

    <!-- Hover Probe Panel -->
    <div id="probe-panel" class="hud-panel">
        <div class="group-title" style="margin-bottom: 8px;">Información de Punto</div>
        <div style="display: flex; flex-direction: column; gap: 6px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 12px; color: var(--text-muted);">Coordenadas X, Z:</span>
                <span class="probe-value" id="probe-coords">0m, 0m</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 12px; color: var(--text-muted);">Elevación Real:</span>
                <span class="probe-value" id="probe-height" style="color: #60a5fa; font-weight: bold;">0.0m</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 12px; color: var(--text-muted);">Zona:</span>
                <span class="probe-value" id="probe-zone" style="font-weight: 600;">Zona de Juego</span>
            </div>
        </div>
    </div>

    <div id="help-btn" onclick="toggleHelp(true)">?</div>

    <div id="help-modal">
        <div class="modal-title">Navegación 3D</div>
        <div class="modal-body">
            <p>Usa tu ratón (o gestos táctiles) para rotar, trasladar y hacer zoom en el modelo del terreno:</p>
            <div class="modal-controls">
                <span class="control-key">Clic Izq + Arrastrar</span>
                <span>Rotar la cámara sobre el terreno</span>
                
                <span class="control-key">Rueda del Ratón</span>
                <span>Acercar y alejar (Zoom)</span>
                
                <span class="control-key">Clic Der + Arrastrar</span>
                <span>Trasladar / Panorámica (Mover la vista)</span>
            </div>
            <p style="margin-top: 16px;">Coloca el puntero del ratón sobre el mapa para analizar las coordenadas y elevación del terreno en tiempo real.</p>
        </div>
        <button class="btn-primary close-modal" onclick="toggleHelp(false)">¡Entendido!</button>
    </div>

    <script>
        // Elevation ranges from Python
        const MIN_HEIGHT = {h_min_m};
        const MAX_HEIGHT = {h_max_m};
        const MAP_SIZE = 8192; // 8192m width and length
        const PLAYABLE_SIZE = 8192;
        const PLAYABLE_OFFSET = (MAP_SIZE - PLAYABLE_SIZE) / 2;
        
        // OSM Bounds and Data (from zoning_map.osm)
        const MIN_LON = 37.715173389134144;
        const MAX_LON = 37.82446513086585;
        const MIN_LAT = 47.57967188702157;
        const MAX_LAT = 47.653344312978426;
        const OSM_DATA = {osm_data_json};
        
        let container, scene, camera, renderer, controls;
        let terrainGeom, terrainMesh, terrainMaterial;
        let heightData = null; // Float32Array storing raw elevations in meters
        let heightWidth = 0, heightHeight = 0;
        
        let currentRes = 512;
        let renderMode = 'texture'; // 'texture', 'elevation', 'wireframe'
        let verticalExaggeration = 1.5;
        
        // Scene objects
        let sunLight, ambientLight;
        let playableBox;
        let raycaster, mouse;
        
        // Textures
        let colorTexture, heightmapImage;

        // Initialize App
        window.onload = function() {{
            init();
        }};

        function init() {{
            container = document.getElementById('canvas-container');
            
            // Set up Scene, Camera, Renderer
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0d0e12);
            scene.fog = new THREE.FogExp2(0x0d0e12, 0.0001);

            camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 10, 30000);
            
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);

            // Orbit Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground
            controls.minDistance = 50;
            controls.maxDistance = 15000;
            
            // Default camera view
            resetCamera();

            // Lighting
            ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);

            sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 2048;
            sunLight.shadow.mapSize.height = 2048;
            sunLight.shadow.camera.near = 100;
            sunLight.shadow.camera.far = 20000;
            const d = 5000;
            sunLight.shadow.camera.left = -d;
            sunLight.shadow.camera.right = d;
            sunLight.shadow.camera.top = d;
            sunLight.shadow.camera.bottom = -d;
            scene.add(sunLight);
            
            updateSunPosition(135, 45);

            // Setup Raycasting
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();

            // Load assets
            loadAssets();

            // Resize handler
            window.addEventListener('resize', onWindowResize, false);
            
            // Mouse move for terrain inspection
            window.addEventListener('mousemove', onMouseMove, false);
            
            // Start Loop
            animate();
        }}

        function resetCamera() {{
            camera.position.set(0, 4000, 6000);
            controls.target.set(0, 100, 0);
            controls.update();
        }}

        function updateSunPosition(angleDeg, altitudeDeg) {{
            const angleRad = (angleDeg * Math.PI) / 180;
            const altRad = (altitudeDeg * Math.PI) / 180;
            
            const r = 8000;
            const y = r * Math.sin(altRad);
            const x = r * Math.cos(altRad) * Math.cos(angleRad);
            const z = r * Math.cos(altRad) * Math.sin(angleRad);
            
            sunLight.position.set(x, y, z);
        }}

        function loadAssets() {{
            const loadingProgress = document.getElementById('loading-progress');
            
            // Load visual texture & RGB Heightmap
            const textureLoader = new THREE.TextureLoader();
            
            loadingProgress.innerText = "Cargando Textura del Mapa...";
            
            // Cache-busting query parameter to force reloading the images from disk
            const cb = '?t=' + Date.now();
            
            textureLoader.load('dem_1024_texture.png' + cb, function(tex) {{
                colorTexture = tex;
                colorTexture.wrapS = THREE.ClampToEdgeWrapping;
                colorTexture.wrapT = THREE.ClampToEdgeWrapping;
                
                loadingProgress.innerText = "Cargando Datos de Elevación (16-bit)...";
                
                const img = new Image();
                img.src = 'dem_1024_rgb16.png' + cb;
                img.onload = function() {{
                    heightmapImage = img;
                    
                    // Parse RGB image to heights
                    parseHeightmap(img);
                    
                    // Build terrain mesh
                    buildTerrainMesh(currentRes);
                    
                    // Build auxiliary lines/guides
                    buildGuides();
                    
                    // Remove loading overlay
                    const loader = document.getElementById('loading-overlay');
                    loader.style.opacity = 0;
                    setTimeout(() => loader.style.display = 'none', 500);
                }};
            }}, undefined, function(err) {{
                console.error("Error loading terrain texture", err);
                loadingProgress.innerText = "Error al cargar texturas. Iniciando con colores planos...";
                // Fallback heightmap parse
                const img = new Image();
                img.src = 'dem_1024_rgb16.png' + cb;
                img.onload = function() {{
                    heightmapImage = img;
                    parseHeightmap(img);
                    buildTerrainMesh(currentRes);
                    buildGuides();
                    document.getElementById('loading-overlay').style.display = 'none';
                }};
            }});
        }}

        function parseHeightmap(img) {{
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            const imgData = ctx.getImageData(0, 0, img.width, img.height);
            const data = imgData.data;
            
            heightWidth = img.width;
            heightHeight = img.height;
            heightData = new Float32Array(heightWidth * heightHeight);
            
            for (let i = 0; i < heightData.length; i++) {{
                const r = data[i * 4];
                const g = data[i * 4 + 1];
                // Decode 16-bit value (in cm) and convert to meters
                const rawHeight = r + g * 256;
                heightData[i] = rawHeight / 100.0;
            }}
        }}

        function getInterpolatedHeight(u, v) {{
            if (!heightData) return 0;
            
            // Map u, v (0 to 1) to image pixel coords
            const px = u * (heightWidth - 1);
            const py = v * (heightHeight - 1);
            
            const x0 = Math.floor(px);
            const y0 = Math.floor(py);
            const x1 = Math.min(x0 + 1, heightWidth - 1);
            const y1 = Math.min(y0 + 1, heightHeight - 1);
            
            const tx = px - x0;
            const ty = py - y0;
            
            const h00 = heightData[y0 * heightWidth + x0];
            const h10 = heightData[y0 * heightWidth + x1];
            const h01 = heightData[y1 * heightWidth + x0];
            const h11 = heightData[y1 * heightWidth + x1];
            
            // Bilinear interpolation
            const h0 = h00 * (1 - tx) + h10 * tx;
            const h1 = h01 * (1 - tx) + h11 * tx;
            return h0 * (1 - ty) + h1 * ty;
        }}

        function buildTerrainMesh(res) {{
            if (terrainMesh) {{
                scene.remove(terrainMesh);
                terrainGeom.dispose();
            }}
            
            document.getElementById('mesh-vertices').innerText = (res * res).toLocaleString();
            
            // Geometry dimensions matching real map scale (8192m x 8192m)
            terrainGeom = new THREE.PlaneGeometry(MAP_SIZE, MAP_SIZE, res - 1, res - 1);
            
            // Displace plane vertices along Y (originally Z before rotation)
            const posAttr = terrainGeom.attributes.position;
            const count = posAttr.count;
            
            for (let i = 0; i < count; i++) {{
                // PlaneCoordinates are from -MAP_SIZE/2 to MAP_SIZE/2
                const x = posAttr.getX(i);
                const z = posAttr.getY(i);
                
                // Map x,z (-4096 to 4096) to u,v (0 to 1)
                const u = (x + MAP_SIZE / 2) / MAP_SIZE;
                const v = 1 - (z + MAP_SIZE / 2) / MAP_SIZE;
                
                const height = getInterpolatedHeight(u, v);
                posAttr.setZ(i, height * verticalExaggeration);
            }}
            
            // Rotate the plane to sit flat horizontally
            terrainGeom.rotateX(-Math.PI / 2);
            terrainGeom.computeVertexNormals();
            
            // Materials
            if (renderMode === 'texture') {{
                terrainMaterial = new THREE.MeshStandardMaterial({{
                    map: colorTexture,
                    roughness: 0.85,
                    metalness: 0.1,
                    flatShading: false
                }});
            }} else if (renderMode === 'elevation') {{
                buildVertexColors(res);
                terrainMaterial = new THREE.MeshStandardMaterial({{
                    vertexColors: true,
                    roughness: 0.8,
                    metalness: 0.1
                }});
            }} else {{
                // Wireframe
                terrainMaterial = new THREE.MeshBasicMaterial({{
                    color: 0x6366f1,
                    wireframe: true
                }});
            }}
            
            terrainMesh = new THREE.Mesh(terrainGeom, terrainMaterial);
            terrainMesh.receiveShadow = true;
            terrainMesh.castShadow = true;
            scene.add(terrainMesh);
        }}

        function buildVertexColors(res) {{
            const count = terrainGeom.attributes.position.count;
            const colors = [];
            
            // Color palettes representing elevations
            // Gradient from Green (lowlands) -> Yellow -> Brown -> White (peaks)
            const colorRamp = [
                {{ h: MIN_HEIGHT, c: new THREE.Color(0x1e4620) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.05, c: new THREE.Color(0x2d6a2e) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.15, c: new THREE.Color(0x658c43) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.40, c: new THREE.Color(0xb8a174) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.60, c: new THREE.Color(0x8e7355) }},
                {{ h: MIN_HEIGHT + (MAX_HEIGHT - MIN_HEIGHT) * 0.80, c: new THREE.Color(0x5c5247) }},
                {{ h: MAX_HEIGHT, c: new THREE.Color(0xffffff) }}
            ];
            
            const posAttr = terrainGeom.attributes.position;
            for (let i = 0; i < count; i++) {{
                const yVal = posAttr.getY(i) / verticalExaggeration;
                
                let col = new THREE.Color(0xffffff);
                if (yVal <= colorRamp[0].h) {{
                    col.copy(colorRamp[0].c);
                }} else if (yVal >= colorRamp[colorRamp.length - 1].h) {{
                    col.copy(colorRamp[colorRamp.length - 1].c);
                }} else {{
                    for (let j = 0; j < colorRamp.length - 1; j++) {{
                        const lower = colorRamp[j];
                        const upper = colorRamp[j+1];
                        if (yVal >= lower.h && yVal <= upper.h) {{
                            const t = (yVal - lower.h) / (upper.h - lower.h);
                            col.copy(lower.c).lerp(upper.c, t);
                            break;
                        }}
                    }}
                }}
                colors.push(col.r, col.g, col.b);
            }}
            
            terrainGeom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        }}

        function buildGuides() {{
            // 1. Playable Area outline box (4km x 4km centered, height matches elevation bounds)
            const playMin = -PLAYABLE_SIZE / 2;
            const playMax = PLAYABLE_SIZE / 2;
            
            const boxGeom = new THREE.BufferGeometry();
            const yMin = MIN_HEIGHT * verticalExaggeration;
            const yMax = MAX_HEIGHT * verticalExaggeration;
            
            const vertices = [
                // Floor
                playMin, yMin, playMin,  playMax, yMin, playMin,
                playMax, yMin, playMin,  playMax, yMin, playMax,
                playMax, yMin, playMax,  playMin, yMin, playMax,
                playMin, yMin, playMax,  playMin, yMin, playMin,
                // Ceiling
                playMin, yMax, playMin,  playMax, yMax, playMin,
                playMax, yMax, playMin,  playMax, yMax, playMax,
                playMax, yMax, playMax,  playMin, yMax, playMax,
                playMin, yMax, playMax,  playMin, yMax, playMin,
                // Pillars
                playMin, yMin, playMin,  playMin, yMax, playMin,
                playMax, yMin, playMin,  playMax, yMax, playMin,
                playMax, yMin, playMax,  playMax, yMax, playMax,
                playMin, yMin, playMax,  playMin, yMax, playMax,
            ];
            boxGeom.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            
            const boxMat = new THREE.LineBasicMaterial({{ 
                color: 0x4f46e5, 
                linewidth: 2, 
                transparent: true,
                opacity: 0.8
            }});
            playableBox = new THREE.LineSegments(boxGeom, boxMat);
            playableBox.visible = false; // Start hidden by default
            scene.add(playableBox);
        }}

        function updateExaggeration(val) {{
            verticalExaggeration = parseFloat(val);
            document.getElementById('exaggeration-val').innerText = val + 'x';
            
            if (heightData) {{
                buildTerrainMesh(currentRes);
                
                // Get the current visible state of playableBox before recreating it
                const wasPlayableVisible = playableBox ? playableBox.visible : false;
                
                if (playableBox) {{
                    scene.remove(playableBox);
                }}
                
                buildGuides();
                
                // Restore the visibility state
                if (playableBox) {{
                    playableBox.visible = wasPlayableVisible;
                }}
            }}
        }}

        function setRenderMode(mode) {{
            renderMode = mode;
            
            const buttons = document.querySelectorAll('#control-panel .control-row:nth-child(1) .btn-toggle');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            if (mode === 'texture') buttons[0].classList.add('active');
            if (mode === 'elevation') buttons[1].classList.add('active');
            if (mode === 'wireframe') buttons[2].classList.add('active');
            
            if (heightData) {{
                buildTerrainMesh(currentRes);
            }}
        }}

        function changeMeshResolution(res) {{
            currentRes = res;
            
            const buttons = document.querySelectorAll('#control-panel .control-row:nth-child(2) .btn-toggle');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            if (res === 256) buttons[0].classList.add('active');
            if (res === 512) buttons[1].classList.add('active');
            if (res === 1024) buttons[2].classList.add('active');
            
            if (heightData) {{
                const overlay = document.getElementById('loading-overlay');
                const pText = document.getElementById('loading-progress');
                document.getElementById('loading-text').innerText = "RECONSTRUYENDO MALLA 3D...";
                pText.innerText = "Calculando vértices a " + res + "²...";
                overlay.style.display = 'flex';
                overlay.style.opacity = 1;
                
                setTimeout(() => {{
                    buildTerrainMesh(res);
                    overlay.style.opacity = 0;
                    setTimeout(() => overlay.style.display = 'none', 300);
                }}, 50);
            }}
        }}

        function togglePlayableBox(visible) {{
            if (playableBox) playableBox.visible = visible;
        }}



        function updateSunAngle(angle) {{
            document.getElementById('sun-angle-val').innerText = angle + '°';
            const alt = document.getElementById('sun-alt').value;
            updateSunPosition(parseInt(angle), parseInt(alt));
        }}

        function updateSunAltitude(alt) {{
            document.getElementById('sun-alt-val').innerText = alt + '°';
            const angle = document.getElementById('sun-angle').value;
            updateSunPosition(parseInt(angle), parseInt(alt));
        }}

        function toggleHelp(show) {{
            const modal = document.getElementById('help-modal');
            if (show) {{
                modal.classList.add('active');
            }} else {{
                modal.classList.remove('active');
            }}
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        }}

        function checkTerrainIntersection() {{
            if (!terrainMesh || !heightData) return;
            
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(terrainMesh);
            
            const probePanel = document.getElementById('probe-panel');
            
            if (intersects.length > 0) {{
                const point = intersects[0].point;
                const x = point.x;
                const z = point.z;
                
                if (Math.abs(x) <= MAP_SIZE/2 && Math.abs(z) <= MAP_SIZE/2) {{
                    probePanel.style.display = 'block';
                    const realY = point.y / verticalExaggeration;
                    
                    document.getElementById('probe-coords').innerText = `X: ${{Math.round(x)}}m | Z: ${{Math.round(z)}}m`;
                    document.getElementById('probe-height').innerText = `${{realY.toFixed(2)}}m`;
                    
                    const inPlayable = Math.abs(x) <= PLAYABLE_SIZE/2 && Math.abs(z) <= PLAYABLE_SIZE/2;
                    const zoneLabel = document.getElementById('probe-zone');
                    
                    if (inPlayable) {{
                        const dxLake = x - 1704;
                        const dzLake = z - 1704;
                        const distLake = Math.max(Math.abs(dxLake), Math.abs(dzLake));
                        
                        if (distLake <= 90) {{
                            zoneLabel.innerText = "Lago / Reserva (225m)";
                            zoneLabel.style.color = "#3b82f6";
                        }} else {{
                            zoneLabel.innerText = "Área Jugable";
                            zoneLabel.style.color = "#10b981";
                        }}
                    }} else {{
                        zoneLabel.innerText = "Fondo No Jugable";
                        zoneLabel.style.color = "#f43f5e";
                    }}
                }} else {{
                    probePanel.style.display = 'none';
                }}
            }} else {{
                probePanel.style.display = 'none';
            }}
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            checkTerrainIntersection();
            renderer.render(scene, camera);
        }}
    </script>
</body>
</html>
"""
    
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML viewer to: {output_html_path}")
    print("\n=== Success! ===")
    print("To view the 3D visualization, start a local HTTP server in this directory:")
    print("  python3 -m http.server 8000")
    print("Then open in your browser:")
    print("  http://localhost:8000/dem_viewer_3d.html")

if __name__ == "__main__":
    main()
