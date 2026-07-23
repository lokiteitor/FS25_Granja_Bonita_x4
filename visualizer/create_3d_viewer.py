#!/usr/bin/env python3
import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

def val_noise(shape, grid_size, weight, seed=20260608):
    """Generates smooth value noise by upscaling a small random grid using bicubic interpolation."""
    np.random.seed(seed)
    small = np.random.uniform(-1.0, 1.0, size=(grid_size, grid_size)).astype(np.float32)
    temp_img = Image.fromarray(small)
    temp_img = temp_img.resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.array(temp_img) * weight

def create_procedural_base_texture(target_size):
    """Generates a gorgeous procedural base texture representing the geographical features of the DEM."""
    # Coordinate grids
    y_indices_px, x_indices_px = np.indices((target_size, target_size), dtype=np.float32)
    x_m = x_indices_px * (8192.0 / target_size)
    y_m = y_indices_px * (8192.0 / target_size)
    
    # Configuration
    playable_size_m = 4096.0
    offset_m = 2048.0
    
    # Background weight
    dx_bg = np.maximum(0.0, np.maximum(offset_m - x_m, x_m - (8192.0 - offset_m)))
    dy_bg = np.maximum(0.0, np.maximum(offset_m - y_m, y_m - (8192.0 - offset_m)))
    dist_border_bg = np.sqrt(dx_bg*dx_bg + dy_bg*dy_bg)
    t_bg = np.clip(dist_border_bg / 1024.0, 0.0, 1.0)
    w_bg = 0.5 * (1.0 - np.cos(np.pi * t_bg))
    
    # North Flat Area boundary modulation
    seed = 20260608
    np.random.seed(seed)
    boundary_noise_1d = val_noise((target_size, target_size), 8, 150.0, seed=seed+15)[0, :]
    boundary_y = offset_m + 1000.0 + boundary_noise_1d[np.newaxis, :]
    
    # Weight of the undulating rest area
    t_y = np.clip((y_m - boundary_y) / 400.0, 0.0, 1.0)
    w_rest = 0.5 * (1.0 - np.cos(np.pi * t_y))
    
    # South Mountain barrier
    d_south = (offset_m + playable_size_m) - y_m
    t_m = np.clip(d_south / 400.0, 0.0, 1.0)
    w_mountain = 0.5 * (1.0 + np.cos(np.pi * t_m))
    
    # Southwest peak
    dx_sw = x_m - offset_m
    dy_sw = y_m - (offset_m + playable_size_m)
    d_sw = np.sqrt(dx_sw**2 + dy_sw**2)
    t_sw = np.clip(d_sw / 300.0, 0.0, 1.0)
    w_sw = 0.5 * (1.0 + np.cos(np.pi * t_sw))
    
    w_rocky = np.maximum(w_mountain, w_sw)
    
    # Southeast flat square
    sq_size = 707.1
    sq_x1 = offset_m + playable_size_m - 15.0
    sq_x0 = sq_x1 - sq_size
    sq_y1 = offset_m + playable_size_m - 450.0
    sq_y0 = sq_y1 - sq_size
    scx = (sq_x0 + sq_x1) / 2.0
    scy = (sq_y0 + sq_y1) / 2.0
    srx = (sq_x1 - sq_x0) / 2.0
    sry = (sq_y1 - sq_y0) / 2.0
    dx_sq = np.abs(x_m - scx) - srx
    dy_sq = np.abs(y_m - scy) - sry
    dist_outside_sq = np.sqrt(np.maximum(0.0, dx_sq)**2 + np.maximum(0.0, dy_sq)**2)
    dist_inside_sq = np.minimum(0.0, np.maximum(dx_sq, dy_sq))
    sdf_sq = dist_outside_sq + dist_inside_sq
    t_sq = np.clip(sdf_sq / 100.0, 0.0, 1.0)
    w_sq = 0.5 * (1.0 - np.cos(np.pi * t_sq))
    
    # Reservoir
    rx0 = offset_m + 15.0
    rx1 = rx0 + 500.0
    ry0 = offset_m + 15.0
    ry1 = ry0 + 500.0
    rcx = (rx0 + rx1) / 2.0
    rcy = (ry0 + ry1) / 2.0
    rrx = (rx1 - rx0) / 2.0
    rry = (ry1 - ry0) / 2.0
    dx_res = np.abs(x_m - rcx) - rrx
    dy_res = np.abs(y_m - rcy) - rry
    dist_outside = np.sqrt(np.maximum(0.0, dx_res)**2 + np.maximum(0.0, dy_res)**2)
    dist_inside = np.minimum(0.0, np.maximum(dx_res, dy_res))
    sdf_res = dist_outside + dist_inside
    t_res = np.clip(-sdf_res / 15.0, 0.0, 1.0)
    w_res = 3 * t_res**2 - 2 * t_res**3
    
    # West channel
    x_c = offset_m + 15.0 + 7.5
    y_start_ch = offset_m + 15.0
    y_end_ch = offset_m + playable_size_m - 400.0
    t_ch_segment = np.clip((y_m - y_start_ch) / (y_end_ch - y_start_ch), 0.0, 1.0)
    proj_y = y_start_ch + t_ch_segment * (y_end_ch - y_start_ch)
    proj_x = x_c
    dist_segment = np.sqrt((x_m - proj_x)**2 + (y_m - proj_y)**2)
    t_ch = np.clip((dist_segment - 3.5) / 4.0, 0.0, 1.0)
    w_ch = 0.5 * (1.0 + np.cos(np.pi * t_ch))
    
    # Southern pond
    px0 = offset_m + 15.0
    px1 = px0 + 50.0
    py0 = offset_m + playable_size_m - 450.0
    py1 = py0 + 50.0
    pcx = (px0 + px1) / 2.0
    pcy = (py0 + py1) / 2.0
    prx = (px1 - px0) / 2.0
    pry = (py1 - py0) / 2.0
    dx_pond = np.abs(x_m - pcx) - prx
    dy_pond = np.abs(y_m - pcy) - pry
    dist_outside_pond = np.sqrt(np.maximum(0.0, dx_pond)**2 + np.maximum(0.0, dy_pond)**2)
    dist_inside_pond = np.minimum(0.0, np.maximum(dx_pond, dy_pond))
    sdf_pond = dist_outside_pond + dist_inside_pond
    t_pond = np.clip(-sdf_pond / 5.0, 0.0, 1.0)
    w_pond = 3 * t_pond**2 - 2 * t_pond**3
    
    w_water = np.maximum(w_res, np.maximum(w_ch, w_pond))
    
    # Colors
    c_bg = np.array([45, 90, 48])           # Dark Forest Green
    c_flat_north = np.array([215, 185, 95])  # Golden Wheat Fields
    c_undulating = np.array([85, 165, 90])   # Lush Green Pasture
    c_rocky = np.array([125, 120, 115])      # Slate Rocky Mountain
    c_sq = np.array([195, 180, 165])         # Sandy Farmyard Dirt
    c_water = np.array([25, 105, 195])       # Deep Water Blue
    
    # Generate field grid grain noise (subtle farming texture)
    np.random.seed(seed + 99)
    noise_grain = np.random.normal(0.0, 6.0, size=(target_size, target_size, 1))
    
    # Blend playable area colors
    c_playable = w_rest[:, :, np.newaxis] * c_undulating + (1.0 - w_rest[:, :, np.newaxis]) * c_flat_north
    
    # Rocky mountains overlay
    c_playable = w_rocky[:, :, np.newaxis] * c_rocky + (1.0 - w_rocky[:, :, np.newaxis]) * c_playable
    
    # Southeast square (farmyard yard)
    c_playable = w_sq[:, :, np.newaxis] * c_playable + (1.0 - w_sq[:, :, np.newaxis]) * c_sq
    
    # Add subtle grain to land
    c_playable = np.clip(c_playable + noise_grain * (1.0 - w_rocky[:, :, np.newaxis] * 0.5), 0, 255)
    
    # Map-wide blend: background mountains vs playable area
    c_full = w_bg[:, :, np.newaxis] * c_bg + (1.0 - w_bg[:, :, np.newaxis]) * c_playable
    
    # Water overlay on top
    c_full = w_water[:, :, np.newaxis] * c_water + (1.0 - w_water[:, :, np.newaxis]) * c_full
    
    base_color_img = Image.fromarray(c_full.astype(np.uint8), mode="RGB")
    return base_color_img

def main():
    print("=== DEM 3D Viewer Asset Generator ===")
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_path = os.path.join(project_root, "dem_generator", "dem_new_8k.png")
    
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
    
    # Min/Max in raw values and meters (1 meter = 128 units)
    h_min_raw = data_resized.min()
    h_max_raw = data_resized.max()
    h_min_m = h_min_raw / 128.0
    h_max_m = h_max_raw / 128.0
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
    
    # Default coordinates (can be overridden by the OSM bounds if available)
    min_lon = 37.715173389134144
    max_lon = 37.82446513086585
    min_lat = 47.57967188702157
    max_lat = 47.653344312978426

    # 2.5. Parse OSM way coordinates to build polygon mask for texture coloring
    osm_path = os.path.join(project_root, "osm_generator", "outputs", "manual.osm")
    if not os.path.exists(osm_path):
        osm_path = os.path.join(project_root, "osm_generator", "outputs", "zoning_map.osm")
    if not os.path.exists(osm_path):
        osm_path = os.path.join(current_dir, "map.osm")
        
    ways_data = []
    if os.path.exists(osm_path):
        print(f"Found OSM file at: {osm_path}. Parsing features...")
        try:
            tree = ET.parse(osm_path)
            root = tree.getroot()
            
            # Try to read bounding box from the OSM file bounds tag
            bounds = root.find("bounds")
            if bounds is not None:
                min_lat = float(bounds.get("minlat"))
                max_lat = float(bounds.get("maxlat"))
                min_lon = float(bounds.get("minlon"))
                max_lon = float(bounds.get("maxlon"))
                print(f"Using bounds from OSM file: lat=({min_lat}, {max_lat}), lon=({min_lon}, {max_lon})")
            
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
    # Default background is the gorgeous procedural base texture representing geography
    base_color_img = create_procedural_base_texture(target_size)
    
    if ways_data:
        # Create a transparent overlay image for transparency
        overlay = Image.new("RGBA", base_color_img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # Color definitions for tags with transparency
        # Format: (tag_key, tag_value, fill_hex, fill_alpha, outline_hex, outline_alpha, outline_width)
        color_rules = [
            ("natural", "wood", "#15803D", 80, "#14532D", 140, 1),
            ("landuse", "forest", "#15803D", 80, "#14532D", 140, 1),
            ("landuse", "farmyard", "#A8A29E", 100, "#57534E", 180, 2),
            ("landuse", "farmland", "#D8A060", 60, "#5C3E21", 150, 2),
            ("natural", "water", "#1D4ED8", 180, "#1E3A8A", 220, 2),
            ("water", None, "#1D4ED8", 180, "#1E3A8A", 220, 2),
            ("highway", "primary", "#374151", 220, "#1F2937", 220, 6),
            ("highway", "secondary", "#4B5563", 220, "#374151", 220, 4),
            ("highway", None, "#52525B", 220, "#3F3F46", 220, 3),
        ]
        
        def hex_to_rgba(hex_str, alpha):
            h = hex_str.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            return rgb + (alpha,)
            
        for way in ways_data:
            tags = way.get("tags", {})
            coords = way.get("coords", [])
            if len(coords) < 2:
                continue
                
            # Find matching color rule
            match_rule = None
            for rule in color_rules:
                key, val, fill_hex, fill_alpha, outline_hex, outline_alpha, outline_width = rule
                if key in tags:
                    if val is None or tags[key] == val:
                        match_rule = rule
                        break
                        
            if match_rule is None:
                continue
                
            key, val, fill_hex, fill_alpha, outline_hex, outline_alpha, outline_width = match_rule
            fill_rgba = hex_to_rgba(fill_hex, fill_alpha)
            outline_rgba = hex_to_rgba(outline_hex, outline_alpha)
            
            # Convert coords to pixel coordinates on the 1024x1024 texture.
            # The DEM is 8192px total but only the central 4096px correspond to the playable area.
            # Ratio = 4096/8192 = 0.5. In the 1024px texture that central band is:
            #   osm_px = 1024 * (4096/8192) = 512px, offset = (1024 - 512) / 2 = 256px
            osm_band = target_size * (4096 / 8192)  # 512px
            osm_offset = (target_size - osm_band) / 2  # 256px
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
                draw_overlay.polygon(poly_points, fill=fill_rgba)
                if outline_width > 0:
                    draw_overlay.line(poly_points + [poly_points[0]], fill=outline_rgba, width=outline_width)
            else:
                draw_overlay.line(poly_points, fill=fill_rgba, width=outline_width if outline_width > 0 else 4)
                
        # Composite the overlay onto base_color_img
        base_color_img = Image.alpha_composite(base_color_img.convert("RGBA"), overlay).convert("RGB")

                
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
        shaded = ls.shade_rgb(rgb_input, elevation=data_resized / 128.0, blend_mode='overlay', vert_exag=1.5, dx=8.0, dy=8.0)
        
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
        <h1>Granja Bonita x4 3D</h1>
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
                <div class="stat-value">4.10 × 4.10 km</div>
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
                <span style="font-size: 13px;">Límite Jugable (4km)</span>
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
        const PLAYABLE_SIZE = 4096;
        const PLAYABLE_OFFSET = (MAP_SIZE - PLAYABLE_SIZE) / 2;
        
        // OSM Bounds and Data (from zoning_map.osm)
        const MIN_LON = {min_lon};
        const MAX_LON = {max_lon};
        const MIN_LAT = {min_lat};
        const MAX_LAT = {max_lat};
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
                heightData[i] = rawHeight / 128.0;
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
                        const dxRes = x - (-1783);
                        const dzRes = z - 1783;
                        if (Math.abs(dxRes) <= 250 && Math.abs(dzRes) <= 250) {{
                            zoneLabel.innerText = "Embalse Noroeste (500x500x15m)";
                            zoneLabel.style.color = "#3b82f6";
                        }} else if (Math.abs(x - (-2008)) <= 25 && Math.abs(z - (-1623)) <= 25) {{
                            zoneLabel.innerText = "Estanque Sur (50x50x3m)";
                            zoneLabel.style.color = "#3b82f6";
                        }} else if (Math.abs(x - (-2025.5)) <= 7.5 && z >= -1648 && z <= 2033) {{
                            zoneLabel.innerText = "Canal de Agua Oeste (15m)";
                            zoneLabel.style.color = "#60a5fa";
                        }} else if (Math.sqrt((x - (-2048))**2 + (z - (-2048))**2) <= 300) {{
                            zoneLabel.innerText = "Pico Suroeste (Elevación 100m)";
                            zoneLabel.style.color = "#fbbf24";
                        }} else if (Math.abs(x - 1679) <= 354 && Math.abs(z - (-1244)) <= 354) {{
                            zoneLabel.innerText = "Cuadrante Llano SE (50 ha)";
                            zoneLabel.style.color = "#ec4899";
                        }} else if (z >= -2048 && z <= -1648) {{
                            zoneLabel.innerText = "Barrera Montañosa Sur";
                            zoneLabel.style.color = "#f59e0b";
                        }} else if (z >= 1048 && z <= 2048) {{
                            zoneLabel.innerText = "Llanura Norte (Área Llana)";
                            zoneLabel.style.color = "#10b981";
                        }} else {{
                            zoneLabel.innerText = "Valle Ondulado (Área Jugable)";
                            zoneLabel.style.color = "#34d399";
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
