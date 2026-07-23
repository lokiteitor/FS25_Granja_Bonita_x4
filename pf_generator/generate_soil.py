#!/usr/bin/env python3
"""
generate_soil.py - FS25 Precision Farming Soil Map Generator
============================================================
Generates two files:
  - soilMap.png      : The raw game map (indexed P mode, values 0-3)
  - soilMap_vis.png  : High-quality dark-themed visualization with legend

This script produces natural organic soil distributions using multi-scale noise
and guarantees realistic pixel counts matching FS25 Precision Farming.
"""

import os
import argparse
import random
import time
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description="Generate natural soil maps for FS25 Precision Farming")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed for the random generator (default: random based on time)")
    parser.add_argument("-o", "--output-dir", type=str, default=".",
                        help="Output directory for generated images (default: current directory)")
    parser.add_argument("--scale-coarse", type=float, default=120.0,
                        help="Sigma for coarse geological features (default: 120)")
    parser.add_argument("--scale-medium", type=float, default=40.0,
                        help="Sigma for medium geological features (default: 40)")
    parser.add_argument("--scale-fine", type=float, default=12.0,
                        help="Sigma for fine details/borders (default: 12)")
    parser.add_argument("--pixel-scale", type=float, default=2.0,
                        help="Scale ratio: meters per pixel (default: 2.0, i.e. 1px = 2m for a 4096x4096m map)")
    return parser.parse_args()


def generate_noise_field(seed, size, sigma_coarse, sigma_medium, sigma_fine):
    """
    Generates a standardized multi-scale noise field.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Coarse layer (large structural features)
    noise_c = rng.standard_normal((size, size))
    noise_c = ndimage.gaussian_filter(noise_c, sigma=sigma_coarse)
    noise_c = (noise_c - np.mean(noise_c)) / (np.std(noise_c) + 1e-8)
    
    # 2. Medium layer (interspersed patches)
    noise_m = rng.standard_normal((size, size))
    noise_m = ndimage.gaussian_filter(noise_m, sigma=sigma_medium)
    noise_m = (noise_m - np.mean(noise_m)) / (np.std(noise_m) + 1e-8)
    
    # 3. Fine layer (edge texture)
    noise_f = rng.standard_normal((size, size))
    noise_f = ndimage.gaussian_filter(noise_f, sigma=sigma_fine)
    noise_f = (noise_f - np.mean(noise_f)) / (np.std(noise_f) + 1e-8)
    
    # Combined with weights
    combined = noise_c * 0.65 + noise_m * 0.28 + noise_f * 0.07
    return (combined - np.mean(combined)) / (np.std(combined) + 1e-8)

def main():
    args = parse_args()
    
    # Determine seed
    if args.seed is None:
        # Use system entropy or milliseconds to ensure a different seed in each run
        seed = int((time.time() * 1000) % 99999999)
    else:
        seed = args.seed
        
    print(f"[*] Starting soil map generation using seed: {seed}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Size of the map (2048 x 2048)
    S = 2048
    
    # Generate two independent noise fields for nested categorization
    print("[*] Generating multi-scale noise fields...")
    noise_zone = generate_noise_field(seed, S, args.scale_coarse, args.scale_medium, args.scale_fine)
    noise_patch = generate_noise_field(seed + 12345, S, args.scale_coarse * 0.7, args.scale_medium * 0.7, args.scale_fine)
    
    # Target proportions (Adjusted: Arena Limosa (0) and Arcilla Limosa (3) reduced to 10.0%, Franco (2) kept at 38.55%):
    # 0 (Arena Limosa / Loamy Sand): 10.0% (419430 px)
    # 1 (Franco Arenoso / Sandy Loam): 41.45% (1738539 px)
    # 2 (Franco / Loam): 38.55% (1616904 px)
    # 3 (Arcilla Limosa / Silty Clay): 10.0% (419430 px)
    
    # Total pixels = 4,194,304
    # Zone B = Franco + Arcilla Limosa = 48.55%
    # Zone A = Arena Limosa + Franco Arenoso = 51.45%
    pct_zone_B = 48.55
    
    print("[*] Performing nested percentile thresholding...")
    # Step 1: Split noise_zone into Zone A and Zone B
    threshold_zone = np.percentile(noise_zone, pct_zone_B)
    zone_B_mask = (noise_zone < threshold_zone)
    zone_A_mask = ~zone_B_mask
    
    # Step 2: Split Zone A (Loamy Sand 10.0% vs Sandy Loam 41.45%)
    # Inside Zone A, Loamy Sand proportion is 10.0 / 51.45 = 19.436346%
    pct_val0_in_A = 19.436346
    noise_patch_A = noise_patch[zone_A_mask]
    threshold_A = np.percentile(noise_patch_A, pct_val0_in_A)
    
    # Step 3: Split Zone B (Loam 38.55% vs Silty Clay 10.0%)
    # Inside Zone B, Loam proportion is 38.55 / 48.55 = 79.402678%
    pct_val2_in_B = 79.402678
    noise_patch_B = noise_patch[zone_B_mask]
    threshold_B = np.percentile(noise_patch_B, pct_val2_in_B)
    
    # Initialize soil map array
    soil_map = np.zeros((S, S), dtype=np.uint8)
    
    # Assign soil types
    soil_map[zone_A_mask] = np.where(noise_patch[zone_A_mask] < threshold_A, 0, 1)
    soil_map[zone_B_mask] = np.where(noise_patch[zone_B_mask] < threshold_B, 2, 3)
    
    # Check pixel distribution
    unique, counts = np.unique(soil_map, return_counts=True)
    dist = dict(zip(unique, counts))
    
    # Soil Metadata
    soil_meta = {
        0: {"name_es": "Arena Limosa", "name_en": "Loamy Sand", "yield": "75%", "color": (220, 185, 80)},
        1: {"name_es": "Franco Arenoso", "name_en": "Sandy Loam", "yield": "100%", "color": (180, 130, 60)},
        2: {"name_es": "Franco", "name_en": "Loam", "yield": "125%", "color": (70, 150, 50)},
        3: {"name_es": "Arcilla Limosa", "name_en": "Silty Clay", "yield": "80%", "color": (120, 70, 160)}
    }
    
    print("\n[+] Pixel counts and statistics:")
    print("-" * 85)
    print(f"{'Val':<3} | {'Tipo de Suelo (ES)':<20} | {'Tipo de Suelo (EN)':<15} | {'Hectáreas':<12} | {'Porcentaje':<10}")
    print("-" * 85)
    # 1 pixel = pixel_scale meters -> 1 pixel area = (pixel_scale)^2 m^2
    pixel_area_m2 = args.pixel_scale ** 2
    total_area_ha = (S * S * pixel_area_m2) / 10000.0
    
    for v in range(4):
        cnt = dist.get(v, 0)
        pct = (cnt / (S * S)) * 100
        ha = (cnt * pixel_area_m2) / 10000.0
        meta = soil_meta[v]
        print(f"{v:<3} | {meta['name_es']:<20} | {meta['name_en']:<15} | {ha:<12.2f} | {pct:<10.2f}%")
    print("-" * 85)
    print(f"Total: {S*S} px | {total_area_ha:.2f} ha\n")

    
    # ----------------------------------------------------
    # Save soilMap.png (Raw Game Map - Indexed Palette P)
    # ----------------------------------------------------
    print("[*] Saving soilMap.png...")
    img_raw = Image.fromarray(soil_map, mode='P')
    # Exact original palette
    palette_data = [1, 1, 1, 2, 2, 2, 0, 0, 0, 3, 3, 3]
    palette_data += [0] * (768 - len(palette_data))
    img_raw.putpalette(palette_data)
    
    output_raw_path = os.path.join(args.output_dir, "soilMap.png")
    img_raw.save(output_raw_path)
    print(f"[+] Raw soil map saved to: {output_raw_path}")
    
    # ----------------------------------------------------
    # Save soilMap_vis.png (RGB Visualization + Legend)
    # ----------------------------------------------------
    print("[*] Creating soilMap_vis.png...")
    vis_h = S + 200
    img_vis = Image.new("RGB", (S, vis_h), (20, 24, 33)) # Dark theme background
    
    # 1. Fill the map section (top 2048 x 2048)
    # Create RGB representation of the map
    map_rgb = np.zeros((S, S, 3), dtype=np.uint8)
    for v in range(4):
        map_rgb[soil_map == v] = soil_meta[v]["color"]
    
    img_vis.paste(Image.fromarray(map_rgb, mode="RGB"), (0, 0))
    
    # 2. Draw the legend section (bottom 2048 x 200)
    draw = ImageDraw.Draw(img_vis)
    
    # Draw a thin division line
    draw.line([(0, S), (S, S)], fill=(45, 52, 68), width=2)
    
    # Load fonts
    try:
        font_path_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        font_path_reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        
        font_title = ImageFont.truetype(font_path_bold, 24)
        font_subtitle = ImageFont.truetype(font_path_reg, 14)
        font_bold = ImageFont.truetype(font_path_bold, 17)
        font_regular = ImageFont.truetype(font_path_reg, 14)
        font_small = ImageFont.truetype(font_path_reg, 12)
    except Exception:
        # Fallback to default if fonts are missing
        font_title = font_subtitle = font_bold = font_regular = font_small = ImageFont.load_default()
        
    # Draw Title and Seed Info
    draw.text((40, S + 35), "DISTRIBUCIÓN DE SUELOS", fill=(255, 255, 255), font=font_title)
    draw.text((40, S + 70), "Precision Farming - Farming Simulator 25", fill=(150, 160, 180), font=font_subtitle)
    draw.text((40, S + 105), f"Semilla (Seed): {seed}", fill=(200, 180, 100), font=font_subtitle)
    # Format pixel scale label. If integer, show without decimal places.
    scale_val_str = f"{int(args.pixel_scale)}" if args.pixel_scale.is_integer() else f"{args.pixel_scale}"
    draw.text((40, S + 130), f"Escala: 1 píxel = {scale_val_str} metros", fill=(120, 130, 150), font=font_small)
    draw.text((40, S + 150), f"Área total: {total_area_ha:.2f} Hectáreas", fill=(120, 130, 150), font=font_small)

    
    # Draw the 4 soil columns
    start_x = 650
    col_width = 340
    y_offset = S + 35
    
    for v in range(4):
        col_x = start_x + v * col_width
        meta = soil_meta[v]
        cnt = dist.get(v, 0)
        pct = (cnt / (S * S)) * 100
        ha = (cnt * pixel_area_m2) / 10000.0

        
        # Color box
        box_coords = [col_x, y_offset + 5, col_x + 32, y_offset + 37]
        draw.rectangle(box_coords, fill=meta["color"], outline=(255, 255, 255), width=1)
        
        # Text details
        text_x = col_x + 45
        draw.text((text_x, y_offset), meta["name_es"], fill=(255, 255, 255), font=font_bold)
        draw.text((text_x, y_offset + 25), f"({meta['name_en']})", fill=(160, 170, 190), font=font_small)
        
        # Rendimiento & stats
        y_pot_str = "Rendimiento: " + meta["yield"]
        if v == 2:  # Highlight Loam
            y_pot_str += " \u2728"
            
        draw.text((text_x, y_offset + 48), y_pot_str, fill=(200, 210, 200), font=font_regular)
        draw.text((text_x, y_offset + 70), f"Área: {ha:.2f} ha ({pct:.2f}%)", fill=(220, 220, 220), font=font_regular)
        draw.text((text_x, y_offset + 90), f"Píxeles: {cnt:,}", fill=(130, 140, 150), font=font_small)
        
    output_vis_path = os.path.join(args.output_dir, "soilMap_vis.png")
    img_vis.save(output_vis_path)
    print(f"[+] Visualization soil map saved to: {output_vis_path}")
    print("[*] Successfully generated both images.")

if __name__ == "__main__":
    main()
