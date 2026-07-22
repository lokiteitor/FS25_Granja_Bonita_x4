import os
import time
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# For generating visual maps
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

def val_noise(shape, grid_size, weight, seed=20260608):
    """Generates smooth value noise by upscaling a small random grid using bicubic interpolation."""
    np.random.seed(seed)
    small = np.random.uniform(-1.0, 1.0, size=(grid_size, grid_size)).astype(np.float32)
    temp_img = Image.fromarray(small)
    temp_img = temp_img.resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.array(temp_img) * weight

def main():
    t_start = time.time()
    # Configuration
    playable_size_m = 4096.0  # Playable area (4096x4096m)
    offset_m = 2048.0         # Border offset in meters (2048m on all sides)
    S_m = playable_size_m + 2 * offset_m  # Total map size in meters (8192m)
    S_px = int(S_m)           # Heightmap resolution in pixels (exactly 8192x8192)
    scale_m_to_px = 1.0       # 1 pixel = 1 meter
    
    print(f"=== FS25 {int(S_px/1024)}K New DEM Generator (Exactly {S_px}x{S_px} for {int(playable_size_m/1024)}K Maps) ===")
    
    seed = 20260608
    np.random.seed(seed)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dem_path = os.path.join(script_dir, f"dem_new_{int(S_px/1024)}k.png")
    output_vis_path = os.path.join(script_dir, f"dem_new_visual_{int(S_px/1024)}k.png")
    output_detail_vis_path = os.path.join(script_dir, f"dem_new_visual_detail_{int(S_px/1024)}k.png")
    
    print(f"1. Generating coordinate grids for size {S_px}x{S_px} pixels ({S_m}x{S_m} meters)...")
    y_indices_px, x_indices_px = np.indices((S_px, S_px), dtype=np.float32)
    
    # Convert pixel indices to meter coordinates
    x_m = x_indices_px / scale_m_to_px
    y_m = y_indices_px / scale_m_to_px
    
    print("2. Generating geographic features (slope + background hills)...")
    # Global geographic slope: NW to SE (based on normalized coordinates)
    slope = (x_indices_px / (S_px - 1)) * 8000 + (y_indices_px / (S_px - 1)) * 26000 + 12000
    
    # Background hills noise (low amplitude rolling plains/steppes to match Donetsk/Donets Ridge geography)
    noise_mountains = (
        val_noise((S_px, S_px), 8, 4500, seed=seed+4) +
        val_noise((S_px, S_px), 16, 2000, seed=seed+5) +
        val_noise((S_px, S_px), 32, 500, seed=seed+6)
    )
    
    # Compute distance from the border of the playable area
    # (Playable area is centered: x_m and y_m in [offset_m, S_m - offset_m])
    dx_bg = np.maximum(0.0, np.maximum(offset_m - x_m, x_m - (S_m - offset_m)))
    dy_bg = np.maximum(0.0, np.maximum(offset_m - y_m, y_m - (S_m - offset_m)))
    dist_border_bg = np.sqrt(dx_bg*dx_bg + dy_bg*dy_bg)
    
    # Define weight: 0 inside the playable area, transitions to 1.0 over 1024m outside using a smooth cosine blend
    t_bg = np.minimum(1.0, dist_border_bg / 1024.0)
    w_bg = 0.5 * (1.0 - np.cos(np.pi * t_bg))
    
    # Set the playable area base height
    H_flat = 29000.0
    print(f"   Using a base playable height (H_flat): {H_flat:.1f}")
    
    # Generate the playable area terrain (flat area and undulating area)
    # We create a natural, undulating boundary for the North flat area using a 1D value noise slice
    boundary_noise_1d = val_noise((S_px, S_px), 8, 150.0, seed=seed+15)[0, :]
    boundary_y = offset_m + 1000.0 + boundary_noise_1d[np.newaxis, :] # Broadcast along y_m
    
    # Natural transition width (gentle 400-meter slope)
    trans_w = 400.0
    
    # Distance to the modulated flat boundary
    d_north = y_m - boundary_y
    
    # Weight of the "rest of the map" undulating area (using smooth cosine blend)
    t_y = np.clip(d_north / trans_w, 0.0, 1.0)
    w_rest = 0.5 * (1.0 - np.cos(np.pi * t_y))
    
    # Gentle undulating noise (0 to 10 meters of difference: [-5m, +5m])
    # 1 meter = 128 units
    undulating_noise_raw = (
        val_noise((S_px, S_px), 16, 4.0, seed=seed+11) +
        val_noise((S_px, S_px), 32, 2.0, seed=seed+12) +
        val_noise((S_px, S_px), 64, 0.5, seed=seed+13)
    )
    # Normalize to exact range [-640.0, 640.0]
    n_min, n_max = undulating_noise_raw.min(), undulating_noise_raw.max()
    if n_max > n_min:
        undulating_noise = -640.0 + 1280.0 * (undulating_noise_raw - n_min) / (n_max - n_min)
    else:
        undulating_noise = np.zeros_like(undulating_noise_raw)
        
    # Combine playable terrain: H_flat + undulating variation in the "rest" area
    playable_terrain = H_flat + w_rest * undulating_noise
    
    # Generate the South Mountain barrier along the southern border of the playable area
    # Max height: 100 meters (100 * 128 = 12800 units), extending 400 meters inward (up to y_m = 6144.0 - 400 = 5744.0)
    d_south = (offset_m + playable_size_m) - y_m
    t_m = np.clip(d_south / 400.0, 0.0, 1.0)
    # Cosine blend envelope (1 at the border y_m = 6144, 0 at y_m = 5744)
    w_mountain = 0.5 * (1.0 + np.cos(np.pi * t_m))
    
    # 1D noise slice along X to modulate the height of the mountain ridge (between 40% and 100% of max height)
    m_noise_1d = val_noise((S_px, S_px), 12, 1.0, seed=seed+20)[0, :]
    m_noise_min, m_noise_max = m_noise_1d.min(), m_noise_1d.max()
    if m_noise_max > m_noise_min:
        m_noise_1d = 0.4 + 0.6 * (m_noise_1d - m_noise_min) / (m_noise_max - m_noise_min)
    else:
        m_noise_1d = np.ones_like(m_noise_1d)
    m_height_mod = m_noise_1d[np.newaxis, :]  # Broadcasting
    
    # High-frequency rugged details for a natural mountain feel (about 3 meters amplitude)
    rugged_noise = val_noise((S_px, S_px), 64, 400.0, seed=seed+21)
    
    # Combine the modulated envelope and details
    south_mountain = w_mountain * (100.0 * 128.0 * m_height_mod + rugged_noise)
    south_mountain = np.maximum(0.0, south_mountain) # Ensure no negative adjustments
    
    # Generate a localized mountain peak in the Southwest corner of the playable area
    # Center: (offset_m, offset_m + playable_size_m) = (2048.0, 6144.0)
    # Height: 100 meters (12800 units), Radius of influence: 300 meters
    dx_sw = x_m - offset_m
    dy_sw = y_m - (offset_m + playable_size_m)
    d_sw = np.sqrt(dx_sw**2 + dy_sw**2)
    t_sw = np.clip(d_sw / 300.0, 0.0, 1.0)
    w_sw = 0.5 * (1.0 + np.cos(np.pi * t_sw))
    
    sw_peak = w_sw * (100.0 * 128.0 + rugged_noise)
    sw_peak = np.maximum(0.0, sw_peak)
    
    # Add the mountain and the SW peak to the playable terrain
    playable_terrain = playable_terrain + south_mountain + sw_peak
    
    # Apply the flat Southeast square (50 hectares = 500,000 m^2, side = 707.1m)
    # Positioned 15m from the east playable border and 450m from the south playable border (to prevent mountain overlap)
    sq_size = 707.1
    sq_x1 = offset_m + playable_size_m - 15.0
    sq_x0 = sq_x1 - sq_size
    sq_y1 = offset_m + playable_size_m - 450.0
    sq_y0 = sq_y1 - sq_size
    
    scx = (sq_x0 + sq_x1) / 2.0
    scy = (sq_y0 + sq_y1) / 2.0
    srx = (sq_x1 - sq_x0) / 2.0
    sry = (sq_y1 - sq_y0) / 2.0
    
    # SDF to the square
    dx_sq = np.abs(x_m - scx) - srx
    dy_sq = np.abs(y_m - scy) - sry
    dist_outside_sq = np.sqrt(np.maximum(0.0, dx_sq)**2 + np.maximum(0.0, dy_sq)**2)
    dist_inside_sq = np.minimum(0.0, np.maximum(dx_sq, dy_sq))
    sdf_sq = dist_outside_sq + dist_inside_sq
    
    # 100 meters transition width for smooth blend with surrounding terrain
    blend_w = 100.0
    t_sq = np.clip(sdf_sq / blend_w, 0.0, 1.0)
    w_sq = 0.5 * (1.0 - np.cos(np.pi * t_sq))
    
    # Blend the flat height H_flat into playable_terrain
    playable_terrain = w_sq * playable_terrain + (1.0 - w_sq) * H_flat
    
    # Blend the playable terrain with the background mountain terrain
    # inside playable area (w_bg=0), terrain is playable_terrain
    # outside playable area (w_bg>0), terrain transitions to slope + noise
    terrain = w_bg * (slope + noise_mountains) + (1.0 - w_bg) * playable_terrain
    
    print("3. Smoothing the transition zone at the border...")
    terrain = gaussian_filter(terrain, sigma=6 * scale_m_to_px)
    
    print("4. Carving the northwest reservoir (500x500x15m)...")
    # Carve the rectangular reservoir in the northwest of the playable area
    # Size: 500 x 500 meters, Depth: 15 meters, Location: 15 meters from the northwest playable border (offset_m, offset_m)
    rx0 = offset_m + 15.0
    rx1 = rx0 + 500.0
    ry0 = offset_m + 15.0
    ry1 = ry0 + 500.0
    
    rcx = (rx0 + rx1) / 2.0
    rcy = (ry0 + ry1) / 2.0
    rrx = (rx1 - rx0) / 2.0
    rry = (ry1 - ry0) / 2.0
    
    # SDF to the rectangle
    dx_res = np.abs(x_m - rcx) - rrx
    dy_res = np.abs(y_m - rcy) - rry
    dist_outside = np.sqrt(np.maximum(0.0, dx_res)**2 + np.maximum(0.0, dy_res)**2)
    dist_inside = np.minimum(0.0, np.maximum(dx_res, dy_res))
    sdf_res = dist_outside + dist_inside
    
    # 15 meters bank width for smooth slope
    bank_w = 15.0
    res_depth = 15.0 * 128.0  # 15 meters depth in units
    
    # Compute carving weight: 1.0 deep inside, 0.0 outside
    t_res = np.clip(-sdf_res / bank_w, 0.0, 1.0)
    w_res = 3 * t_res**2 - 2 * t_res**3  # Smoothstep
    
    reservoir_carve = w_res * res_depth
    terrain = terrain - reservoir_carve
    
    print("5. Carving the west water channel (15m)...")
    # Channel from southern mountain boundary (y_m = 5744.0) to reservoir (y_m = 2063.0), 15m from west border (x_m = 2063.0)
    x_c = offset_m + 15.0 + 7.5
    y_start_ch = offset_m + 15.0
    y_end_ch = offset_m + playable_size_m - 400.0
    
    # Segment projection
    t_ch_segment = np.clip((y_m - y_start_ch) / (y_end_ch - y_start_ch), 0.0, 1.0)
    proj_y = y_start_ch + t_ch_segment * (y_end_ch - y_start_ch)
    proj_x = x_c
    
    # Distance to segment
    dist_segment = np.sqrt((x_m - proj_x)**2 + (y_m - proj_y)**2)
    
    # 15m wide channel means 7.5m radius.
    # Center 7m is flat bottom (3.5m radius), then 4m transition to 0.
    t_ch = np.clip((dist_segment - 3.5) / 4.0, 0.0, 1.0)
    w_ch = 0.5 * (1.0 + np.cos(np.pi * t_ch))
    
    # Depth: 4 meters (4 * 128 = 512 units)
    channel_depth = 4.0 * 128.0
    channel_carve = w_ch * channel_depth
    terrain = terrain - channel_carve
    
    print("5.2 Carving the southern pond at the end of the channel (50x50x3m)...")
    # Center (pcx, pcy) = (offset_m + 15.0 + 25.0, offset_m + playable_size_m - 400.0 - 25.0)
    # Range in X: [offset_m + 15.0, offset_m + 65.0], Range in Y: [offset_m + playable_size_m - 450.0, offset_m + playable_size_m - 400.0]
    px0 = offset_m + 15.0
    px1 = px0 + 50.0
    py0 = offset_m + playable_size_m - 450.0
    py1 = offset_m + playable_size_m - 400.0
    
    pcx = (px0 + px1) / 2.0
    pcy = (py0 + py1) / 2.0
    prx = (px1 - px0) / 2.0
    pry = (py1 - py0) / 2.0
    
    # SDF to the rectangle
    dx_pond = np.abs(x_m - pcx) - prx
    dy_pond = np.abs(y_m - pcy) - pry
    dist_outside_pond = np.sqrt(np.maximum(0.0, dx_pond)**2 + np.maximum(0.0, dy_pond)**2)
    dist_inside_pond = np.minimum(0.0, np.maximum(dx_pond, dy_pond))
    sdf_pond = dist_outside_pond + dist_inside_pond
    
    # Bank width: 5m, Depth: 3m
    bank_w_pond = 5.0
    pond_depth = 3.0 * 128.0
    t_pond = np.clip(-sdf_pond / bank_w_pond, 0.0, 1.0)
    w_pond = 3 * t_pond**2 - 2 * t_pond**3
    pond_carve = w_pond * pond_depth
    terrain = terrain - pond_carve
    
    # Clamp terrain to valid 16-bit range
    terrain = np.clip(terrain, 2000.0, 62000.0)
    
    print(f"6. Saving final DEM heightmap to '{output_dem_path}'...")
    img_out = Image.fromarray(terrain.astype(np.int32), mode="I")
    img_out.save(output_dem_path)
    print(f"   Saved heightmap successfully (Min={terrain.min():.1f}, Max={terrain.max():.1f}).")
    
    print("7. Generating visual maps...")
    # Calculate scale dynamically to downsample to 1024x1024 visual dimension
    vis_scale = max(1, int(S_px / 1024))
    terrain_vis = terrain[::vis_scale, ::vis_scale]
    
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.shade(terrain_vis, cmap=plt.get_cmap('terrain'), vert_exag=0.12, blend_mode='overlay')
    
    # Define scale from meters to visualization coordinates: (scale_m_to_px / vis_scale)
    scale_m_to_vis = scale_m_to_px / vis_scale
    
    # --- Map 1: Full Map View ---
    print("   Generating full map visualization...")
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.imshow(hs)
    ax.axis('off')
    ax.set_title(f"Full {int(S_px/1024)}K DEM Map (100% Flat Playable Area)", fontsize=16, fontweight='bold', pad=15)
    
    playable_size_vis = (S_m - 2 * offset_m) * scale_m_to_vis
    playable_start_vis = offset_m * scale_m_to_vis
    rect_playable = plt.Rectangle((playable_start_vis, playable_start_vis), playable_size_vis, playable_size_vis, 
                                  fill=False, edgecolor='white', linewidth=2, linestyle='--', 
                                  label=f'Playable Border ({int(playable_size_m/1000)}km)')
    ax.add_patch(rect_playable)
    
    plt.savefig(output_vis_path, bbox_inches='tight')
    plt.close()
    print(f"   Saved full visualization to '{output_vis_path}'.")
    
    # --- Map 2: Zoomed-in Playable Area View ---
    print("   Generating detailed playable area visualization...")
    p_start = int(offset_m / vis_scale)
    p_end = int((S_m - offset_m) / vis_scale)
    hs_detail = hs[p_start:p_end, p_start:p_end]
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(hs_detail)
    
    # Draw natural flat North area boundary line
    boundary_noise_vis = boundary_noise_1d[::vis_scale]
    boundary_y_vis = (offset_m + 1000.0 + boundary_noise_vis) / vis_scale
    boundary_y_vis_local = boundary_y_vis[p_start:p_end] - p_start
    ax.plot(np.arange(len(boundary_y_vis_local)), boundary_y_vis_local, color='yellow', linestyle='--', linewidth=1.5, label='Flat North boundary (Natural)')
    
    # Draw South Mountain boundary line
    y_mountain_start = (playable_size_m - 400.0) / vis_scale
    ax.axhline(y=y_mountain_start, color='orange', linestyle=':', linewidth=1.5, label='South Mountain boundary (400m)')
    
    # Draw Southwest peak indicator (radius 300m)
    r_vis = 300.0 / vis_scale
    circ_peak = plt.Circle((0, playable_size_m / vis_scale), r_vis,
                           fill=False, edgecolor='orange', linewidth=2.0,
                           linestyle='--', label='SW Peak (100m)')
    ax.add_patch(circ_peak)
    
    # Draw flat Southeast square (50 hectares)
    sq_size = 707.1
    sq_x1 = offset_m + playable_size_m - 15.0
    sq_x0 = sq_x1 - sq_size
    sq_y1 = offset_m + playable_size_m - 450.0
    sq_y0 = sq_y1 - sq_size
    
    sq_x0_vis = (sq_x0 - offset_m) / vis_scale
    sq_y0_vis = (sq_y0 - offset_m) / vis_scale
    sq_w_vis = sq_size / vis_scale
    sq_h_vis = sq_size / vis_scale
    
    rect_sq = plt.Rectangle((sq_x0_vis, sq_y0_vis), sq_w_vis, sq_h_vis,
                            fill=False, edgecolor='cyan', linewidth=2.0,
                            linestyle='-', label='SE Flat Square (50 ha)')
    ax.add_patch(rect_sq)
    

    
    # Draw reservoir rectangle
    rx0_vis = 15.0 / vis_scale
    ry0_vis = 15.0 / vis_scale
    rw_vis = 500.0 / vis_scale
    rh_vis = 500.0 / vis_scale
    rect_res = plt.Rectangle((rx0_vis, ry0_vis), rw_vis, rh_vis,
                             fill=False, edgecolor='red', linewidth=2.0,
                             label='Reservoir (500x500x15m)')
    ax.add_patch(rect_res)
    
    # Draw channel line (blue, width representing 15m, separated 15m from west border)
    y_ch_start = 15.0 / vis_scale
    y_ch_end = (playable_size_m - 400.0) / vis_scale
    x_ch_center = (15.0 + 7.5) / vis_scale
    ax.plot([x_ch_center, x_ch_center], [y_ch_start, y_ch_end], color='blue', linewidth=2.0, label='Water Channel (15m)')
    
    # Draw southern pond rectangle (50x50m)
    px0_vis = 15.0 / vis_scale
    py0_vis = (playable_size_m - 450.0) / vis_scale
    pw_vis = 50.0 / vis_scale
    ph_vis = 50.0 / vis_scale
    rect_pond = plt.Rectangle((px0_vis, py0_vis), pw_vis, ph_vis,
                              fill=False, edgecolor='red', linewidth=1.5,
                              linestyle='-', label='South Pond (50x50x3m)')
    ax.add_patch(rect_pond)
    
    ax.axis('off')
    ax.set_title(f"Detailed Playable Area ({int(playable_size_m)}x{int(playable_size_m)}m Sandbox)", fontsize=16, fontweight='bold', pad=15)
    
    plt.savefig(output_detail_vis_path, bbox_inches='tight')
    plt.close()
    print(f"   Saved detailed visualization to '{output_detail_vis_path}'.")
    
    t_end = time.time()
    print(f"\n=== Script Completed Successfully in {t_end - t_start:.2f} seconds ===")
    print(f"Output files:")
    print(f" - New Heightmap: {output_dem_path}")
    print(f" - Full Map Visual: {output_vis_path}")
    print(f" - Detailed Visual: {output_detail_vis_path}")

if __name__ == "__main__":
    main()
