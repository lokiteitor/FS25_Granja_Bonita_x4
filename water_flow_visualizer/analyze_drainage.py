#!/usr/bin/env python3
import os
import json
import numpy as np
from PIL import Image

def main():
    print("=== Analyzing Terrain Drainage and Flow Paths ===")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # We will use the 1024x1024 heightmap for performance and consistency with the 3D viewer
    heightmap_path = os.path.join(project_root, "visualizer", "dem_1024_rgb16.png")
    
    if not os.path.exists(heightmap_path):
        print(f"Error: {heightmap_path} not found. Please run the visualizer asset generator first.")
        # Fallback to generating it if needed
        import sys
        sys.exit(1)
        
    print(f"Loading heightmap from: {heightmap_path}")
    img = Image.open(heightmap_path)
    data = np.array(img)
    
    # Decode 16-bit RGB encoded heightmap
    # Red = height % 256, Green = height // 256
    r = data[:, :, 0].astype(np.float32)
    g = data[:, :, 1].astype(np.float32)
    heights = (r + g * 256.0) / 128.0
    
    H, W = heights.shape
    print(f"Heightmap dimensions: {W}x{H}")
    
    # Playable area bounds (centered 4096m in 8192m map)
    # At 1024x1024 resolution, the playable area is the central 512x512 pixels.
    # From index 256 to 768.
    offset = 256
    size = 512
    p_x0, p_x1 = offset, offset + size
    p_y0, p_y1 = offset, offset + size
    
    # Identify key hydrological features based on coordinates & height carving in generate_new_dem_8k.py
    # H_flat is 29000/128 = 226.5625m
    H_flat = 226.5625
    
    # Let's map pixels to their hydrological classifications:
    # 0 = Normal land
    # 1 = Reservoir (NW corner: x in [256, 320], y in [256, 320], height < 226.0)
    # 2 = West Channel & Southern Pond (x < 275, height < 226.0)
    
    feature_map = np.zeros((H, W), dtype=np.uint8)
    
    for y in range(p_y0, p_y1):
        for x in range(p_x0, p_x1):
            h = heights[y, x]
            if h < H_flat - 0.5: # Carved out features
                if x <= 325 and y <= 325:
                    feature_map[y, x] = 1 # Reservoir
                elif x <= 275:
                    feature_map[y, x] = 2 # West Channel / Pond
    
    print(f"Hydrological features mapped:")
    print(f"  Reservoir cells: {np.sum(feature_map == 1)}")
    print(f"  West Channel / Pond cells: {np.sum(feature_map == 2)}")
    
    # Flow direction routing (D8 algorithm)
    # Neighbors layout:
    # 0: (x+1, y)   1: (x+1, y+1)  2: (x, y+1)   3: (x-1, y+1)
    # 4: (x-1, y)   5: (x-1, y-1)  6: (x, y-1)   7: (x+1, y-1)
    dx = [1, 1, 0, -1, -1, -1,  0,  1]
    dy = [0, 1, 1,  1,  0, -1, -1, -1]
    
    # Precompute steepest descent for each pixel
    # flow_to stores the (ny, nx) coords of the neighbor, or (-1, -1) if it's a sink
    flow_to_y = -np.ones((H, W), dtype=np.int32)
    flow_to_x = -np.ones((H, W), dtype=np.int32)
    is_sink = np.zeros((H, W), dtype=bool)
    
    for y in range(1, H-1):
        for x in range(1, W-1):
            h_curr = heights[y, x]
            
            # Find neighbor with steepest slope
            max_slope = 0.0
            best_idx = -1
            
            for i in range(8):
                nx, ny = x + dx[i], y + dy[i]
                h_neigh = heights[ny, nx]
                
                # Distance: 1 for orthogonal, sqrt(2) for diagonal
                dist = 1.41421356 if (dx[i] != 0 and dy[i] != 0) else 1.0
                slope = (h_curr - h_neigh) / dist
                
                if slope > max_slope:
                    max_slope = slope
                    best_idx = i
            
            if best_idx != -1:
                flow_to_y[y, x] = y + dy[best_idx]
                flow_to_x[y, x] = x + dx[best_idx]
            else:
                is_sink[y, x] = True
                
    # Now trace the drainage path for every pixel in the playable area
    # Destination classification:
    # 0 = Stuck in local sink (puddle)
    # 1 = Drains into Reservoir
    # 2 = Drains into West Channel / Pond
    # 3 = Drains off-map (non-playable border)
    
    dest_map = -np.ones((H, W), dtype=np.int32)
    sinks_list = []
    
    # We trace paths. To avoid infinite loops in flat areas, we keep track of visited cells in current path
    print("Tracing flow paths for all playable cells...")
    
    # To store sink locations and how many cells drain into them
    sink_drainage_count = {}
    
    for y in range(p_y0, p_y1):
        for x in range(p_x0, p_x1):
            # If it's already a water body cell, it drains to itself
            if feature_map[y, x] == 1:
                dest_map[y, x] = 1
                continue
            if feature_map[y, x] == 2:
                dest_map[y, x] = 2
                continue
                
            path = [(y, x)]
            visited = {(y, x)}
            cy, cx = y, x
            
            terminated = False
            dest = 0 # default: stuck in sink
            
            while not terminated:
                ny = flow_to_y[cy, cx]
                nx = flow_to_x[cy, cx]
                
                # Out of bounds check
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    dest = 3 # Off-map
                    terminated = True
                    break
                    
                # Check if it entered a water body
                if feature_map[ny, nx] == 1:
                    dest = 1 # Reservoir
                    terminated = True
                    break
                elif feature_map[ny, nx] == 2:
                    dest = 2 # West Channel
                    terminated = True
                    break
                
                # Check if it's a sink
                if is_sink[ny, nx]:
                    dest = 0 # Stuck in sink
                    sink_loc = (ny, nx)
                    terminated = True
                    break
                    
                # Loop detection (flat areas or numerical precision loops)
                if (ny, nx) in visited:
                    dest = 0 # Stuck in local loop/sink
                    # Find lowest point in the loop to represent the sink
                    lowest_y, lowest_x = cy, cx
                    for py, px in path:
                        if heights[py, px] < heights[lowest_y, lowest_x]:
                            lowest_y, lowest_x = py, px
                    sink_loc = (lowest_y, lowest_x)
                    terminated = True
                    break
                    
                # Continue tracing
                cy, cx = ny, nx
                path.append((cy, cx))
                visited.add((cy, cx))
            
            if dest == 0:
                # If the sink is located on the boundary of the playable area, it has successfully drained off-map!
                if sink_loc[0] == p_y0 or sink_loc[0] == p_y1 - 1 or sink_loc[1] == p_x0 or sink_loc[1] == p_x1 - 1:
                    dest_map[y, x] = 3
                else:
                    dest_map[y, x] = 0
                    # Record sink usage
                    if sink_loc not in sink_drainage_count:
                        sink_drainage_count[sink_loc] = 0
                    sink_drainage_count[sink_loc] += 1
            else:
                dest_map[y, x] = dest
                
    # Collect statistics
    total_playable_cells = size * size
    reservoir_drains = np.sum(dest_map[p_y0:p_y1, p_x0:p_x1] == 1)
    channel_drains = np.sum(dest_map[p_y0:p_y1, p_x0:p_x1] == 2)
    offmap_drains = np.sum(dest_map[p_y0:p_y1, p_x0:p_x1] == 3)
    sink_drains = np.sum(dest_map[p_y0:p_y1, p_x0:p_x1] == 0)
    
    pct_reservoir = (reservoir_drains / total_playable_cells) * 100.0
    pct_channel = (channel_drains / total_playable_cells) * 100.0
    pct_offmap = (offmap_drains / total_playable_cells) * 100.0
    pct_sink = (sink_drains / total_playable_cells) * 100.0
    
    print("\n--- Drainage Statistics ---")
    print(f"Total Playable Area Cells: {total_playable_cells}")
    print(f"Drains to NW Reservoir: {reservoir_drains} ({pct_reservoir:.2f}%)")
    print(f"Drains to West Channel & Pond: {channel_drains} ({pct_channel:.2f}%)")
    print(f"Drains Off-Map: {offmap_drains} ({pct_offmap:.2f}%)")
    print(f"Trapped in local sinks (depressions): {sink_drains} ({pct_sink:.2f}%)")
    
    # Process significant sinks (e.g., those draining at least 100 cells, which is ~100m² of land)
    significant_sinks = []
    for loc, count in sink_drainage_count.items():
        sy, sx = loc
        sh = heights[sy, sx]
        if count >= 100:
            significant_sinks.append({
                "x": int(sx),
                "y": int(sy),
                "height": float(sh),
                "drained_cells": int(count),
                "drained_percentage": float((count / total_playable_cells) * 100.0)
            })
            
    # Sort significant sinks by drainage count descending
    significant_sinks = sorted(significant_sinks, key=lambda s: s["drained_cells"], reverse=True)
    
    print(f"\nFound {len(significant_sinks)} significant depressions (draining >= 100 cells):")
    for i, s in enumerate(significant_sinks[:10]):
        print(f"  Sink {i+1}: Location ({s['x']}, {s['y']}), Elev={s['height']:.2f}m, Drains {s['drained_cells']} cells ({s['drained_percentage']:.2f}%)")
        
    # Write analysis data to JSON
    analysis_data = {
        "summary": {
            "total_cells": int(total_playable_cells),
            "reservoir": {
                "cells": int(reservoir_drains),
                "percentage": float(pct_reservoir)
            },
            "channel": {
                "cells": int(channel_drains),
                "percentage": float(pct_channel)
            },
            "offmap": {
                "cells": int(offmap_drains),
                "percentage": float(pct_offmap)
            },
            "sinks": {
                "cells": int(sink_drains),
                "percentage": float(pct_sink)
            }
        },
        "significant_sinks": significant_sinks,
        # We can also export a downsampled destination map for fast rendering
        # Let's save a 128x128 grid for overlay rendering to save JSON size
        "dest_grid_128": dest_map[::8, ::8].tolist()
    }
    
    output_json_path = os.path.join(current_dir, "drainage_data.json")
    with open(output_json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"\nDrainage data saved to: {output_json_path}")
    
if __name__ == "__main__":
    main()
