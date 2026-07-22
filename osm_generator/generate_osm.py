#!/usr/bin/env python3
import os
import math
import xml.etree.ElementTree as ET
import numpy as np

# Non-interactive matplotlib backend for running on headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches

def main():
    # 1. Geographic and coordinate configuration
    # Map center as requested
    lat_c = 47.68780500
    lon_c = 37.73548700
    
    # Playable area dimensions: 4096 x 4096 meters
    size_m = 4096.0
    half_size = size_m / 2.0
    
    # Earth radius in meters (WGS-84 equatorial)
    R = 6378137.0
    
    # Calculate geographical bounds for the 4096x4096m area
    dlat = (half_size / R) * (180.0 / math.pi)
    dlon = (half_size / (R * math.cos(math.radians(lat_c)))) * (180.0 / math.pi)
    
    min_lat = lat_c - dlat
    max_lat = lat_c + dlat
    min_lon = lon_c - dlon
    max_lon = lon_c + dlon
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    output_osm_path = os.path.join(output_dir, "zoning_map.osm")
    output_png_path = os.path.join(output_dir, "zoning_map_preview.png")
    
    print("=== OSM Zoning Generator ===")
    print(f"Center Coordinate: {lat_c:.8f}, {lon_c:.8f}")
    print(f"Dimensions: {int(size_m)}x{int(size_m)} meters")
    print(f"Calculated geographical bounding box (Playable Area):")
    print(f"  Latitude:  {min_lat:.8f} to {max_lat:.8f}")
    print(f"  Longitude: {min_lon:.8f} to {max_lon:.8f}")
    
    # Converter function from local meters (x, y) to (lat, lon)
    # x range: [0, 4096] (0 = West, 4096 = East)
    # y range: [0, 4096] (0 = North, 4096 = South)
    def to_geo(x, y):
        lon = min_lon + (x / size_m) * (max_lon - min_lon)
        lat = max_lat - (y / size_m) * (max_lat - min_lat)
        return lat, lon

    # 2. Define features (based on dem_generator coordinates)
    #
    # Feature 1: Southeast Flat Square (50 hectares = 707.1m x 707.1m)
    # In generate_new_dem_8k.py:
    # x in [3373.9, 4081.0], y in [2938.9, 3646.0]
    # Tagged as: landuse=farmyard
    sq_points = [
        to_geo(3373.9, 2938.9),
        to_geo(4081.0, 2938.9),
        to_geo(4081.0, 3646.0),
        to_geo(3373.9, 3646.0),
        to_geo(3373.9, 2938.9) # closed loop
    ]
    
    # Feature 2: South Mountain area (400 meters depth from south playable border)
    # In generate_new_dem_8k.py:
    # y in [3696.0, 4096.0], x in [0.0, 4096.0]
    # Tagged as: landuse=farmyard, natural=wood
    mountain_points = [
        to_geo(0.0, 3696.0),
        to_geo(4096.0, 3696.0),
        to_geo(4096.0, 4096.0),
        to_geo(0.0, 4096.0),
        to_geo(0.0, 3696.0) # closed loop
    ]
    
    # Feature 3: Northwest Reservoir (500m x 500m)
    # In generate_new_dem_8k.py:
    # x in [15.0, 515.0], y in [15.0, 515.0]
    # Tagged as: landuse=farmyard
    reservoir_points = [
        to_geo(15.0, 15.0),
        to_geo(515.0, 15.0),
        to_geo(515.0, 515.0),
        to_geo(15.0, 515.0),
        to_geo(15.0, 15.0) # closed loop
    ]
    
    # Feature 4: West Water Channel (15m width)
    # In generate_new_dem_8k.py:
    # x in [15.0, 30.0], y in [15.0, 3696.0]
    # Tagged as: landuse=farmyard, natural=wood
    channel_points = [
        to_geo(15.0, 15.0),
        to_geo(30.0, 15.0),
        to_geo(30.0, 3696.0),
        to_geo(15.0, 3696.0),
        to_geo(15.0, 15.0) # closed loop
    ]
    
    # Feature 5: Southern Pond (50m x 50m pond at the end of the channel)
    # In generate_new_dem_8k.py:
    # x in [15.0, 65.0], y in [3646.0, 3696.0]
    # Tagged as: landuse=farmyard, natural=wood
    pond_points = [
        to_geo(15.0, 3646.0),
        to_geo(65.0, 3646.0),
        to_geo(65.0, 3696.0),
        to_geo(15.0, 3696.0),
        to_geo(15.0, 3646.0) # closed loop
    ]
    
    # Feature 6: North Primary Highway (runs East to West, along the south border of the Northwest Reservoir)
    # Reservoir is y in [15.0, 515.0], so south border of reservoir is at y = 515.0
    # Runs from East (x=4096.0) to West (x=0.0)
    highway_north_points = [
        to_geo(4096.0, 515.0),
        to_geo(0.0, 515.0)
    ]
    
    # Feature 7: South-Mid Primary Highway (runs East to West, north of the 50-hectare Flat Square)
    # Square north border is at y = 2938.9
    # Runs from East (x=4096.0) to West (x=0.0)
    highway_south_mid_points = [
        to_geo(4096.0, 2938.9),
        to_geo(0.0, 2938.9)
    ]
    
    # Feature 8: Diagonal Sinuous Primary Highway (runs North to South, passes East of Reservoir, West of Flat Square)
    # Starts at the north highway intersection (y = 515.0) and ends at the south highway intersection (y = 2938.9)
    # Modulated by a sine wave to make it sinuous
    highway_diagonal_sinuous_points = []
    num_steps = 32
    y_start_sinuous = 515.0
    y_end_sinuous = 2938.9
    x_start = 650.0
    x_end = 3200.0
    amp = 150.0
    wavelength = 1500.0
    for i in range(num_steps + 1):
        y_val = y_start_sinuous + (i / num_steps) * (y_end_sinuous - y_start_sinuous)
        x_diag = x_start + (y_val / size_m) * (x_end - x_start)
        x_val = x_diag + amp * math.sin(2.0 * math.pi * y_val / wavelength)
        highway_diagonal_sinuous_points.append(to_geo(x_val, y_val))
    
    # Feature 9: East-Reservoir Vertical Highway (runs North to South, ending at the North Highway)
    # Starts at north border (y = 0.0) and ends at North Highway (y = 515.0), at x = 515.0 (east of reservoir)
    highway_east_reservoir_points = [
        to_geo(515.0, 0.0),
        to_geo(515.0, 515.0)
    ]
    
    # Feature 10: Town Area (Pueblo)
    # 30 Hectares = 300,000 m2. Located east of reservoir past vertical highway.
    # x in [548.0, 1204.45], y in [25.0, 482.0]
    town_area_points = [
        to_geo(548.0, 25.0),
        to_geo(1204.45, 25.0),
        to_geo(1204.45, 482.0),
        to_geo(548.0, 482.0),
        to_geo(548.0, 25.0) # closed loop
    ]
    
    # Feature 11: Town Grid Secondary Roads
    town_road_v1_points = [to_geo(766.8, 25.0), to_geo(766.8, 515.0)]
    town_road_v2_points = [to_geo(985.6, 25.0), to_geo(985.6, 515.0)]
    town_road_h1_points = [to_geo(515.0, 177.0), to_geo(1204.45, 177.0)]
    town_road_h2_points = [to_geo(515.0, 330.0), to_geo(1204.45, 330.0)]
    
    features = [
        {
            "name": "Southeast Flat Square",
            "points": sq_points,
            "tags": {"landuse": "farmyard"},
            "color": "#EC4899", # Pink
            "closed": True
        },
        {
            "name": "South Mountain",
            "points": mountain_points,
            "tags": {"landuse": "farmyard", "natural": "wood"},
            "color": "#22C55E", # Green
            "closed": True
        },
        {
            "name": "Northwest Reservoir",
            "points": reservoir_points,
            "tags": {"landuse": "farmyard"},
            "color": "#EC4899", # Pink
            "closed": True
        },
        {
            "name": "Water Channel",
            "points": channel_points,
            "tags": {"landuse": "farmyard", "natural": "wood"},
            "color": "#22C55E", # Green
            "closed": True
        },
        {
            "name": "Southern Pond",
            "points": pond_points,
            "tags": {"landuse": "farmyard", "natural": "wood"},
            "color": "#22C55E", # Green
            "closed": True
        },
        {
            "name": "North Primary Highway",
            "points": highway_north_points,
            "tags": {"highway": "primary"},
            "color": "#4B5563", # Road gray
            "closed": False
        },
        {
            "name": "South-Mid Primary Highway",
            "points": highway_south_mid_points,
            "tags": {"highway": "primary"},
            "color": "#4B5563", # Road gray
            "closed": False
        },
        {
            "name": "Diagonal Sinuous Primary Highway",
            "points": highway_diagonal_sinuous_points,
            "tags": {"highway": "primary"},
            "color": "#4B5563", # Road gray
            "closed": False
        },
        {
            "name": "East-Reservoir Vertical Highway",
            "points": highway_east_reservoir_points,
            "tags": {"highway": "primary"},
            "color": "#4B5563", # Road gray
            "closed": False
        },
        {
            "name": "Town Area",
            "points": town_area_points,
            "tags": {"landuse": "farmyard"},
            "color": "#6E5078", # C_YARD purple color
            "closed": True
        },
        {
            "name": "Town Secondary Road V1",
            "points": town_road_v1_points,
            "tags": {"highway": "secondary"},
            "color": "#AF5F28",
            "closed": False
        },
        {
            "name": "Town Secondary Road V2",
            "points": town_road_v2_points,
            "tags": {"highway": "secondary"},
            "color": "#AF5F28",
            "closed": False
        },
        {
            "name": "Town Secondary Road H1",
            "points": town_road_h1_points,
            "tags": {"highway": "secondary"},
            "color": "#AF5F28",
            "closed": False
        },
        {
            "name": "Town Secondary Road H2",
            "points": town_road_h2_points,
            "tags": {"highway": "secondary"},
            "color": "#AF5F28",
            "closed": False
        }
    ]
    
    # Dynamic field packer: Optimize space utilizing max 10m separation
    # 2.5ha and 5ha fields MUST be square:
    # 2.5ha square -> width = height = 158.11m
    # 5ha square -> width = height = 223.61m
    # 10ha and 20ha fields are rectangular.
    fields_data = []
    field_count = 1
    
    # 1. Above North Highway
    # Town occupies x in [548.0, 1204.45]. Fields area starts at x = 1214.45 to x_max_n = 4071.0.
    columns_north = [
        ("10ha_rect", 218.82),
        ("20ha_rect", 437.64),
        ("5ha_squares", 223.61),
        ("2.5ha_squares", 158.11),
        ("20ha_rect", 437.64),
        ("10ha_rect", 218.82),
        ("5ha_squares", 223.61),
        ("2.5ha_squares", 158.11),
        ("20ha_rect", 437.64),
        ("10ha_rect", 218.82)
    ]
    x_curr = 1214.45
    for col_type, w in columns_north:
        if x_curr + w > 4071.0:
            break
        x0, x1 = x_curr, x_curr + w
        if col_type == "10ha_rect":
            fields_data.append((f"Field N{field_count} (10.0ha)", x0, 25.0, x1, 482.0))
            field_count += 1
        elif col_type == "20ha_rect":
            fields_data.append((f"Field N{field_count} (20.0ha)", x0, 25.0, x1, 482.0))
            field_count += 1
        elif col_type == "5ha_squares":
            fields_data.append((f"Field N{field_count} (5.0ha)", x0, 25.0, x1, 248.61))
            field_count += 1
            fields_data.append((f"Field N{field_count} (5.0ha)", x0, 258.61, x1, 482.22))
            field_count += 1
        elif col_type == "2.5ha_squares":
            fields_data.append((f"Field N{field_count} (2.5ha)", x0, 25.0, x1, 183.11))
            field_count += 1
            fields_data.append((f"Field N{field_count} (2.5ha)", x0, 193.11, x1, 351.22))
            field_count += 1
            fields_data.append((f"Field N{field_count} (2.06ha)", x0, 361.22, x1, 482.0))
            field_count += 1
        x_curr = x1 + 10.0 # 10m separation

    # 2. Below North Highway, West of Sinuous Road (x in [40.0, 1059.0])
    columns_south_west = [
        ("5ha_square", 223.61),
        ("2.5ha_squares", 158.11),
        ("20ha_rect", 442.48),
        ("2.5ha_squares", 158.11)
    ]
    x_curr = 40.0
    for col_type, w in columns_south_west:
        if x_curr + w > 1059.0:
            break
        x0, x1 = x_curr, x_curr + w
        if col_type == "5ha_square":
            fields_data.append((f"Field N{field_count} (5.0ha)", x0, 548.0, x1, 771.61))
            field_count += 1
        elif col_type == "2.5ha_squares":
            fields_data.append((f"Field N{field_count} (2.5ha)", x0, 548.0, x1, 706.11))
            field_count += 1
            fields_data.append((f"Field N{field_count} (2.5ha)", x0, 716.11, x1, 874.22))
            field_count += 1
        elif col_type == "20ha_rect":
            fields_data.append((f"Field N{field_count} (20.0ha)", x0, 548.0, x1, 1000.0))
            field_count += 1
        x_curr = x1 + 10.0

    # 3. Below North Highway, East of Sinuous Road (x in [1185.0, 4071.0])
    columns_south_east = [
        ("20ha_rect", 442.48),
        ("10ha_rect", 221.24),
        ("5ha_squares", 223.61),
        ("2.5ha_squares", 158.11),
        ("20ha_rect", 442.48),
        ("10ha_rect", 221.24),
        ("5ha_squares", 223.61),
        ("2.5ha_squares", 158.11),
        ("20ha_rect", 442.48),
        ("10ha_rect", 221.24)
    ]
    x_curr = 1185.0
    for col_type, w in columns_south_east:
        if x_curr + w > 4071.0:
            break
        x0, x1 = x_curr, x_curr + w
        if col_type == "10ha_rect":
            fields_data.append((f"Field N{field_count} (10.0ha)", x0, 548.0, x1, 1000.0))
            field_count += 1
        elif col_type == "20ha_rect":
            fields_data.append((f"Field N{field_count} (20.0ha)", x0, 548.0, x1, 1000.0))
            field_count += 1
        elif col_type == "5ha_squares":
            fields_data.append((f"Field N{field_count} (5.0ha)", x0, 548.0, x1, 771.61))
            field_count += 1
            fields_data.append((f"Field N{field_count} (5.0ha)", x0, 781.61, x1, 1005.22))
            field_count += 1
        elif col_type == "2.5ha_squares":
            fields_data.append((f"Field N{field_count} (2.5ha)", x0, 548.0, x1, 706.11))
            field_count += 1
            fields_data.append((f"Field N{field_count} (2.5ha)", x0, 716.11, x1, 874.22))
            field_count += 1
            fields_data.append((f"Field N{field_count} (2.01ha)", x0, 884.22, x1, 1000.0))
            field_count += 1
        x_curr = x1 + 10.0

    # Populate features list
    for name, x0, y0, x1, y1 in fields_data:
        pts = [
            to_geo(x0, y0),
            to_geo(x1, y0),
            to_geo(x1, y1),
            to_geo(x0, y1),
            to_geo(x0, y0)
        ]
        features.append({
            "name": name,
            "points": pts,
            "tags": {"landuse": "farmland"},
            "color": "#96A858",
            "closed": True,
            "is_field": True,
            "local_rect": (x0, y0, x1, y1)
        })
    
    # 3. Generate OSM XML File
    print(f"Creating OSM XML at: {output_osm_path}...")
    root = ET.Element("osm", version="0.6", generator="osm_generator")
    ET.SubElement(
        root, 
        "bounds", 
        minlat=f"{min_lat:.8f}", 
        minlon=f"{min_lon:.8f}", 
        maxlat=f"{max_lat:.8f}", 
        maxlon=f"{max_lon:.8f}"
    )
    
    node_id = 1
    way_id = 1
    
    for feat in features:
        pts = feat["points"]
        node_refs = []
        is_closed = feat.get("closed", True)
        
        if is_closed:
            # Create OSM node elements for vertices of a polygon (except last which is same as first)
            for i, (lat, lon) in enumerate(pts[:-1]):
                ET.SubElement(
                    root, 
                    "node", 
                    id=str(node_id), 
                    lat=f"{lat:.8f}", 
                    lon=f"{lon:.8f}", 
                    version="1",
                    visible="true"
                )
                node_refs.append(node_id)
                node_id += 1
            # Append the first node reference at the end to close the polygon
            node_refs.append(node_refs[0])
        else:
            # Create OSM node elements for all vertices of a line (open way)
            for i, (lat, lon) in enumerate(pts):
                ET.SubElement(
                    root, 
                    "node", 
                    id=str(node_id), 
                    lat=f"{lat:.8f}", 
                    lon=f"{lon:.8f}", 
                    version="1",
                    visible="true"
                )
                node_refs.append(node_id)
                node_id += 1
            
        # Create OSM way element
        way = ET.SubElement(root, "way", id=str(way_id), version="1", visible="true")
        way_id += 1
        
        for ref in node_refs:
            ET.SubElement(way, "nd", ref=str(ref))
            
        for k, v in feat["tags"].items():
            ET.SubElement(way, "tag", k=k, v=v)
            
    # Format and save OSM file
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(output_osm_path, encoding="utf-8", xml_declaration=True)
    print(f"  OSM XML file saved successfully with {node_id - 1} nodes and {way_id - 1} ways.")
    
    # 4. Generate visual representation (PNG) using Pillow to match the reference styling exactly
    print(f"Generating visual preview PNG at: {output_png_path}...")
    from PIL import Image, ImageDraw
    
    # Configurations matching common.py and genmap.py
    S_canvas = 4096
    C_FARM   = (150, 168, 88)  # farm green background
    C_FARMB  = (0, 0, 0)        # black border
    C_ROADP  = (240, 200, 30)   # yellow main road
    C_ROADS  = (175, 95, 40)    # orange-brown secondary road
    C_FOREST = (38, 74, 44)     # forest green
    C_YARD   = (110, 80, 120)   # yard purple
    C_YARDB  = (60, 42, 72)     # yard border
    
    TH_P = 22                   # primary road thickness
    TH_S = 16                   # secondary road thickness
    W_ROAD_BORDER = 12          # black road margin
    BORDER = 25                 # black perimeter border width
    
    # Initialize the Pillow image canvas
    img = Image.new("RGB", (S_canvas, S_canvas), C_FARM)
    d = ImageDraw.Draw(img)
    
    # 4.0 Draw agricultural fields (parcels)
    for feat in features:
        if feat.get("is_field", False):
            x0, y0, x1, y1 = feat["local_rect"]
            d.rectangle([x0, y0, x1, y1], fill=C_FARM, outline=C_FARMB, width=12)
            
    # 4.1 Draw yards (polygons)
    # Southeast Flat Square
    sq_pts = [(3373.9, 2938.9), (4081.0, 2938.9), (4081.0, 3646.0), (3373.9, 3646.0)]
    d.polygon(sq_pts, fill=C_YARD)
    d.line(sq_pts + [sq_pts[0]], fill=C_YARDB, width=5)
    
    # Northwest Reservoir
    res_pts = [(15.0, 15.0), (515.0, 15.0), (515.0, 515.0), (15.0, 515.0)]
    d.polygon(res_pts, fill=C_YARD)
    d.line(res_pts + [res_pts[0]], fill=C_YARDB, width=5)
    
    # Town Area (30 Hectares)
    town_pts = [(548.0, 25.0), (1204.45, 25.0), (1204.45, 482.0), (548.0, 482.0)]
    d.polygon(town_pts, fill=C_YARD)
    d.line(town_pts + [town_pts[0]], fill=C_YARDB, width=5)
    
    # 4.2 Draw forests (polygons)
    # South Mountain
    mountain_pts = [(0.0, 3696.0), (4096.0, 3696.0), (4096.0, 4096.0), (0.0, 4096.0)]
    d.polygon(mountain_pts, fill=C_FOREST)
    d.line(mountain_pts + [mountain_pts[0]], fill=C_FARMB, width=24)
    
    # Water Channel
    channel_pts = [(15.0, 15.0), (30.0, 15.0), (30.0, 3696.0), (15.0, 3696.0)]
    d.polygon(channel_pts, fill=C_FOREST)
    d.line(channel_pts + [channel_pts[0]], fill=C_FARMB, width=12)
    
    # Southern Pond
    pond_pts = [(15.0, 3646.0), (65.0, 3646.0), (65.0, 3696.0), (15.0, 3696.0)]
    d.polygon(pond_pts, fill=C_FOREST)
    d.line(pond_pts + [pond_pts[0]], fill=C_FARMB, width=12)
    
    # 4.3 Draw Road Outlines (black margins)
    # North Primary Highway
    d.line([(4096.0, 515.0), (0.0, 515.0)], fill=C_FARMB, width=TH_P + 2 * W_ROAD_BORDER, joint="round")
    # South-Mid Primary Highway
    d.line([(4096.0, 2938.9), (0.0, 2938.9)], fill=C_FARMB, width=TH_P + 2 * W_ROAD_BORDER, joint="round")
    # Diagonal Sinuous Primary Highway
    road_diagonal_pts = []
    for i in range(num_steps + 1):
        y_val = y_start_sinuous + (i / num_steps) * (y_end_sinuous - y_start_sinuous)
        x_diag = x_start + (y_val / size_m) * (x_end - x_start)
        x_val = x_diag + amp * math.sin(2.0 * math.pi * y_val / wavelength)
        road_diagonal_pts.append((x_val, y_val))
    d.line(road_diagonal_pts, fill=C_FARMB, width=TH_P + 2 * W_ROAD_BORDER, joint="round")
    # East-Reservoir Vertical Highway
    d.line([(515.0, 0.0), (515.0, 515.0)], fill=C_FARMB, width=TH_P + 2 * W_ROAD_BORDER, joint="round")
    # Town Secondary Roads
    d.line([(766.8, 25.0), (766.8, 515.0)], fill=C_FARMB, width=TH_S + 2 * W_ROAD_BORDER, joint="round")
    d.line([(985.6, 25.0), (985.6, 515.0)], fill=C_FARMB, width=TH_S + 2 * W_ROAD_BORDER, joint="round")
    d.line([(515.0, 177.0), (1204.45, 177.0)], fill=C_FARMB, width=TH_S + 2 * W_ROAD_BORDER, joint="round")
    d.line([(515.0, 330.0), (1204.45, 330.0)], fill=C_FARMB, width=TH_S + 2 * W_ROAD_BORDER, joint="round")
    
    # 4.4 Draw Road Fills (yellow/orange)
    # North Primary Highway
    d.line([(4096.0, 515.0), (0.0, 515.0)], fill=C_ROADP, width=TH_P, joint="round")
    # South-Mid Primary Highway
    d.line([(4096.0, 2938.9), (0.0, 2938.9)], fill=C_ROADP, width=TH_P, joint="round")
    # Diagonal Sinuous Primary Highway
    d.line(road_diagonal_pts, fill=C_ROADP, width=TH_P, joint="round")
    # East-Reservoir Vertical Highway
    d.line([(515.0, 0.0), (515.0, 515.0)], fill=C_ROADP, width=TH_P, joint="round")
    # Town Secondary Roads
    d.line([(766.8, 25.0), (766.8, 515.0)], fill=C_ROADS, width=TH_S, joint="round")
    d.line([(985.6, 25.0), (985.6, 515.0)], fill=C_ROADS, width=TH_S, joint="round")
    d.line([(515.0, 177.0), (1204.45, 177.0)], fill=C_ROADS, width=TH_S, joint="round")
    d.line([(515.0, 330.0), (1204.45, 330.0)], fill=C_ROADS, width=TH_S, joint="round")
    
    # 4.5 Paint 25px black border around the perimeter
    d.rectangle([0, 0, S_canvas, BORDER], fill=C_FARMB)
    d.rectangle([0, S_canvas - BORDER, S_canvas, S_canvas], fill=C_FARMB)
    d.rectangle([0, 0, BORDER, S_canvas], fill=C_FARMB)
    d.rectangle([S_canvas - BORDER, 0, S_canvas, S_canvas], fill=C_FARMB)
    
    # Save the output image
    img.save(output_png_path)
    print(f"  Preview image saved successfully to: {output_png_path}")
    print("\n=== Generation Complete ===")

if __name__ == "__main__":
    main()
