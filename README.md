# FS25_Granja_Bonita_x4

Generador del mapa "Granja Bonita" para Farming Simulator 25 en versión x4: 4096 m
jugables, DEM de 8192 px (2048 m de borde a cada lado). Produce el heightmap
(`dem_generator/dem_8k.png`) y el vector (`osm_generator/map.osm`) que luego se importan a
mano en Giants Editor.

    python3 map_layout.py                          # comprobación del layout
    python3 dem_generator/generate_dem.py          # DEM + terrain_stats.json + previews
    python3 dem_generator/measure_elevation.py     # aceptación del DEM
    python3 osm_generator/generate_osm.py          # map.osm
    python3 osm_generator/check_osm.py             # aceptación del OSM
    python3 osm_generator/visualize_osm.py         # map_osm_visual.png
    python3 visualizer/create_3d_viewer.py         # dem_viewer_3d.html

La geometría sale de `input/custom_osm.osm` (survey JOSM de 4096 m) y el relieve de
`input/valle_bonito.png`. Los dos se obtuvieron una vez del mapa 16x con
`tools/convert_from_x16.py` (deformación del survey con pueblos y granjas a tamaño real,
reparcelación de campos, aclarado de caminos, DEM remuestreado con las reparaciones del
fuente ya aplicadas); el survey se retoca en JOSM a partir de ahí. `CLAUDE.md` describe
la arquitectura y lo que hace la conversión.
