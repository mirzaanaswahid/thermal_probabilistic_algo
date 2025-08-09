import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.affinity import rotate
from pyproj import Transformer
import numpy as np

# ----------------------------
# CONFIGURATION
# ----------------------------
region_name = "Ahuntsic-Cartierville"
geojson_path = "/home/px4_sitl/Downloads/limites-administratives-agglomeration-nad83.geojson"
swath_width = 100  # meters
rotation_angle = 0  # rotation (if region is tilted)
utm_epsg = "EPSG:32618"  # UTM Zone 18N (covers Montreal)

# ----------------------------
# LOAD REGION AND CONVERT TO UTM
# ----------------------------
gdf = gpd.read_file(geojson_path)
region_gdf = gdf[gdf["NOM"] == region_name]

if region_gdf.empty:
    raise ValueError(f"Region '{region_name}' not found in the GeoJSON file.")

geom = region_gdf.geometry.values[0]
geom_utm = region_gdf.to_crs(utm_epsg).geometry.values[0]

# ----------------------------
# GENERATE LAWN MOWER PATH
# ----------------------------
bounds = geom_utm.bounds
minx, miny, maxx, maxy = bounds
lines = []
y = miny

while y <= maxy:
    line = LineString([(minx, y), (maxx, y)])
    clipped = line.intersection(geom_utm)
    if not clipped.is_empty:
        if isinstance(clipped, LineString):
            lines.append(clipped)
        elif hasattr(clipped, 'geoms'):
            lines.extend([seg for seg in clipped.geoms if isinstance(seg, LineString)])
    y += swath_width

if rotation_angle != 0:
    lines = [rotate(line, rotation_angle, origin='center') for line in lines]

# ----------------------------
# DIVIDE PATH BETWEEN TWO UAVs
# ----------------------------
uav1_lines = lines[::2]
uav2_lines = lines[1::2]

def extract_waypoints(line_list):
    waypoints = []
    for i, line in enumerate(line_list):
        pts = list(line.coords)
        if i % 2 == 1:
            pts = pts[::-1]  # alternate direction
        waypoints.extend(pts)
    return waypoints

uav1_path = extract_waypoints(uav1_lines)
uav2_path = extract_waypoints(uav2_lines)

takeoff_uav1 = uav1_path[0]
landing_uav1 = uav1_path[-1]
takeoff_uav2 = uav2_path[0]
landing_uav2 = uav2_path[-1]

# ----------------------------
# PLOT REGION AND PATHS
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 10))
region_gdf.to_crs(utm_epsg).plot(ax=ax, facecolor='lightgrey', edgecolor='black')

# Plot UAV 1 path
for line in uav1_lines:
    x, y = line.xy
    ax.plot(x, y, color='blue', label="UAV 1" if 'UAV 1' not in ax.get_legend_handles_labels()[1] else "")

# Plot UAV 2 path
for line in uav2_lines:
    x, y = line.xy
    ax.plot(x, y, color='green', label="UAV 2" if 'UAV 2' not in ax.get_legend_handles_labels()[1] else "")

# Plot takeoff and landing
ax.plot(*takeoff_uav1, 'bo', label="Takeoff UAV 1")
ax.plot(*landing_uav1, 'bx', label="Landing UAV 1")
ax.plot(*takeoff_uav2, 'go', label="Takeoff UAV 2")
ax.plot(*landing_uav2, 'gx', label="Landing UAV 2")

# Plot style
ax.set_title(f"Two-UAV Coverage Path for {region_name}", fontsize=14)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
ax.set_aspect('equal')
ax.grid(True)
plt.xlabel("Easting (meters)")
plt.ylabel("Northing (meters)")
plt.tight_layout()
plt.show()

