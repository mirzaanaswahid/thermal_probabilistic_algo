import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.affinity import rotate
from pyproj import Transformer
import numpy as np

# -----------------------
# CONFIGURATION
# -----------------------
region_name = "Ahuntsic-Cartierville"
geojson_path = "/home/px4_sitl/Downloads/limites-administratives-agglomeration-nad83.geojson"
swath_width = 100  # meters between passes
rotation_angle = 0  # degrees (can rotate to better align path to region)
altitude = 120  # meters (for future extension)

# -----------------------
# LOAD REGION
# -----------------------
gdf = gpd.read_file(geojson_path)
gdf = gdf[gdf["NOM"] == region_name]
geom = gdf.geometry.values[0]

# -----------------------
# TRANSFORM TO METRIC (UTM)
# -----------------------
centroid = geom.centroid
lat, lon = centroid.y, centroid.x
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)  # Montreal -> UTM Zone 18N
geom_utm = gdf.to_crs("EPSG:32618").geometry.values[0]

# -----------------------
# GENERATE LAWN MOWER PATH
# -----------------------
bounds = geom_utm.bounds
minx, miny, maxx, maxy = bounds
lines = []
y = miny

while y <= maxy:
    line = LineString([(minx, y), (maxx, y)])
    line = line.intersection(geom_utm)
    if not line.is_empty:
        if isinstance(line, LineString):
            lines.append(line)
        elif hasattr(line, 'geoms'):
            lines.extend([seg for seg in line.geoms if isinstance(seg, LineString)])
    y += swath_width

# Optional: rotate the lines if needed
if rotation_angle != 0:
    lines = [rotate(line, rotation_angle, origin='center') for line in lines]

# -----------------------
# PLOT REGION AND PATH
# -----------------------
fig, ax = plt.subplots(figsize=(10, 10))
gpd.GeoSeries(geom_utm).plot(ax=ax, facecolor='lightgrey', edgecolor='black', linewidth=1)
for line in lines:
    x, y = line.xy
    ax.plot(x, y, color='blue', linewidth=1)

ax.set_title(f"Lawnmower Path for {region_name} at {swath_width}m swath width", fontsize=14)
ax.set_aspect('equal')
plt.xlabel("Easting (meters)")
plt.ylabel("Northing (meters)")
plt.grid(True)
plt.tight_layout()
plt.show()

