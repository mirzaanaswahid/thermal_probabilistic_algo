import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.affinity import rotate
from matplotlib.patches import Patch
import pandas as pd
import unicodedata
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
geojson_path = "/home/px4_sitl/Downloads/limites-administratives-agglomeration-nad83.geojson"
utm_crs = "EPSG:32618"
output_dir = "/home/px4_sitl/ets_work/montreal_research/montreal_region_uav_maps_with_paths"
swath_width = 100  # meters between path strips
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# STRING NORMALIZATION
# ----------------------------
def normalize(s):
    s = s.replace("’", "'").replace("‘", "'")
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode()
    return s.lower().strip()

# ----------------------------
# DENSITY + POPULATION DATA
# ----------------------------
region_data = [  # (Name, Area, Population)
    ("Ahuntsic-Cartierville", 25.58, 138923),
    ("Anjou", 13.89, 45288),
    ("Baie-D’Urfé", 8.03, 3823),
    ("Beaconsfield", 24.68, 19908),
    ("Côte-des-Neiges-Notre-Dame-de-Grâce", 21.49, 173729),
    ("Côte-Saint-Luc", 6.81, 34425),
    ("Dollard-des-Ormeaux", 15.09, 49713),
    ("Dorval", 29.13, 18970),
    ("Hampstead", 1.77, 7153),
    ("Kirkland", 9.62, 21255),
    ("Lachine", 22.57, 46971),
    ("LaSalle", 25.21, 82933),
    ("Le Plateau-Mont-Royal", 8.14, 110329),
    ("Le Sud-Ouest", 18.10, 86347),
    ("L’Île-Bizard-Sainte-Geneviève", 36.32, 26099),
    ("L’Île-Dorval", 0.18, 134),
    ("Mercier-Hochelaga-Maisonneuve", 27.41, 142753),
    ("Mont-Royal", 7.46, 21202),
    ("Montréal-Est", 13.99, 3850),
    ("Montréal-Nord", 12.46, 86857),
    ("Montréal-Ouest", 1.42, 5254),
    ("Outremont", 3.80, 26505),
    ("Pierrefonds-Roxboro", 33.98, 73194),
    ("Pointe-Claire", 34.30, 33488),
    ("Rivière-des-Prairies-Pointe-aux-Trembles", 51.27, 113868),
    ("Rosemont-La Petite-Patrie", 15.88, 146501),
    ("Saint-Laurent", 43.06, 104366),
    ("Saint-Léonard", 13.51, 80983),
    ("Sainte-Anne-de-Bellevue", 11.18, 5158),
    ("Senneville", 18.59, 923),
    ("Verdun", 22.29, 72820),
    ("Ville-Marie", 21.50, 103017),
    ("Villeray-Saint-Michel-Parc-Extension", 16.48, 144814),
    ("Westmount", 4.02, 20832),
]
df = pd.DataFrame(region_data, columns=["Name", "Area_km2", "Population"])
df["Density"] = df["Population"] / df["Area_km2"]

def classify_density(d):
    if d > 7000:
        return 'High'
    elif d > 3000:
        return 'Medium'
    elif d > 1000:
        return 'Low'
    else:
        return 'Very Low'

df["DensityCategory"] = df["Density"].apply(classify_density)

density_to_uavs = {"High": 4, "Medium": 3, "Low": 2, "Very Low": 1}
density_colors = {"High": "red", "Medium": "orange", "Low": "lightgreen", "Very Low": "grey"}
uav_colors = ["blue", "green", "purple", "cyan"]

# ----------------------------
# LOAD GEOJSON AND NORMALIZE
# ----------------------------
gdf = gpd.read_file(geojson_path).to_crs(utm_crs)
gdf["NOM_normalized"] = gdf["NOM"].apply(normalize)

# ----------------------------
# MAIN LOOP: FOR EACH REGION
# ----------------------------
for _, row in df.iterrows():
    name = row["Name"]
    norm_name = normalize(name)
    density = row["DensityCategory"]
    area = row["Area_km2"]
    num_uavs = density_to_uavs[density]
    fill_color = density_colors[density]

    region_geom = gdf[gdf["NOM_normalized"] == norm_name]
    if region_geom.empty:
        print(f"Region not found: {name}")
        continue
    geom = region_geom.geometry.values[0]

    # Generate lawnmower lines
    bounds = geom.bounds
    minx, miny, maxx, maxy = bounds
    y = miny
    strips = []
    while y <= maxy:
        line = LineString([(minx, y), (maxx, y)])
        clipped = line.intersection(geom)
        if not clipped.is_empty:
            if isinstance(clipped, LineString):
                strips.append(clipped)
            elif hasattr(clipped, 'geoms'):
                strips.extend([s for s in clipped.geoms if isinstance(s, LineString)])
        y += swath_width

    # Split strips among UAVs (round-robin)
    uav_paths = [[] for _ in range(num_uavs)]
    for i, strip in enumerate(strips):
        uav_paths[i % num_uavs].append(strip)

    # ----------------------------
    # PLOT
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    region_geom.plot(ax=ax, color=fill_color, edgecolor='black')

    for i, path in enumerate(uav_paths):
        for segment in path:
            x, y = segment.xy
            ax.plot(x, y, color=uav_colors[i % len(uav_colors)], linewidth=1, label=f"UAV {i+1}" if i == 0 else "")

    # Ticks, bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([minx, (minx + maxx) / 2, maxx])
    ax.set_yticks([miny, (miny + maxy) / 2, maxy])
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')
    ax.grid(True)

    # Title and Legend
    ax.set_title(f"{name}", fontsize=14)
    legend_patches = [
        Patch(facecolor=fill_color, edgecolor='black', label=f"Density: {density}"),
        Patch(facecolor='white', edgecolor='black', label=f"Area: {area:.2f} km²"),
        Patch(facecolor='white', edgecolor='black', label=f"UAVs Required: {num_uavs}")
    ]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=True)

    # Save figure
    safe_name = name.replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join(output_dir, f"{safe_name}.png"), bbox_inches='tight')
    plt.close(fig)

print(f"✅ All regional UAV path maps saved to: {output_dir}")

