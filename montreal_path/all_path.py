import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, MultiPolygon, Polygon
from matplotlib.patches import Patch
import pandas as pd
import os

# ----------------------------
# CONFIGURATION
# ----------------------------
geojson_path = "/home/px4_sitl/Downloads/limites-administratives-agglomeration-nad83.geojson"
utm_crs = "EPSG:32618"  # Montreal UTM Zone
output_dir = "/home/px4_sitl/ets_work/montreal_research/montreal_region_uav_mapsv2.0"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Population Data (from table)
# ----------------------------
region_data = [
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

# Classification logic
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

# UAV allocation
density_to_uavs = {
    "High": 4,
    "Medium": 3,
    "Low": 2,
    "Very Low": 1
}

# Color map
density_colors = {
    "High": "red",
    "Medium": "orange",
    "Low": "lightgreen",
    "Very Low": "grey"
}

# ----------------------------
# LOAD GEOJSON & PLOT EACH REGION
# ----------------------------
gdf = gpd.read_file(geojson_path)
gdf = gdf.to_crs(utm_crs)

for _, row in df.iterrows():
    region_name = row["Name"]
    density_cat = row["DensityCategory"]
    color = density_colors[density_cat]
    uavs = density_to_uavs[density_cat]
    area = row["Area_km2"]

    # Match geometry
    region_gdf = gdf[gdf["NOM"] == region_name]
    if region_gdf.empty:
        print(f"Region not found: {region_name}")
        continue

    geom = region_gdf.geometry.values[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    region_gdf.plot(ax=ax, color=color, edgecolor='black', linewidth=1)

    # Add legend
    legend_elements = [
        Patch(facecolor=color, edgecolor='black', label=f"Density: {density_cat}"),
        Patch(facecolor='white', edgecolor='black', label=f"Area: {area:.2f} km²"),
        Patch(facecolor='white', edgecolor='black', label=f"UAVs Required: {uavs}")
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=True)

    # Axis and titles
    ax.set_title(f"{region_name}", fontsize=14)
    minx, miny, maxx, maxy = geom.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([minx, (minx+maxx)/2, maxx])
    ax.set_yticks([miny, (miny+maxy)/2, maxy])
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')
    ax.grid(True)

    # Save the figure
    fname = region_name.replace(" ", "_").replace("/", "_") + ".png"
    fig.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
    plt.close(fig)

print(f"✅ Saved regional maps to {output_dir}")

