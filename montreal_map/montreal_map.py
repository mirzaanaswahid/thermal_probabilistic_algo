#!/usr/bin/env python3
# --------------------------------------------------------------
#  Montréal – Complete Borough Map with Density Classes & UAVs
# --------------------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import unicodedata

# ---------- 1.  FILE PATHS & CONSTANTS ------------------------
geojson_path = "/home/px4_sitl/Downloads/limites-administratives-agglomeration-nad83.geojson"
utm_crs       = "EPSG:32618"      # same CRS you used before

# ---------- 2.  POPULATION, AREA, DENSITY TABLE ---------------
region_data = [  # (Name, Area_km2, Population)
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
df_attr = pd.DataFrame(region_data, columns=["Name", "Area_km2", "Population"])
df_attr["Density"] = df_attr["Population"] / df_attr["Area_km2"]

def classify_density(d):
    if d > 7000:
        return "High"
    elif d > 3000:
        return "Medium"
    elif d > 1000:
        return "Low"
    else:
        return "Very Low"

df_attr["DensityCat"] = df_attr["Density"].apply(classify_density)

density_to_uavs = {"High": 4, "Medium": 3, "Low": 2, "Very Low": 1}
density_colors  = {"High": "red", "Medium": "orange", "Low": "lightgreen", "Very Low": "grey"}

# ---------- 3.  LOAD GEOJSON & MERGE ATTRIBUTES ---------------
def normalize(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    return unicodedata.normalize("NFKD", s).encode("ASCII","ignore").decode().lower().strip()

gdf = gpd.read_file(geojson_path).to_crs(utm_crs)
gdf["key"] = gdf["NOM"].apply(normalize)
df_attr["key"] = df_attr["Name"].apply(normalize)

gdf = gdf.merge(df_attr, on="key", how="left")  # adds Area_km2, Population, DensityCat …

# drop boroughs we didn't have numbers for
gdf = gdf.dropna(subset=["DensityCat"])

# ---------- 4.  PLOT THE COMPLETE MAP -------------------------
fig, ax = plt.subplots(figsize=(12, 12))

for cat, sub in gdf.groupby("DensityCat"):
    sub.plot(ax=ax, color=density_colors[cat], edgecolor="black", linewidth=0.3, label=cat)

# annotate borough names (small font)
for _, row in gdf.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    ax.text(cx, cy, row["NOM"], fontsize=5, ha="center", va="center")

# Legend: density + UAV allocation
legend_patches = [
    Patch(facecolor=density_colors["High"],   edgecolor='black', label="High density  → 4 UAVs"),
    Patch(facecolor=density_colors["Medium"], edgecolor='black', label="Medium density → 3 UAVs"),
    Patch(facecolor=density_colors["Low"],    edgecolor='black', label="Low density  → 2 UAVs"),
    Patch(facecolor=density_colors["Very Low"], edgecolor='black', label="Very Low density → 1 UAV"),
]
ax.legend(handles=legend_patches, title="Population‑density class", loc="lower left", frameon=True)

ax.set_title("Montréal – Density Classes & UAV Allocation", fontsize=16)
ax.set_axis_off()
plt.tight_layout()
plt.show()

