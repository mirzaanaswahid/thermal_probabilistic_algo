import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Polygon, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.collections import PatchCollection
import fiona
import pandas as pd
import os

# Path to your local GeoJSON file
geojson_path = "/home/px4_sitl/Downloads/limites-administratives-agglomeration-nad83.geojson"

# Step 1: Load features from the GeoJSON
with fiona.open(geojson_path) as src:
    features = list(src)

# Step 2: Add population and density data
# Population and area data for Montreal's boroughs
data = [
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
df = pd.DataFrame(data, columns=["Name", "Area", "Population"])
df["Density"] = df["Population"] / df["Area"]

# Step 3: Define density categories and colors
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
density_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'lightgreen', 'Very Low': 'grey'}
region_to_density = {row["Name"]: row["DensityCategory"] for _, row in df.iterrows()}
region_to_color = {name: density_colors[cat] for name, cat in region_to_density.items()}

# Step 4: Create output folder
output_dir = "/home/px4_sitl/ets_work/montreal_research/montreal_region_maps"
os.makedirs(output_dir, exist_ok=True)

# Step 5: Generate and save plots
for feature in features:
    name = feature['properties'].get('NOM', '').strip()
    geom = shape(feature['geometry'])
    density_category = region_to_density.get(name, 'Very Low')
    color = region_to_color.get(name, 'grey')

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw geometry
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        poly = MplPolygon(coords, closed=True, edgecolor='black', facecolor=color, alpha=0.6)
        ax.add_patch(poly)
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            coords = list(part.exterior.coords)
            poly = MplPolygon(coords, closed=True, edgecolor='black', facecolor=color, alpha=0.6)
            ax.add_patch(poly)

    # Set axis bounds and ticks
    minx, miny, maxx, maxy = geom.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([minx, (minx + maxx) / 2, maxx])
    ax.set_yticks([miny, (miny + maxy) / 2, maxy])
    ax.tick_params(labelsize=8)

    # Title and legend
    ax.set_title(f"{name} (Density: {density_category})", fontsize=14)
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='High Density'),
        Patch(facecolor='orange', edgecolor='black', label='Medium Density'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Low Density'),
        Patch(facecolor='grey', edgecolor='black', label='Very Low Density')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    ax.set_aspect('equal')

    # Save the figure
    safe_name = name.replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join(output_dir, f"{safe_name}.png"), bbox_inches='tight')
    plt.close(fig)

print(f"✅ All region maps saved to: {output_dir}")

