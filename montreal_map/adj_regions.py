#!/usr/bin/env python3
# --------------------------------------------------------------
#  Montréal – 4 Adjacent Boroughs (density ≥ Medium) with ≥1 High
# --------------------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import unicodedata
from itertools import combinations

# ---------- 1) FILE PATHS & CRS --------------------------------
geojson_path = "/home/px4_sitl/Downloads/limites-administratives-agglomeration-nad83.geojson"
utm_crs      = "EPSG:32618"   # projected CRS (meters) for robust length calcs

# ---------- 2) POPULATION, AREA, DENSITY -----------------------
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
        return "Very Low"

df_attr["DensityCat"] = df_attr["Density"].apply(classify_density)

density_to_uavs = {"High": 4, "Medium": 3, "Low": 2, "Very Low": 1}
density_colors  = {"High": "red", "Medium": "orange", "Low": "lightgreen", "Very Low": "grey"}

# ---------- 3) LOAD GEOJSON & MERGE ----------------------------
def normalize(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'")
    return unicodedata.normalize("NFKD", s).encode("ASCII","ignore").decode().lower().strip()

gdf = gpd.read_file(geojson_path).to_crs(utm_crs)
gdf["key"] = gdf["NOM"].apply(normalize)
df_attr["key"] = df_attr["Name"].apply(normalize)

gdf = gdf.merge(df_attr, on="key", how="left")
gdf = gdf.dropna(subset=["DensityCat"]).reset_index(drop=True)

# ---------- 4) ADJACENCY BY SHARED BORDER (> 0 length) ---------
# adj[i][j] = shared boundary length (meters) if polygons share an edge (no point-only touches)
adj = {i: {} for i in range(len(gdf))}
for i in range(len(gdf)):
    ai = gdf.geometry.iloc[i]
    for j in range(i+1, len(gdf)):
        bj = gdf.geometry.iloc[j]
        shared_line = ai.boundary.intersection(bj.boundary)
        shared_len = float(shared_line.length)
        if shared_len > 0:
            adj[i][j] = shared_len
            adj[j][i] = shared_len

# ---------- 5) FIND CONNECTED 4-SET WITH CONSTRAINTS -----------
ALLOWED       = {"High", "Medium"}  # Medium-or-higher filter
CLUSTER_SIZE  = 4
min_high      = 1                   # <-- require at least this many High boroughs
strict_pairwise = False             # True → require all 6 pairs to share a border (K4)
focus_name    = None                # e.g., "Ville-Marie" to force inclusion; or None

eligible_idx = [i for i, row in gdf.iterrows() if row["DensityCat"] in ALLOWED]

def is_connected_subset(nodes, adj):
    nodes = list(nodes)
    if not nodes:
        return False
    S = set(nodes)
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        u = stack.pop()
        for v in adj[u].keys():
            if v in S and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)

def total_internal_border(nodes, adj):
    return sum(adj[i][j] for i, j in combinations(nodes, 2) if j in adj[i])

def count_high(nodes, gdf):
    return sum(gdf.iloc[i]["DensityCat"] == "High" for i in nodes)

def find_cluster(gdf, adj, eligible_idx, cluster_size=4, focus_name=None,
                 strict_pairwise=False, min_high=1):
    # Map focus to index (if provided), ensure it is eligible
    focus_idx = None
    if focus_name is not None:
        k = normalize(focus_name)
        hits = gdf.index[gdf["key"] == k].tolist()
        if not hits:
            raise ValueError(f"Borough '{focus_name}' not found.")
        focus_idx = hits[0]
        if gdf.iloc[focus_idx]["DensityCat"] not in ALLOWED:
            raise RuntimeError(f"Borough '{focus_name}' is not Medium/High; cannot satisfy constraint.")
        if focus_idx not in eligible_idx:
            eligible_idx.append(focus_idx)

    pool = list(set(eligible_idx))
    if len(pool) < cluster_size:
        raise RuntimeError(f"Only {len(pool)} Medium/High borough(s); need at least {cluster_size}.")

    best_subset = None
    best_key    = (-1, -1.0)  # (num_high, total_shared_len)

    for sub in combinations(pool, cluster_size):
        if focus_idx is not None and focus_idx not in sub:
            continue
        # structural constraint
        if strict_pairwise:
            if any(j not in adj[i] for i, j in combinations(sub, 2)):
                continue
        else:
            if not is_connected_subset(sub, adj):
                continue
        # high-count constraint
        num_h = count_high(sub, gdf)
        if num_h < min_high:
            continue

        score = total_internal_border(sub, adj)
        # lexicographic: maximize num_high, then shared length
        key = (num_h, score)
        if key > best_key:
            best_key = key
            best_subset = sub

    if best_subset is None:
        mode = "pairwise-adjacent (K4)" if strict_pairwise else "connected"
        raise RuntimeError(f"No {mode} cluster of {cluster_size} boroughs with ≥{min_high} High found.")
    return list(best_subset), best_key

indices, (num_high, score) = find_cluster(
    gdf, adj, eligible_idx,
    cluster_size=CLUSTER_SIZE,
    focus_name=focus_name,
    strict_pairwise=strict_pairwise,
    min_high=min_high
)

subset = gdf.iloc[indices].copy()

# ---------- 6) PLOT --------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

for cat, sub in subset.groupby("DensityCat"):
    sub.plot(ax=ax, color=density_colors[cat], edgecolor="black", linewidth=0.6, label=cat)

# Emphasize subset boundaries
subset.boundary.plot(ax=ax, linewidth=1.4, color="black")

# Labels
for _, row in subset.iterrows():
    c = row.geometry.centroid
    ax.text(c.x, c.y, row["NOM"], fontsize=7, ha="center", va="center")

legend_patches = [
    Patch(facecolor=density_colors["High"],     edgecolor='black', label="High density  → 4 UAVs"),
    Patch(facecolor=density_colors["Medium"],   edgecolor='black', label="Medium        → 3 UAVs"),
]
ax.legend(handles=legend_patches, title="Population-density class (shown only)", loc="lower left", frameon=True)

mode = "pairwise-adjacent (K4)" if strict_pairwise else "connected"
focus_tag = f" incl. {focus_name}" if focus_name else ""
ax.set_title(
    f"Montréal – 4-borough {mode} cluster (density ≥ Medium; ≥{min_high} High){focus_tag}\n"
    f"High in set: {num_high} | Max total shared-border length = {score:.0f} m",
    fontsize=14
)
ax.set_axis_off()
plt.tight_layout()
plt.show()

print("Selected boroughs:", list(subset["NOM"]))
print("Density classes:  ", list(subset["DensityCat"]))
print("Num High:", num_high, "| Internal shared-border length (m):", round(score, 1))

