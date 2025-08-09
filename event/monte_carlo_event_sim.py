# monte_carlo_event_sim_real_time_v5_fixed.py
# Real-time Monte Carlo Event Simulation & Professional Multi-Category Heatmaps
# Uses Matplotlib for live updates. Includes smoothed density & improved visuals.
# VERSION 5: Correctly passes WARMUP_DURATION_SEC to run_live_simulation.
# Runs indefinitely until stopped with Ctrl+C.

import random
import time
import math
import numpy as np
import sys
import os
import collections
import traceback

# --- Try importing GeoPandas and Plotly ---
try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.errors import GEOSException
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("CRITICAL Error: Geopandas library not found (pip install geopandas).")
    print("              This library is essential for handling the geographic area.")
    sys.exit(1)

# --- Try importing Matplotlib and Scipy ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors # For normalization
    from scipy.ndimage import gaussian_filter # For smoothing
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("CRITICAL Error: Matplotlib or Scipy not found (pip install matplotlib scipy).")
    sys.exit(1)


# --- Parameters & Assumptions (Review and Adjust!) ---
CELL_SIZE_METERS = 100
EVENT_DURATION_SEC = 600 # ASSUMPTION: Fixed event lifespan
TIME_STEP_SEC = 0.5            # Simulation step size
WARMUP_DURATION_SEC = 60       # Time before collecting stats (e.g., 60s) - ADJUST
SAMPLING_INTERVAL_SEC = 5      # How often to update cumulative stats (e.g., 5s)
PLOT_UPDATE_INTERVAL_SEC = 1.0 # How often to redraw plots (e.g., 1s)

# Event categories and per-second generation rates (ASSUMPTIONS)
CATEGORY_RATES = collections.OrderedDict([
    ('Low', 40),    # 50%
    ('Medium', 24), # 30%
    ('High', 12),   # 15%
    ('Very_High', 4) # 5%
])
CATEGORIES = list(CATEGORY_RATES.keys())
NUM_CATEGORIES = len(CATEGORIES)
CATEGORY_IDS = {name: i for i, name in enumerate(CATEGORIES)}
CATEGORY_COLORS = plt.cm.get_cmap('viridis', NUM_CATEGORIES) # Colors for scatter plot

MONTREAL_BOUNDARY_FILE = "montreal_boundary.geojson" #<- REQUIRED FILE
TARGET_CRS = "EPSG:32188" # Projected CRS for Montreal area - VERIFY SUITABILITY

# --- Plotting Parameters ---
DENSITY_SMOOTHING_SIGMA = 1.5  # Sigma for Gaussian filter (adjust for more/less smoothing)
CMAP_DENSITY = 'plasma'      # Colormap for density/mean plots
CMAP_PROBABILITY = 'viridis' # Colormap for probability plots
VMAX_PERCENTILE = 99.5       # Use e.g., 99.5th percentile for density/mean vmax to handle outliers
PLOT1_BG_COLOR = 'white'
PLOT2_BG_COLOR = '#FFFFE0' # Light Yellow
PLOT3_BG_COLOR = '#F0F0F0' # Light Grey
PLOT3_CMAP = 'magma_r'     # Reversed: light background, dark high values
PLOT1_MARKERSIZE = 1       # Size for active event dots

# --- Other Assumptions ---
OVERLAP_RULE = 'Allow' # Only 'Allow' is implemented simply here.
PLACEMENT_RULE = 'Uniform' # Only uniform placement within polygon implemented.

# --- Global Variables ---
active_events = [] # Stores: [id, category_idx, x, y, expiry_time]
event_id_counter = 0

# --- Functions ---

def load_area_and_setup_grid(boundary_file, target_crs, cell_size_m):
    """Loads polygon, projects, calculates area/bounds, defines grid."""
    if not os.path.exists(boundary_file):
        raise FileNotFoundError(f"Boundary file '{boundary_file}' not found.")
    print(f"Loading boundary file: {boundary_file}...")
    gdf = gpd.read_file(boundary_file)
    print(f"Projecting to CRS: {target_crs}...")
    if not gdf.geometry.is_valid.all():
         print("Warning: Input geometries may not be valid, attempting fix with buffer(0).")
         gdf.geometry = gdf.geometry.buffer(0)
    gdf_proj = gdf.to_crs(target_crs)
    # Use unary_union (or union_all in newer GeoPandas)
    if hasattr(gdf_proj, 'union_all'):
        montreal_polygon = gdf_proj.union_all()
    else: # Fallback for older GeoPandas
        print("Using deprecated 'unary_union'. Consider updating GeoPandas.")
        montreal_polygon = gdf_proj.unary_union

    if not montreal_polygon.is_valid: montreal_polygon = montreal_polygon.buffer(0)
    if not montreal_polygon.is_valid or montreal_polygon.is_empty or montreal_polygon.area == 0:
         raise ValueError("Invalid, empty, or zero-area polygon after projection/buffer.")

    total_area_m2 = montreal_polygon.area
    minx, miny, maxx, maxy = montreal_polygon.bounds
    width_m, height_m = maxx - minx, maxy - miny
    if width_m <= 0 or height_m <= 0: raise ValueError("Area width or height is non-positive.")
    n_cols = int(math.ceil(width_m / cell_size_m))
    n_rows = int(math.ceil(height_m / cell_size_m))
    grid_info = {'min_x': minx, 'min_y': miny,'max_x': maxx, 'max_y': maxy,
                 'cell_size': cell_size_m,'n_cols': n_cols, 'n_rows': n_rows }
    print(f"Projection successful. Area: {total_area_m2 / 1e6:.2f} km^2")
    print(f"Bounds (m): minX={minx:.1f}, minY={miny:.1f}, maxX={maxx:.1f}, maxY={maxy:.1f}")
    print(f"Grid Dimensions: {n_cols} columns (X) x {n_rows} rows (Y)")
    # Prepare polygon geometry for potentially faster point-in-polygon checks
    from shapely.prepared import prep
    prepared = prep(montreal_polygon)
    print("Using prepared geometry for faster point-in-polygon checks.")
    return montreal_polygon, prepared, grid_info

def get_cell_index(x, y, grid_info):
    """Maps projected coordinates to grid cell (row, col) index."""
    if not (grid_info['min_x'] <= x <= grid_info['max_x'] and
            grid_info['min_y'] <= y <= grid_info['max_y']):
       if not (abs(x - grid_info['max_x']) < 1e-9 or abs(y - grid_info['max_y']) < 1e-9):
            return None # Outside bounding box + tolerance

    col = int((x - grid_info['min_x']) / grid_info['cell_size'])
    row = int((y - grid_info['min_y']) / grid_info['cell_size'])
    # Clamp indices to ensure they are within the valid range [0, n-1]
    col = max(0, min(col, grid_info['n_cols'] - 1))
    row = max(0, min(row, grid_info['n_rows'] - 1))
    return row, col

def setup_plots(grid_info):
    """Initializes the three Matplotlib figures."""
    plt.ion() # Turn on interactive mode

    # --- Figure 1: Active Event Scatter Plot ---
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    fig1.canvas.manager.set_window_title('Active Events')
    fig1.patch.set_facecolor(PLOT1_BG_COLOR)
    ax1.set_facecolor(PLOT1_BG_COLOR)
    ax1.set_title("Live Active Events")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.set_xlim(grid_info['min_x'], grid_info['max_x'])
    ax1.set_ylim(grid_info['min_y'], grid_info['max_y'])
    ax1.set_aspect('equal', adjustable='box')
    # Add legend placeholders
    for i, cat in enumerate(CATEGORIES):
        ax1.plot([], [], '.', color=CATEGORY_COLORS(i / NUM_CATEGORIES), label=cat, markersize=PLOT1_MARKERSIZE*2)
    ax1.legend(loc='upper right', fontsize='small')
    fig1.tight_layout()

    # --- Figure 2: Mean E[N] Heatmap ---
    fig2, axes2 = plt.subplots(nrows=1, ncols=NUM_CATEGORIES, figsize=(5 * NUM_CATEGORIES, 5),
                              sharex=True, sharey=True, constrained_layout=True)
    fig2.canvas.manager.set_window_title('Mean Event Count E[N]')
    fig2.patch.set_facecolor(PLOT2_BG_COLOR)
    fig2.suptitle("Estimated Mean Event Count E[N] per Cell")
    images_mean = []
    extent = [grid_info['min_x'], grid_info['min_x'] + grid_info['n_cols'] * grid_info['cell_size'],
              grid_info['min_y'], grid_info['min_y'] + grid_info['n_rows'] * grid_info['cell_size']]
    for i, cat in enumerate(CATEGORIES):
        ax = axes2[i] if NUM_CATEGORIES > 1 else axes2
        ax.set_facecolor(PLOT2_BG_COLOR)
        im = ax.imshow(np.zeros((grid_info['n_rows'], grid_info['n_cols'])), origin='lower', extent=extent,
                       cmap='plasma', aspect='equal', vmin=0) # Use specified cmap
        ax.set_title(cat)
        ax.set_xlabel("Easting (m)")
        if i == 0: ax.set_ylabel("Northing (m)")
        images_mean.append(im)
    cbar_mean = fig2.colorbar(images_mean[-1], ax=axes2, orientation='vertical', pad=0.02)
    cbar_mean.set_label('Mean Events / Cell')

    # --- Figure 3: Probability P(N>=1) Heatmap ---
    fig3, axes3 = plt.subplots(nrows=1, ncols=NUM_CATEGORIES, figsize=(5 * NUM_CATEGORIES, 5),
                              sharex=True, sharey=True, constrained_layout=True)
    fig3.canvas.manager.set_window_title('Event Probability P(N>=1)')
    fig3.patch.set_facecolor(PLOT3_BG_COLOR)
    fig3.suptitle("Estimated Probability P(N>=1) per Cell")
    images_prob = []
    for i, cat in enumerate(CATEGORIES):
        ax = axes3[i] if NUM_CATEGORIES > 1 else axes3
        ax.set_facecolor(PLOT3_BG_COLOR)
        im = ax.imshow(np.zeros((grid_info['n_rows'], grid_info['n_cols'])), origin='lower', extent=extent,
                       cmap=PLOT3_CMAP, aspect='equal', vmin=0, vmax=1.0) # Use specified cmap and fixed 0-1 scale
        ax.set_title(cat)
        ax.set_xlabel("Easting (m)")
        if i == 0: ax.set_ylabel("Northing (m)")
        images_prob.append(im)
    cbar_prob = fig3.colorbar(images_prob[-1], ax=axes3, orientation='vertical', pad=0.02)
    cbar_prob.set_label('Probability')

    # Store figures and objects for updating
    figures = {'active': fig1, 'mean': fig2, 'prob': fig3}
    axes = {'active': ax1, 'mean': axes2, 'prob': axes3}
    image_mappables = {'mean': images_mean, 'prob': images_prob}
    colorbars = {'mean': cbar_mean, 'prob': cbar_prob}

    plt.show(block=False)
    plt.pause(0.1) # Allow time for windows to draw

    return figures, axes, image_mappables, colorbars


def update_plots(figures, axes, image_mappables, colorbars, active_events_list,
                 mean_count_grid, prob_grid, grid_info, current_sim_time, num_snapshots):
    """Updates the data in the Matplotlib figures."""

    # --- Update Figure 1: Active Event Scatter Plot ---
    ax1 = axes['active']
    # Clear previous points efficiently - remove lines but keep legend handles
    current_lines = ax1.get_lines()
    # Keep only the initial legend placeholder lines
    num_legend_lines = len(CATEGORIES)
    for line in current_lines[num_legend_lines:]: # Remove lines added in previous updates
        line.remove()

    events_by_cat = [[] for _ in range(NUM_CATEGORIES)]
    for _, cat_idx, x, y, _ in active_events_list:
        events_by_cat[cat_idx].append((x, y))

    handles, _ = ax1.get_legend_handles_labels() # Get original handles
    new_labels = []
    for i, cat_events in enumerate(events_by_cat):
        if cat_events:
            x_coords, y_coords = zip(*cat_events)
            # Plot new data, importantly, do NOT provide a label here again
            ax1.plot(x_coords, y_coords, '.', color=CATEGORY_COLORS(i / NUM_CATEGORIES), markersize=PLOT1_MARKERSIZE)
        new_labels.append(f"{CATEGORIES[i]} ({len(cat_events)})") # Update count for legend

    # Update legend with new labels but original handles
    ax1.legend(handles=handles, labels=new_labels, loc='upper right', fontsize='small')
    ax1.set_title(f"Live Active Events (T={current_sim_time:.1f}s, N={len(active_events_list)})")

    # --- Update Figure 2: Mean E[N] Heatmap ---
    max_mean_overall = 0.1
    if mean_count_grid is not None and mean_count_grid.size > 0:
        all_mean_values = mean_count_grid[mean_count_grid > 0]
        if all_mean_values.size > 0:
            max_mean_overall = max(0.1, np.percentile(all_mean_values, VMAX_PERCENTILE))

    for i in range(NUM_CATEGORIES):
        ax = axes['mean'][i] if NUM_CATEGORIES > 1 else axes['mean']
        im = image_mappables['mean'][i]
        # Data needs to be (rows, cols) for imshow, might need transpose depending on source
        im.set_data(mean_count_grid[i])
        im.set_clim(0, max_mean_overall)

    if num_snapshots > 0 and colorbars['mean'].mappable.get_array().size > 0 : # Check mappable has data
         try:
             colorbars['mean'].mappable.set_clim(0, max_mean_overall)
         except Exception as e_cbar: # Handle potential edge cases during update
             print(f"Minor issue updating mean colorbar limits: {e_cbar}")


    # --- Update Figure 3: Probability P(N>=1) Heatmap ---
    for i in range(NUM_CATEGORIES):
        ax = axes['prob'][i] if NUM_CATEGORIES > 1 else axes['prob']
        im = image_mappables['prob'][i]
        # Data needs to be (rows, cols) for imshow
        im.set_data(prob_grid[i])
        # vmin/vmax are fixed at 0/1, no need to update clim

    # --- Redraw Canvases ---
    try:
        for fig in figures.values():
            fig.canvas.draw_idle()
        plt.pause(0.01) # Crucial pause to allow GUI event loop to process drawing
    except Exception as e:
        print(f"Warning: Error during plot update - {e}")


# --- Main Simulation Function ---
def run_live_simulation_multiplot(polygon, prepared, grid_info, warmup_duration_sec): # Argument passed correctly
    """Runs simulation and updates multiple Matplotlib plots."""
    global active_events, event_id_counter

    presence_counts = np.zeros((NUM_CATEGORIES, grid_info['n_rows'], grid_info['n_cols']), dtype=np.uint32)
    total_counts = np.zeros((NUM_CATEGORIES, grid_info['n_rows'], grid_info['n_cols']), dtype=np.uint64)
    total_snapshots = 0
    active_events = []
    accumulators = {i: 0.0 for i in range(NUM_CATEGORIES)}
    event_id = 0
    current_time = 0.0
    next_sample_time = SAMPLING_INTERVAL_SEC if warmup_duration_sec == 0 else warmup_duration_sec # Uses argument
    last_plot_update_wall_time = time.time()

    # Setup the plot figures
    figures, axes, image_mappables, colorbars = setup_plots(grid_info)
    polygon_contains = prepared.contains
    min_x, min_y, max_x, max_y = polygon.bounds

    print("\n--- Starting Simulation Loop ---")
    print("Press Ctrl+C in terminal to stop.")
    start_real_time = time.time()

    try:
        # --- The main infinite loop ---
        while True:
            loop_start_time = time.time()

            # 1. Expire Events
            active_events = [e for e in active_events if current_time < e[4]]

            # 2. Generate New Events
            for cat_idx, rate in enumerate(CATEGORY_RATES.values()):
                accumulators[cat_idx] += rate * TIME_STEP_SEC
                num_to_gen = int(accumulators[cat_idx])
                if num_to_gen > 0:
                    accumulators[cat_idx] -= num_to_gen
                    for _ in range(num_to_gen):
                        for _ in range(100): # Placement attempts
                            x = random.uniform(min_x, max_x)
                            y = random.uniform(min_y, max_y)
                            if polygon_contains(Point(x, y)):
                                event_id += 1
                                expiry = current_time + EVENT_DURATION_SEC
                                active_events.append([event_id, cat_idx, x, y, expiry])
                                break

            # 3. Sample Statistics
            if current_time >= warmup_duration_sec and current_time >= next_sample_time - TIME_STEP_SEC / 2:
                total_snapshots += 1
                current_counts_snap = np.zeros((NUM_CATEGORIES, grid_info['n_rows'], grid_info['n_cols']), dtype=np.uint16)
                for _, cat_idx, x, y, _ in active_events:
                    idx = get_cell_index(x, y, grid_info)
                    if idx:
                        r, c = idx
                        current_counts_snap[cat_idx, r, c] += 1
                presence_counts += (current_counts_snap >= 1).astype(np.uint32)
                total_counts += current_counts_snap.astype(np.uint64)
                while next_sample_time <= current_time + TIME_STEP_SEC / 2:
                    next_sample_time += SAMPLING_INTERVAL_SEC

            # 4. Update Plots Periodically
            now_wall_clock = time.time()
            if now_wall_clock - last_plot_update_wall_time >= PLOT_UPDATE_INTERVAL_SEC:
                last_plot_update_wall_time = now_wall_clock

                mean_count_est = np.zeros_like(total_counts, dtype=float)
                prob_map_est = np.zeros_like(presence_counts, dtype=float)
                if total_snapshots > 0:
                    mean_count_est = total_counts / total_snapshots
                    prob_map_est = presence_counts / total_snapshots

                update_plots(figures, axes, image_mappables, colorbars,
                             active_events, mean_count_est, prob_map_est,
                             grid_info, current_time, total_snapshots)

                print(f"  Plot Update @ Sim Time: {current_time:.1f}s | Active: {len(active_events)} | Snaps: {total_snapshots}", end='\r')


            # 5. Advance Time
            current_time += TIME_STEP_SEC

            # 6. Control Loop Speed
            loop_end_time = time.time()
            wall_time_elapsed = loop_end_time - loop_start_time
            sleep_time = max(0, TIME_STEP_SEC - wall_time_elapsed)
            time.sleep(min(0.02, sleep_time)) # Slightly longer min sleep


    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        traceback.print_exc()
    finally:
        print("\nCleaning up plots...")
        plt.ioff()
        # Try closing all figures that might have been created
        for fig_id in plt.get_fignums():
             plt.close(fig_id)
        print("Simulation Finished.")


# --- Main Execution ---
if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE or not GEOPANDAS_AVAILABLE: sys.exit(1)

    # Command-line duration is ignored for the 'run_live_simulation_multiplot' as it runs until Ctrl+C
    print("NOTE: Real-time simulation runs indefinitely until stopped with Ctrl+C.")

    try:
        print("="*60)
        print("Real-Time Monte Carlo Simulation with Multi-Window Plots")
        print("Version 6 (Corrected WARMUP_DURATION_SEC Pass)")
        print("="*60)
        poly, prep, grid = load_area_and_setup_grid(
            MONTREAL_BOUNDARY_FILE, TARGET_CRS, CELL_SIZE_METERS
        )
        # Pass WARMUP_DURATION_SEC correctly here
        run_live_simulation_multiplot(poly, prep, grid, WARMUP_DURATION_SEC)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the boundary file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
