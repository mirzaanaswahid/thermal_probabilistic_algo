# monte_carlo_event_sim_real_time_v7_single_script_pop_dummy.py

import random
import time
import math
import numpy as np
import sys
import os
import collections
import traceback

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.errors import GEOSException
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("CRITICAL Error: Geopandas library not found (pip install geopandas).")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("CRITICAL Error: Matplotlib or Scipy not found (pip install matplotlib scipy).")
    sys.exit(1)

CELL_SIZE_METERS = 100
EVENT_DURATION_SEC = 45
TIME_STEP_SEC = 0.5
WARMUP_DURATION_SEC = 60
SAMPLING_INTERVAL_SEC = 5
PLOT_UPDATE_INTERVAL_SEC = 1.0

CATEGORY_RATES = collections.OrderedDict([
    ('Low', 40), ('Medium', 24), ('High', 12), ('Very_High', 4)
])
CATEGORIES = list(CATEGORY_RATES.keys())
NUM_CATEGORIES = len(CATEGORIES)
CATEGORY_IDS = {name: i for i, name in enumerate(CATEGORIES)}
CATEGORY_COLORS = plt.cm.get_cmap('viridis', NUM_CATEGORIES)

MONTREAL_BOUNDARY_FILE = "montreal_boundary.geojson"
TARGET_CRS = "EPSG:32188"

DENSITY_SMOOTHING_SIGMA = 1.5
CMAP_DENSITY = 'plasma'
CMAP_PROBABILITY = 'viridis'
VMAX_PERCENTILE = 99.5
PLOT1_BG_COLOR = 'white'
PLOT2_BG_COLOR = '#FFFFE0'
PLOT3_BG_COLOR = '#F0F0F0'
PLOT3_CMAP = 'magma_r'
PLOT1_MARKERSIZE = 1

OVERLAP_RULE = 'Allow'
PLACEMENT_RULE = 'PopulationWeighted' # Changed default

active_events = []
event_id_counter = 0

def load_area_and_setup_grid(boundary_file, target_crs, cell_size_m):
    if not os.path.exists(boundary_file):
        raise FileNotFoundError(f"Boundary file '{boundary_file}' not found.")
    print(f"Loading boundary file: {boundary_file}...")
    gdf = gpd.read_file(boundary_file)
    print(f"Projecting to CRS: {target_crs}...")
    if not gdf.geometry.is_valid.all():
         print("Warning: Input geometries may not be valid, attempting fix with buffer(0).")
         gdf.geometry = gdf.geometry.buffer(0)
    gdf_proj = gdf.to_crs(target_crs)
    if hasattr(gdf_proj, 'union_all'):
        montreal_polygon = gdf_proj.union_all()
    else:
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
    from shapely.prepared import prep
    prepared = prep(montreal_polygon)
    print("Using prepared geometry for faster point-in-polygon checks.")
    return montreal_polygon, prepared, grid_info

def get_cell_index(x, y, grid_info):
    if not (grid_info['min_x'] <= x <= grid_info['max_x'] and
            grid_info['min_y'] <= y <= grid_info['max_y']):
       if not (abs(x - grid_info['max_x']) < 1e-9 or abs(y - grid_info['max_y']) < 1e-9):
            return None
    col = int((x - grid_info['min_x']) / grid_info['cell_size'])
    row = int((y - grid_info['min_y']) / grid_info['cell_size'])
    col = max(0, min(col, grid_info['n_cols'] - 1))
    row = max(0, min(row, grid_info['n_rows'] - 1))
    return row, col

def create_dummy_density_grid(grid_info):
    print("WARNING: Generating DUMMY population density grid (center peak).")
    print("         Replace this with loading real pre-processed data for research.")
    n_rows, n_cols = grid_info['n_rows'], grid_info['n_cols']
    center_r, center_c = n_rows / 2, n_cols / 2
    max_dist = math.sqrt(center_r**2 + center_c**2)
    dummy_grid = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            dist = math.sqrt((r - center_r)**2 + (c - center_c)**2)
            # Simple linear decay from center - adjust formula for different patterns
            dummy_grid[r, c] = max(0, 1.0 - (dist / max_dist))
    dummy_grid = dummy_grid * 10000 # Scale up to resemble population density values
    print(f"Dummy density grid created with max value: {dummy_grid.max():.0f}")
    return dummy_grid

class EventGenerator:
    def __init__(self, category_rates, event_duration_sec, placement_rule,
                 population_density_grid, grid_info, polygon_checker, overlap_rule):
        self.category_rates = category_rates
        self.num_categories = len(category_rates)
        self.event_duration_sec = event_duration_sec
        self.placement_rule = placement_rule
        self.population_density_grid = population_density_grid
        self.grid_info = grid_info
        self.polygon_checker = polygon_checker
        self.overlap_rule = overlap_rule
        self.accumulators = {i: 0.0 for i in range(self.num_categories)}
        self.min_x, self.min_y = grid_info['min_x'], grid_info['min_y']
        self.max_x, self.max_y = grid_info['max_x'], grid_info['max_y']
        self.max_density = np.max(population_density_grid) if placement_rule == 'PopulationWeighted' and population_density_grid is not None else 1.0
        self.event_id_counter = 0

    def _get_cell_density(self, r, c):
         if self.population_density_grid is not None and \
            0 <= r < self.grid_info['n_rows'] and 0 <= c < self.grid_info['n_cols']:
              return self.population_density_grid[r, c]
         return 0

    def _place_event_uniformly(self):
         attempts = 0; max_attempts = 100
         while attempts < max_attempts:
            attempts += 1
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            if self.polygon_checker(Point(x, y)):
                return x, y
         return None

    def _place_event_population_weighted(self):
        if self.population_density_grid is None or self.max_density <= 0:
            # Fallback to uniform if density grid is missing or all zero
            if self.population_density_grid is None:
                 print("Warning: Population density grid not available, falling back to uniform placement.")
            else:
                 print("Warning: Max population density is zero, falling back to uniform placement.")
            self.placement_rule = 'Uniform' # Change rule permanently for this run
            return self._place_event_uniformly()

        attempts = 0; max_attempts = 500
        while attempts < max_attempts:
            attempts += 1
            x_cand = random.uniform(self.min_x, self.max_x)
            y_cand = random.uniform(self.min_y, self.max_y)
            idx = get_cell_index(x_cand, y_cand, self.grid_info)
            if idx:
                r, c = idx
                cell_density = self._get_cell_density(r, c)
                if cell_density <= 0: continue
                u = random.uniform(0, self.max_density)
                if u < cell_density:
                    if self.polygon_checker(Point(x_cand, y_cand)):
                         return x_cand, y_cand
        return None

    def generate(self, dt, current_time):
        global event_id_counter # Use global counter
        new_events = []
        category_counts_to_gen = {}
        num_to_generate_total = 0
        for i, (cat_name, rate_c) in enumerate(self.category_rates.items()):
            self.accumulators[i] += rate_c * dt
            num_for_cat = int(self.accumulators[i])
            if num_for_cat > 0:
                 category_counts_to_gen[i] = num_for_cat
                 num_to_generate_total += num_for_cat
                 self.accumulators[i] -= num_for_cat

        for cat_idx, num_gen in category_counts_to_gen.items():
            for _ in range(num_gen):
                location = None
                if self.placement_rule == 'Uniform':
                    location = self._place_event_uniformly()
                elif self.placement_rule == 'PopulationWeighted':
                     location = self._place_event_population_weighted()
                else:
                     location = self._place_event_uniformly()

                if location:
                    x, y = location
                    can_add = True
                    if self.overlap_rule == 'Prevent':
                        # Requires checking 'active_events' list passed in or accessed globally
                        # Add complex check here
                        print("Warning: 'Prevent' overlap rule check is NOT implemented.")
                        pass
                    if can_add:
                        event_id_counter += 1
                        expiry = current_time + self.event_duration_sec
                        new_events.append([event_id_counter, cat_idx, x, y, expiry])
        return new_events

class StatsCollector:
    def __init__(self, grid_info, num_categories, warmup_sec, sampling_interval_sec):
        self.grid_info = grid_info
        self.num_categories = num_categories
        self.warmup_sec = warmup_sec
        self.sampling_interval_sec = sampling_interval_sec
        self.presence_counts = np.zeros((num_categories, grid_info['n_rows'], grid_info['n_cols']), dtype=np.uint32)
        self.total_counts = np.zeros((num_categories, grid_info['n_rows'], grid_info['n_cols']), dtype=np.uint64)
        self.total_snapshots = 0
        self.next_sample_time = sampling_interval_sec if warmup_sec == 0 else warmup_sec

    def sample(self, current_time, active_events, dt): # Pass dt for tolerance
        sampled_this_step = False
        if current_time >= self.warmup_sec and current_time >= self.next_sample_time - dt / 2:
            self.total_snapshots += 1
            sampled_this_step = True
            current_counts_snap = np.zeros((self.num_categories, self.grid_info['n_rows'], self.grid_info['n_cols']), dtype=np.uint16)
            for _, cat_idx, x, y, _ in active_events:
                idx = get_cell_index(x, y, self.grid_info)
                if idx:
                    r, c = idx
                    current_counts_snap[cat_idx, r, c] += 1
            self.presence_counts += (current_counts_snap >= 1).astype(np.uint32)
            self.total_counts += current_counts_snap.astype(np.uint64)

            while self.next_sample_time <= current_time + dt / 2:
                 self.next_sample_time += self.sampling_interval_sec
        return sampled_this_step

    def get_probabilities(self):
        if self.total_snapshots == 0: return np.zeros_like(self.presence_counts, dtype=float)
        # Use np.errstate to avoid warning for division by zero if needed, though checked
        with np.errstate(divide='ignore', invalid='ignore'):
            probs = self.presence_counts.astype(np.float64) / self.total_snapshots
        return np.nan_to_num(probs) # Convert NaN to 0

    def get_expected_values(self):
        if self.total_snapshots == 0: return np.zeros_like(self.total_counts, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            exp_vals = self.total_counts.astype(np.float64) / self.total_snapshots
        return np.nan_to_num(exp_vals)


class RealTimeVisualizer:
    def __init__(self, grid_info, categories, category_colors, plotting_params):
        if not MATPLOTLIB_AVAILABLE: raise ImportError("Matplotlib required for visualization.")
        self.grid_info = grid_info
        self.categories = categories
        self.num_categories = len(categories)
        self.category_colors = category_colors
        self.params = plotting_params
        self.figures = {}
        self.axes = {}
        self.image_mappables = {}
        self.colorbars = {}
        self.last_plot_update_wall_time = time.time()
        self._setup_plots()

    def _setup_plots(self):
        plt.ion()
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        fig1.canvas.manager.set_window_title('Active Events')
        fig1.patch.set_facecolor(self.params['PLOT1_BG_COLOR'])
        ax1.set_facecolor(self.params['PLOT1_BG_COLOR'])
        ax1.set_title("Live Active Events")
        ax1.set_xlabel("Easting (m)")
        ax1.set_ylabel("Northing (m)")
        ax1.set_xlim(self.grid_info['min_x'], self.grid_info['max_x'])
        ax1.set_ylim(self.grid_info['min_y'], self.grid_info['max_y'])
        ax1.set_aspect('equal', adjustable='box')
        for i, cat in enumerate(self.categories):
            ax1.plot([], [], '.', color=self.category_colors(i / self.num_categories), label=cat, markersize=self.params['PLOT1_MARKERSIZE']*2)
        ax1.legend(loc='upper right', fontsize='small')
        fig1.tight_layout()
        self.figures['active'] = fig1
        self.axes['active'] = ax1

        fig2, axes2 = plt.subplots(nrows=1, ncols=self.num_categories, figsize=(5 * self.num_categories, 5),
                                  sharex=True, sharey=True, constrained_layout=True)
        fig2.canvas.manager.set_window_title('Mean Event Count E[N]')
        fig2.patch.set_facecolor(self.params['PLOT2_BG_COLOR'])
        fig2.suptitle("Estimated Mean Event Count E[N] per Cell")
        images_mean = []
        extent = [self.grid_info['min_x'], self.grid_info['min_x'] + self.grid_info['n_cols'] * self.grid_info['cell_size'],
                  self.grid_info['min_y'], self.grid_info['min_y'] + self.grid_info['n_rows'] * self.grid_info['cell_size']]
        for i, cat in enumerate(self.categories):
            ax = axes2[i] if self.num_categories > 1 else axes2
            ax.set_facecolor(self.params['PLOT2_BG_COLOR'])
            im = ax.imshow(np.zeros((self.grid_info['n_rows'], self.grid_info['n_cols'])), origin='lower', extent=extent,
                           cmap=self.params['CMAP_DENSITY'], aspect='equal', vmin=0)
            ax.set_title(cat)
            ax.set_xlabel("Easting (m)")
            if i == 0: ax.set_ylabel("Northing (m)")
            images_mean.append(im)
        cbar_mean = fig2.colorbar(images_mean[-1], ax=axes2, orientation='vertical', pad=0.02)
        cbar_mean.set_label('Mean Events / Cell')
        self.figures['mean'] = fig2
        self.axes['mean'] = axes2
        self.image_mappables['mean'] = images_mean
        self.colorbars['mean'] = cbar_mean

        fig3, axes3 = plt.subplots(nrows=1, ncols=self.num_categories, figsize=(5 * self.num_categories, 5),
                                  sharex=True, sharey=True, constrained_layout=True)
        fig3.canvas.manager.set_window_title('Event Probability P(N>=1)')
        fig3.patch.set_facecolor(self.params['PLOT3_BG_COLOR'])
        fig3.suptitle("Estimated Probability P(N>=1) per Cell")
        images_prob = []
        for i, cat in enumerate(self.categories):
            ax = axes3[i] if self.num_categories > 1 else axes3
            ax.set_facecolor(self.params['PLOT3_BG_COLOR'])
            im = ax.imshow(np.zeros((self.grid_info['n_rows'], self.grid_info['n_cols'])), origin='lower', extent=extent,
                           cmap=self.params['PLOT3_CMAP'], aspect='equal', vmin=0, vmax=1.0)
            ax.set_title(cat)
            ax.set_xlabel("Easting (m)")
            if i == 0: ax.set_ylabel("Northing (m)")
            images_prob.append(im)
        cbar_prob = fig3.colorbar(images_prob[-1], ax=axes3, orientation='vertical', pad=0.02)
        cbar_prob.set_label('Probability')
        self.figures['prob'] = fig3
        self.axes['prob'] = axes3
        self.image_mappables['prob'] = images_prob
        self.colorbars['prob'] = cbar_prob

        plt.show(block=False)
        plt.pause(0.1)

    def update(self, active_events_list, mean_count_grid, prob_grid, current_sim_time, num_snapshots):
        now_wall_clock = time.time()
        if now_wall_clock - self.last_plot_update_wall_time < self.params['PLOT_UPDATE_INTERVAL_SEC']:
            return # Skip update if too soon
        self.last_plot_update_wall_time = now_wall_clock

        # --- Update Scatter ---
        ax1 = self.axes['active']
        current_lines = ax1.get_lines()
        num_legend_lines = len(self.categories)
        for line in current_lines[num_legend_lines:]: line.remove()
        events_by_cat = [[] for _ in range(self.num_categories)]
        for _, cat_idx, x, y, _ in active_events_list: events_by_cat[cat_idx].append((x, y))
        handles, _ = ax1.get_legend_handles_labels()
        new_labels = []
        for i, cat_events in enumerate(events_by_cat):
            if cat_events:
                x_coords, y_coords = zip(*cat_events)
                ax1.plot(x_coords, y_coords, '.', color=self.category_colors(i / self.num_categories), markersize=self.params['PLOT1_MARKERSIZE'])
            new_labels.append(f"{self.categories[i]} ({len(cat_events)})")
        ax1.legend(handles=handles, labels=new_labels, loc='upper right', fontsize='small')
        ax1.set_title(f"Live Active Events (T={current_sim_time:.1f}s, N={len(active_events_list)})")

        # --- Update Mean Heatmap ---
        max_mean_overall = 0.1
        if mean_count_grid is not None and mean_count_grid.size > 0:
            all_mean_values = mean_count_grid[mean_count_grid > 0]
            if all_mean_values.size > 0:
                max_mean_overall = max(0.1, np.percentile(all_mean_values, self.params['VMAX_PERCENTILE']))
        for i in range(self.num_categories):
            ax = self.axes['mean'][i] if self.num_categories > 1 else self.axes['mean']
            im = self.image_mappables['mean'][i]
            im.set_data(mean_count_grid[i])
            im.set_clim(0, max_mean_overall)
        if num_snapshots > 0 and self.colorbars['mean'].mappable.get_array().size > 0 :
            try: self.colorbars['mean'].mappable.set_clim(0, max_mean_overall)
            except Exception as e_cbar: print(f"Minor issue updating mean colorbar limits: {e_cbar}")

        # --- Update Probability Heatmap ---
        for i in range(self.num_categories):
            ax = self.axes['prob'][i] if self.num_categories > 1 else self.axes['prob']
            im = self.image_mappables['prob'][i]
            im.set_data(prob_grid[i])

        # --- Redraw ---
        try:
            for fig in self.figures.values(): fig.canvas.draw_idle()
            plt.pause(0.01)
            print(f"  Plot Update @ Sim Time: {current_sim_time:.1f}s | Active: {len(active_events_list)} | Snaps: {num_snapshots}", end='\r')
        except Exception as e: print(f"Warning: Error during plot update - {e}")

    def close(self):
        if MATPLOTLIB_AVAILABLE:
            plt.ioff()
            for fig_id in plt.get_fignums():
                plt.close(fig_id)

# --- Main Simulation Function ---
def run_simulation_main_loop(polygon, prepared, grid_info, warmup_duration_sec):
    global active_events # Modify global list

    plotting_params_dict = {
         'PLOT1_BG_COLOR': PLOT1_BG_COLOR, 'PLOT2_BG_COLOR': PLOT2_BG_COLOR,
         'PLOT3_BG_COLOR': PLOT3_BG_COLOR, 'PLOT3_CMAP': PLOT3_CMAP,
         'PLOT1_MARKERSIZE': PLOT1_MARKERSIZE, 'VMAX_PERCENTILE': VMAX_PERCENTILE,
         'DENSITY_SMOOTHING_SIGMA': DENSITY_SMOOTHING_SIGMA, 'CMAP_DENSITY': CMAP_DENSITY,
         'CMAP_PROBABILITY': CMAP_PROBABILITY, 'PLOT_UPDATE_INTERVAL_SEC': PLOT_UPDATE_INTERVAL_SEC
    }

    density_grid = create_dummy_density_grid(grid_info) # Use dummy grid

    event_generator = EventGenerator(
        CATEGORY_RATES, EVENT_DURATION_SEC, PLACEMENT_RULE,
        density_grid, grid_info, prepared.contains, OVERLAP_RULE
    )
    stats_collector = StatsCollector(grid_info, NUM_CATEGORIES, warmup_duration_sec, SAMPLING_INTERVAL_SEC)
    visualizer = RealTimeVisualizer(grid_info, CATEGORIES, CATEGORY_COLORS, plotting_params_dict)

    current_time = 0.0

    print("\n--- Starting Simulation Loop ---")
    print("Press Ctrl+C in terminal to stop.")

    try:
        while True:
            loop_start_time = time.time()

            # 1. Expire Events
            active_events = [e for e in active_events if current_time < e[4]]

            # 2. Generate New Events
            newly_generated = event_generator.generate(TIME_STEP_SEC, current_time)
            active_events.extend(newly_generated)

            # 3. Sample Statistics
            sampled_this_step = stats_collector.sample(current_time, active_events, TIME_STEP_SEC)

            # 4. Update visualization periodically
            prob_grid = stats_collector.get_probabilities()
            mean_grid = stats_collector.get_expected_values()
            visualizer.update(active_events, mean_grid, prob_grid,
                              current_time, stats_collector.total_snapshots)

            # 5. Advance Time
            current_time += TIME_STEP_SEC

            # 6. Control Loop Speed
            loop_end_time = time.time()
            wall_time_elapsed = loop_end_time - loop_start_time
            sleep_time = max(0, TIME_STEP_SEC - wall_time_elapsed)
            time.sleep(min(0.02, sleep_time))

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during simulation: {e}")
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        visualizer.close()
        print("Simulation Finished.")

# --- Main Execution ---
if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE or not GEOPANDAS_AVAILABLE: sys.exit(1)

    print("NOTE: Real-time simulation runs indefinitely until stopped with Ctrl+C.")
    print("WARNING: Using DUMMY population density grid for demonstration.")

    try:
        print("="*60)
        print("Modular Monte Carlo Simulation with Live Plots")
        print("="*60)
        poly, prep, grid = load_area_and_setup_grid(
            MONTREAL_BOUNDARY_FILE, TARGET_CRS, CELL_SIZE_METERS
        )
        run_simulation_main_loop(poly, prep, grid, WARMUP_DURATION_SEC)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the boundary file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
