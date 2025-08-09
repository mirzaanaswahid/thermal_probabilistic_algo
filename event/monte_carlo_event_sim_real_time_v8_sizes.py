# mc_sim_modes_v4_fixed.py

import random
import time
import math
import numpy as np
import sys
import os
import collections
import traceback
import argparse

try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.errors import GEOSException
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False; print("CRITICAL Error: Geopandas not found (pip install geopandas)."); sys.exit(1)
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.ndimage import gaussian_filter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False; print("Warning: Matplotlib/Scipy not found (pip install matplotlib scipy). Realtime mode disabled.")
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False; print("Warning: Plotly not found (pip install plotly). Replication plot saving disabled.")
try:
    import pandas as pd
    from tabulate import tabulate
    TABLES_AVAILABLE = True
except ImportError:
     TABLES_AVAILABLE = False; print("Warning: Pandas/Tabulate not found (pip install pandas tabulate). Summary table disabled.")

CELL_SIZE_METERS = 100
EVENT_DURATION_SEC = 45
TIME_STEP_SEC = 0.5
WARMUP_DURATION_SEC = 60
MINIMUM_SEPARATION_METERS = 50.0
CATEGORY_RATES = collections.OrderedDict([('Low', 40),('Medium', 24),('High', 12),('Very_High', 4)])
EVENT_RADIUS_RANGES_METERS = {'Low':(20,25),'Medium':(25,30),'High':(30,35),'Very_High':(35,40)}
PLACEMENT_RULE = 'PopulationWeighted'
OVERLAP_RULE = 'SeparationDistance'
NUM_REPLICATIONS = 5
SIMULATION_DURATION_PER_REP = 180
SAMPLING_INTERVAL_SEC = 5
PLOT_UPDATE_INTERVAL_SEC = 2.0
MONTREAL_BOUNDARY_FILE = "montreal_boundary.geojson"
TARGET_CRS = "EPSG:32188"
CATEGORIES = list(CATEGORY_RATES.keys())
NUM_CATEGORIES = len(CATEGORIES)
CATEGORY_IDS = {name: i for i, name in enumerate(CATEGORIES)}
DENSITY_SMOOTHING_SIGMA = 1.5
VMAX_PERCENTILE = 99.5
CATEGORY_COLORS_ARRAY = None
if MATPLOTLIB_AVAILABLE:
    try:
        cmap_obj = matplotlib.colormaps.get_cmap('viridis')
        CATEGORY_COLORS_ARRAY = cmap_obj(np.linspace(0, 1, NUM_CATEGORIES))
    except ValueError:
        colors_fallback = ['blue', 'green', 'red', 'purple', 'orange']
        CATEGORY_COLORS_ARRAY = colors_fallback[:NUM_CATEGORIES]
PLOT1_BG_COLOR = 'white'; PLOT2_BG_COLOR = '#FFFFE0'; PLOT3_BG_COLOR = '#F0F0F0'
PLOT3_CMAP = 'magma_r'; PLOT1_MARKERSIZE = 2; CMAP_DENSITY_RT = 'plasma'
CMAP_MEAN = 'viridis'; CMAP_STDDEV = 'magma'; CMAP_PROBABILITY = PLOT3_CMAP
MEAN_PROB_PLOT_HTML_FILE = "mc_rep_mean_probability.html"
STD_PROB_PLOT_HTML_FILE = "mc_rep_std_dev_probability.html"
MEAN_EXP_PLOT_HTML_FILE = "mc_rep_mean_expected_count.html"
STD_EXP_PLOT_HTML_FILE = "mc_rep_std_dev_expected_count.html"

active_events = []
event_id_counter = 0

def load_area_and_setup_grid(boundary_file, target_crs, cell_size_m):
    if not os.path.exists(boundary_file): raise FileNotFoundError(f"Boundary file '{boundary_file}' not found.")
    print(f"Loading boundary file: {boundary_file}...")
    gdf = gpd.read_file(boundary_file)
    print(f"Projecting to CRS: {target_crs}...")
    if not gdf.geometry.is_valid.all(): gdf.geometry = gdf.geometry.buffer(0)
    gdf_proj = gdf.to_crs(target_crs)
    if hasattr(gdf_proj, 'union_all'): montreal_polygon = gdf_proj.union_all()
    else: montreal_polygon = gdf_proj.unary_union
    if not montreal_polygon.is_valid: montreal_polygon = montreal_polygon.buffer(0)
    if not montreal_polygon.is_valid or montreal_polygon.is_empty or montreal_polygon.area == 0: raise ValueError("Invalid polygon.")
    total_area_m2 = montreal_polygon.area
    minx, miny, maxx, maxy = montreal_polygon.bounds
    width_m, height_m = maxx - minx, maxy - miny
    if width_m <= 0 or height_m <= 0: raise ValueError("Area width/height non-positive.")
    n_cols = int(math.ceil(width_m / cell_size_m)); n_rows = int(math.ceil(height_m / cell_size_m))
    grid_info = {'min_x':minx,'min_y':miny,'max_x':maxx,'max_y':maxy,'cell_size':cell_size_m,'n_cols':n_cols,'n_rows':n_rows}
    print(f"Projection successful. Area: {total_area_m2 / 1e6:.2f} km^2")
    print(f"Bounds (m): minX={minx:.1f}, minY={miny:.1f}, maxX={maxx:.1f}, maxY={maxy:.1f}")
    print(f"Grid Dimensions: {n_cols} cols (X) x {n_rows} rows (Y)")
    from shapely.prepared import prep; prepared = prep(montreal_polygon)
    print("Using prepared geometry.")
    return montreal_polygon, prepared, total_area_m2, grid_info

def get_cell_index(x, y, grid_info):
    if not (grid_info['min_x'] <= x <= grid_info['max_x'] and grid_info['min_y'] <= y <= grid_info['max_y']):
       if not (abs(x - grid_info['max_x']) < 1e-9 or abs(y - grid_info['max_y']) < 1e-9): return None
    col = int((x - grid_info['min_x']) / grid_info['cell_size']); row = int((y - grid_info['min_y']) / grid_info['cell_size'])
    col = max(0, min(col, grid_info['n_cols'] - 1)); row = max(0, min(row, grid_info['n_rows'] - 1))
    return row, col

def create_dummy_density_grid(grid_info):
    print("WARNING: Generating DUMMY population density grid (center peak).")
    n_rows, n_cols = grid_info['n_rows'], grid_info['n_cols']; center_r, center_c = n_rows / 2, n_cols / 2
    max_dist = math.sqrt(center_r**2 + center_c**2); dummy_grid = np.zeros((n_rows, n_cols))
    if max_dist > 0:
        r_idx, c_idx = np.indices((n_rows, n_cols)); dist = np.sqrt((r_idx - center_r)**2 + (c_idx - center_c)**2)
        dummy_grid = np.maximum(0, 1.0 - (dist / max_dist)) * 10000
    print(f"Dummy density grid created with max value: {dummy_grid.max():.0f}")
    return dummy_grid

class EventGenerator:
    def __init__(self, category_rates, radius_ranges, event_duration_sec, placement_rule,
                 population_density_grid, grid_info, polygon_checker, min_separation):
        self.category_rates=category_rates; self.radius_ranges=radius_ranges; self.num_categories=len(category_rates)
        self.event_duration_sec=event_duration_sec; self.placement_rule=placement_rule; self.population_density_grid=population_density_grid
        self.grid_info=grid_info; self.polygon_checker=polygon_checker; self.min_separation=min_separation; self.min_separation_sq=min_separation**2
        self.accumulators={i: 0.0 for i in range(self.num_categories)}; self.min_x, self.min_y = grid_info['min_x'], grid_info['min_y']
        self.max_x, self.max_y = grid_info['max_x'], grid_info['max_y']
        self.max_density = np.max(population_density_grid) if placement_rule == 'PopulationWeighted' and population_density_grid is not None else 1.0
        if self.max_density <=0 and placement_rule == 'PopulationWeighted': print("Warning: Max population density is zero. Forcing Uniform."); self.placement_rule='Uniform'
        self.placement_attempts_per_event = 200; self.category_names = list(category_rates.keys())
    def _get_cell_density(self, r, c):
         if self.population_density_grid is not None and 0 <= r < self.grid_info['n_rows'] and 0 <= c < self.grid_info['n_cols']: return self.population_density_grid[r, c]; return 0
    def _check_separation(self, x_cand, y_cand, current_active_events):
        if self.min_separation <= 0: return True
        for _, _, ex, ey, _, _ in current_active_events:
             dist_sq = (x_cand - ex)**2 + (y_cand - ey)**2
             if dist_sq < self.min_separation_sq: return False
        return True
    def _place_event(self, current_active_events):
        placement_method = self._place_event_population_weighted if self.placement_rule == 'PopulationWeighted' else self._place_event_uniformly
        for _ in range(self.placement_attempts_per_event):
            location = placement_method()
            if location: x, y = location;
            if self._check_separation(x, y, current_active_events): return x, y
        return None
    def _place_event_uniformly(self):
         for _ in range(100): x = random.uniform(self.min_x, self.max_x); y = random.uniform(self.min_y, self.max_y);
         if self.polygon_checker(Point(x, y)): return x, y; return None
    def _place_event_population_weighted(self):
        if self.population_density_grid is None or self.max_density <= 0: return self._place_event_uniformly()
        for _ in range(100):
            x_cand=random.uniform(self.min_x, self.max_x); y_cand=random.uniform(self.min_y, self.max_y)
            idx=get_cell_index(x_cand, y_cand, self.grid_info)
            if idx:
                r, c = idx; cell_density = self._get_cell_density(r, c);
                if cell_density <= 0: continue
                # --- FIX: Added this missing line back ---
                u = random.uniform(0, self.max_density)
                # --- End Fix ---
                if u < cell_density and self.polygon_checker(Point(x_cand, y_cand)): return x_cand, y_cand
        return None
    def generate(self, dt, current_time, current_active_events):
        global event_id_counter; new_events=[]; category_counts_to_gen={}
        for i, (cat_name, rate_c) in enumerate(self.category_rates.items()):
            self.accumulators[i] += rate_c*dt; num_for_cat = int(self.accumulators[i])
            if num_for_cat > 0: category_counts_to_gen[i]=num_for_cat; self.accumulators[i] -= num_for_cat
        for cat_idx, num_gen in category_counts_to_gen.items():
            cat_name=self.category_names[cat_idx]; radius_min, radius_max = self.radius_ranges[cat_name]
            for _ in range(num_gen):
                 location = self._place_event(current_active_events + new_events)
                 if location: x, y = location; radius = random.uniform(radius_min, radius_max); event_id_counter += 1
                 expiry = current_time + self.event_duration_sec; new_events.append([event_id_counter, cat_idx, x, y, expiry, radius])
        return new_events

class StatsCollector:
    def __init__(self, grid_info, num_categories, warmup_sec, sampling_interval_sec):
        self.grid_info=grid_info; self.num_categories=num_categories; self.warmup_sec=warmup_sec; self.sampling_interval_sec=sampling_interval_sec
        self.presence_counts=np.zeros((num_categories, grid_info['n_rows'], grid_info['n_cols']), dtype=np.uint32)
        self.total_counts=np.zeros((num_categories, grid_info['n_rows'], grid_info['n_cols']), dtype=np.uint64)
        self.total_snapshots=0; self.next_sample_time = sampling_interval_sec if warmup_sec == 0 else warmup_sec
    def sample(self, current_time, active_events, dt):
        sampled_this_step = False
        if current_time >= self.warmup_sec and current_time >= self.next_sample_time - dt / 2:
            self.total_snapshots += 1; sampled_this_step = True
            current_counts_snap = np.zeros((self.num_categories, self.grid_info['n_rows'], self.grid_info['n_cols']), dtype=np.uint16)
            for _, cat_idx, x, y, _, _ in active_events:
                idx = get_cell_index(x, y, self.grid_info);
                if idx: r, c = idx; current_counts_snap[cat_idx, r, c] += 1
            self.presence_counts += (current_counts_snap >= 1).astype(np.uint32); self.total_counts += current_counts_snap.astype(np.uint64)
            while self.next_sample_time <= current_time + dt / 2: self.next_sample_time += self.sampling_interval_sec
        return sampled_this_step
    def get_results(self):
         if self.total_snapshots <= 0: empty_shape = (self.num_categories, self.grid_info['n_rows'], self.grid_info['n_cols']); return np.zeros(empty_shape), np.zeros(empty_shape), 0
         with np.errstate(divide='ignore', invalid='ignore'): probability_grid=self.presence_counts.astype(np.float64)/self.total_snapshots; expected_value_grid=self.total_counts.astype(np.float64)/self.total_snapshots
         return np.nan_to_num(probability_grid), np.nan_to_num(expected_value_grid), self.total_snapshots

class RealTimeVisualizer:
    def __init__(self, grid_info, categories, category_colors_array, plotting_params):
        if not MATPLOTLIB_AVAILABLE: raise ImportError("Matplotlib required"); self.grid_info=grid_info; self.categories=categories
        self.num_categories=len(categories); self.category_colors_array=category_colors_array; self.params=plotting_params
        self.figures={}; self.axes={}; self.image_mappables={}; self.colorbars={}; self.legend_handles=[]
        self.last_plot_update_wall_time = time.time(); self._setup_plots()
    def _setup_plots(self):
        plt.ion(); fig1, ax1 = plt.subplots(figsize=(8, 7)); fig1.canvas.manager.set_window_title('Active Events'); fig1.patch.set_facecolor(self.params['PLOT1_BG_COLOR'])
        ax1.set_facecolor(self.params['PLOT1_BG_COLOR']); ax1.set_title("Live Active Events"); ax1.set_xlabel("Easting (m)"); ax1.set_ylabel("Northing (m)")
        ax1.set_xlim(self.grid_info['min_x'], self.grid_info['max_x']); ax1.set_ylim(self.grid_info['min_y'], self.grid_info['max_y'])
        ax1.set_aspect('equal', adjustable='box')
        for i, cat in enumerate(self.categories): line, = ax1.plot([], [], '.', color=self.category_colors_array[i], label=cat, markersize=self.params['PLOT1_MARKERSIZE']*2); self.legend_handles.append(line)
        ax1.legend(handles=self.legend_handles, loc='upper right', fontsize='small'); fig1.tight_layout(); self.figures['active']=fig1; self.axes['active']=ax1
        fig2, axes2 = plt.subplots(nrows=1, ncols=self.num_categories, figsize=(5*self.num_categories, 5), sharex=True, sharey=True, constrained_layout=True)
        fig2.canvas.manager.set_window_title('Mean Event Count E[N]'); fig2.patch.set_facecolor(self.params['PLOT2_BG_COLOR']); fig2.suptitle("Estimated Mean Event Count E[N] per Cell"); images_mean = []
        extent = [self.grid_info['min_x'], self.grid_info['min_x'] + self.grid_info['n_cols']*self.grid_info['cell_size'], self.grid_info['min_y'], self.grid_info['min_y'] + self.grid_info['n_rows']*self.grid_info['cell_size']]
        for i, cat in enumerate(self.categories):
            ax = axes2[i] if self.num_categories > 1 else axes2; ax.set_facecolor(self.params['PLOT2_BG_COLOR'])
            im = ax.imshow(np.zeros((self.grid_info['n_rows'], self.grid_info['n_cols'])), origin='lower', extent=extent, cmap=self.params['CMAP_DENSITY'], aspect='equal', vmin=0)
            ax.set_title(cat); ax.set_xlabel("Easting (m)"); images_mean.append(im);
            if i == 0: ax.set_ylabel("Northing (m)")
        cbar_mean = fig2.colorbar(images_mean[-1], ax=axes2, orientation='vertical', pad=0.02); cbar_mean.set_label('Mean Events / Cell')
        self.figures['mean']=fig2; self.axes['mean']=axes2; self.image_mappables['mean']=images_mean; self.colorbars['mean']=cbar_mean
        fig3, axes3 = plt.subplots(nrows=1, ncols=self.num_categories, figsize=(5*self.num_categories, 5), sharex=True, sharey=True, constrained_layout=True)
        fig3.canvas.manager.set_window_title('Event Probability P(N>=1)'); fig3.patch.set_facecolor(self.params['PLOT3_BG_COLOR']); fig3.suptitle("Estimated Probability P(N>=1) per Cell"); images_prob = []
        for i, cat in enumerate(self.categories):
            ax = axes3[i] if self.num_categories > 1 else axes3; ax.set_facecolor(self.params['PLOT3_BG_COLOR'])
            im = ax.imshow(np.zeros((self.grid_info['n_rows'], self.grid_info['n_cols'])), origin='lower', extent=extent, cmap=self.params['PLOT3_CMAP'], aspect='equal', vmin=0, vmax=1.0)
            ax.set_title(cat); ax.set_xlabel("Easting (m)"); images_prob.append(im)
            if i == 0: ax.set_ylabel("Northing (m)")
        cbar_prob = fig3.colorbar(images_prob[-1], ax=axes3, orientation='vertical', pad=0.02); cbar_prob.set_label('Probability')
        self.figures['prob']=fig3; self.axes['prob']=axes3; self.image_mappables['prob']=images_prob; self.colorbars['prob']=cbar_prob
        plt.show(block=False); plt.pause(0.1)
    def update(self, active_events_list, mean_count_grid, prob_grid, current_sim_time, num_snapshots):
        now_wall_clock = time.time()
        if now_wall_clock - self.last_plot_update_wall_time < self.params['PLOT_UPDATE_INTERVAL_SEC']: return
        self.last_plot_update_wall_time = now_wall_clock
        ax1 = self.axes['active']; current_lines = ax1.get_lines(); num_legend_lines = len(self.categories)
        for line in current_lines[num_legend_lines:]: line.remove()
        events_by_cat = [[] for _ in range(self.num_categories)]; new_labels = []
        for _, cat_idx, x, y, _, _ in active_events_list: events_by_cat[cat_idx].append((x, y))
        for i, cat_events in enumerate(events_by_cat):
            if cat_events: x_coords, y_coords = zip(*cat_events); ax1.plot(x_coords, y_coords, '.', color=self.category_colors_array[i], markersize=self.params['PLOT1_MARKERSIZE'])
            new_labels.append(f"{self.categories[i]} ({len(cat_events)})")
        ax1.legend(handles=self.legend_handles, labels=new_labels, loc='upper right', fontsize='small')
        ax1.set_title(f"Live Active Events (T={current_sim_time:.1f}s, N={len(active_events_list)})")
        max_mean_overall = 0.1
        if mean_count_grid is not None and mean_count_grid.size > 0:
            all_mean_values = mean_count_grid[mean_count_grid > 0];
            if all_mean_values.size > 0: max_mean_overall = max(0.1, np.percentile(all_mean_values, self.params['VMAX_PERCENTILE']))
        for i in range(self.num_categories): im = self.image_mappables['mean'][i]; im.set_data(mean_count_grid[i]); im.set_clim(0, max_mean_overall)
        if num_snapshots > 0 and self.colorbars['mean'].mappable.get_array().size > 0 :
             try: self.colorbars['mean'].mappable.set_clim(0, max_mean_overall)
             except Exception as e_cbar: print(f"\rMinor issue updating mean colorbar limits: {e_cbar}", end='')
        for i in range(self.num_categories): im = self.image_mappables['prob'][i]; im.set_data(prob_grid[i])
        try:
            for fig in self.figures.values(): fig.canvas.draw_idle(); plt.pause(0.01)
            print(f"  Plot Update @ Sim Time: {current_sim_time:.1f}s | Active: {len(active_events_list)} | Snaps: {num_snapshots}   ", end='\r')
        except Exception as e: print(f"\nWarning: Error during plot update - {e}")
    def close(self):
        if MATPLOTLIB_AVAILABLE: plt.ioff(); plt.close('all')

def run_realtime_mode(polygon, prepared, grid_info, warmup_sec):
    global active_events
    plotting_params_dict = {'PLOT1_BG_COLOR':PLOT1_BG_COLOR,'PLOT2_BG_COLOR':PLOT2_BG_COLOR,'PLOT3_BG_COLOR':PLOT3_BG_COLOR,
                            'PLOT3_CMAP':PLOT3_CMAP,'PLOT1_MARKERSIZE':PLOT1_MARKERSIZE,'VMAX_PERCENTILE':VMAX_PERCENTILE,
                            'DENSITY_SMOOTHING_SIGMA':DENSITY_SMOOTHING_SIGMA,'CMAP_DENSITY':CMAP_DENSITY_RT,
                            'CMAP_PROBABILITY':CMAP_PROBABILITY,'PLOT_UPDATE_INTERVAL_SEC':PLOT_UPDATE_INTERVAL_SEC }
    density_grid = create_dummy_density_grid(grid_info)
    event_generator = EventGenerator(CATEGORY_RATES, EVENT_RADIUS_RANGES_METERS, EVENT_DURATION_SEC, PLACEMENT_RULE,
                                     density_grid, grid_info, prepared.contains, MINIMUM_SEPARATION_METERS)
    stats_collector = StatsCollector(grid_info, NUM_CATEGORIES, warmup_sec, SAMPLING_INTERVAL_SEC)
    visualizer = RealTimeVisualizer(grid_info, CATEGORIES, CATEGORY_COLORS_ARRAY, plotting_params_dict)
    current_time = 0.0
    print("\n--- Starting Real-Time Simulation Loop ---")
    print(f"NOTICE: Using separation distance: {MINIMUM_SEPARATION_METERS}m. PERFORMANCE WILL BE POOR.")
    print("Press Ctrl+C in terminal to stop.")
    try:
        while True:
            loop_start_time = time.time()
            active_events = [e for e in active_events if current_time < e[4]]
            newly_generated = event_generator.generate(TIME_STEP_SEC, current_time, active_events)
            active_events.extend(newly_generated)
            stats_collector.sample(current_time, active_events, TIME_STEP_SEC)
            prob_grid, mean_grid, num_snaps = stats_collector.get_results()
            visualizer.update(active_events, mean_grid, prob_grid, current_time, num_snaps)
            current_time += TIME_STEP_SEC
            loop_end_time = time.time(); wall_time_elapsed = loop_end_time - loop_start_time
            sleep_time = max(0, TIME_STEP_SEC - wall_time_elapsed)
            time.sleep(min(0.02, sleep_time))
    except KeyboardInterrupt: print("\nSimulation stopped by user.")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()
    finally: print("\nCleaning up..."); visualizer.close(); print("Simulation Finished.")

def run_single_simulation(run_id, simulation_duration_sec, warmup_sec, sampling_interval_sec, dt_sec,
                          montreal_polygon_proj, prepared_polygon, grid_info, category_rates,
                          radius_ranges, event_duration_s, placement_rule, population_density_grid, min_separation):
    global active_events, event_id_counter
    active_events = [] ; event_id_counter = 0; current_time = 0.0
    start_run_time = time.time()
    print(f"--- Starting Replication {run_id} (Duration: {simulation_duration_sec}s) ---")
    event_generator = EventGenerator(category_rates, radius_ranges, event_duration_s, placement_rule,
                                     population_density_grid, grid_info, prepared_polygon.contains, min_separation)
    stats_collector = StatsCollector(grid_info, len(category_rates), warmup_sec, sampling_interval_sec)
    while current_time < simulation_duration_sec:
        active_events = [e for e in active_events if current_time < e[4]]
        newly_generated = event_generator.generate(dt_sec, current_time, active_events)
        active_events.extend(newly_generated)
        stats_collector.sample(current_time, active_events, dt_sec)
        current_time += dt_sec
        if int(current_time*10) % (max(10, int(simulation_duration_sec/10))*10) == 0: print(f"  Rep {run_id} - Time: {current_time:.0f}s / {simulation_duration_sec}s", end='\r')
    probability_grid, expected_value_grid, total_snapshots = stats_collector.get_results()
    end_run_time = time.time()
    print(f"\n--- Finished Replication {run_id} | Wall Time: {end_run_time - start_run_time:.2f}s | Snapshots: {total_snapshots} ---")
    return probability_grid, expected_value_grid

def plot_replication_results_plotly(data_grid, category_names, grid_info, cell_size_m, title_prefix, colorbar_title, output_file, cmap='Viridis'):
    if not PLOTLY_AVAILABLE or data_grid is None or data_grid.size == 0 or np.all(np.isnan(data_grid)): print(f"\nSkipping Plotly heatmap for '{title_prefix}'."); return
    min_x, min_y = grid_info['min_x'], grid_info['min_y']; n_cols, n_rows = grid_info['n_cols'], grid_info['n_rows']
    num_categories = len(category_names)
    x_edges = np.linspace(min_x, min_x + n_cols * cell_size_m, n_cols + 1); y_edges = np.linspace(min_y, min_y + n_rows * cell_size_m, n_rows + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2; y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    ncols_subplot = int(math.ceil(math.sqrt(num_categories))); nrows_subplot = int(math.ceil(num_categories / ncols_subplot))
    fig = make_subplots(rows=nrows_subplot, cols=ncols_subplot, subplot_titles=category_names)
    print(f"\nGenerating Plotly heatmap for {title_prefix}...")
    max_z_value = np.nanmax(data_grid) if np.any(np.isfinite(data_grid)) else 1.0;
    if max_z_value <= 0: max_z_value = 0.1
    for i, category in enumerate(category_names):
        row = i // ncols_subplot + 1; col = i % ncols_subplot + 1
        subplot_id = ((row-1)*ncols_subplot) + col; xaxis_name = f'x{subplot_id}' if subplot_id > 1 else 'x'
        fig.add_trace(go.Heatmap(z=data_grid[i].T, x=x_centers, y=y_centers, colorscale=cmap, zmin=0, zmax=max_z_value,
                                 showscale=(i == num_categories - 1), colorbar=dict(title=colorbar_title) if (i == num_categories - 1) else None,
                                 name=category, hoverongaps=False, xaxis=f'x{subplot_id}', yaxis=f'y{subplot_id}'
                               ), row=row, col=col)
        fig.update_xaxes(title_text="Easting (m)", row=row, col=col); fig.update_yaxes(title_text="Northing (m)", scaleanchor=xaxis_name, scaleratio=1, row=row, col=col)
    fig.update_layout(title_text=f'MC Replication Results: {title_prefix} per {cell_size_m}m Cell', height=max(450, 350 * nrows_subplot), width=max(700, 450 * ncols_subplot), hovermode='closest', margin=dict(l=50,r=50,t=100,b=50))
    try: fig.write_html(output_file); print(f"Interactive heatmap saved to: {output_file}")
    except Exception as e: print(f"Error saving Plotly HTML file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo Event Simulation")
    parser.add_argument('--mode', type=str, default='replicate', choices=['replicate', 'realtime'], help="Operation mode")
    parser.add_argument('--reps', type=int, default=NUM_REPLICATIONS, help="Number of replications ('replicate' mode)")
    parser.add_argument('--duration', type=int, default=SIMULATION_DURATION_PER_REP, help="Duration per replication (sec) ('replicate' mode)")
    args = parser.parse_args()

    if args.mode == 'realtime' and not MATPLOTLIB_AVAILABLE: print("Error: Matplotlib & Scipy required for 'realtime' mode."); sys.exit(1)
    if args.mode == 'replicate' and not PLOTLY_AVAILABLE: print("Warning: Plotly not found. Replication plots disabled.")
    if not GEOPANDAS_AVAILABLE: sys.exit(1)

    try:
        print("="*60); print(f"Mode: {args.mode}"); print("="*60)
        print("WARNING: Using DUMMY population density grid.")
        print("WARNING: Naive O(N) separation check active - expect slow performance.")
        print(f"         Requires boundary file: '{MONTREAL_BOUNDARY_FILE}'"); print("="*60)
        poly, prep, total_area, grid_info = load_area_and_setup_grid(MONTREAL_BOUNDARY_FILE, TARGET_CRS, CELL_SIZE_METERS)
        density_grid = create_dummy_density_grid(grid_info)

        if args.mode == 'replicate':
             all_prob_grids = []; all_exp_grids = []
             print(f"\n--- Starting Replication Mode: {args.reps} runs x {args.duration}s each ---")
             for i in range(args.reps):
                 run_id = i + 1; random.seed(run_id); np.random.seed(run_id)
                 prob_grid, exp_grid = run_single_simulation(
                     run_id=run_id, simulation_duration_sec=args.duration, warmup_sec=WARMUP_DURATION_SEC,
                     sampling_interval_sec=SAMPLING_INTERVAL_SEC, dt_sec=TIME_STEP_SEC, montreal_polygon_proj=poly,
                     prepared_polygon=prep, grid_info=grid_info, category_rates=CATEGORY_RATES,
                     radius_ranges=EVENT_RADIUS_RANGES_METERS, event_duration_s=EVENT_DURATION_SEC, # Pass correct global name
                     placement_rule=PLACEMENT_RULE, population_density_grid=density_grid, min_separation=MINIMUM_SEPARATION_METERS
                 )
                 if prob_grid is not None and exp_grid is not None: all_prob_grids.append(prob_grid); all_exp_grids.append(exp_grid)
                 else: print(f"Warning: Replication {run_id} discarded.")

             if not all_prob_grids or not all_exp_grids: print("\nNo valid replication results collected.")
             else:
                 print("\n--- Analyzing Results Across Replications ---")
                 stacked_probs = np.stack(all_prob_grids); stacked_exps = np.stack(all_exp_grids)
                 mean_probs = np.mean(stacked_probs, axis=0); std_dev_probs = np.std(stacked_probs, axis=0)
                 mean_exps = np.mean(stacked_exps, axis=0); std_dev_exps = np.std(stacked_exps, axis=0)
                 summary_data = []; headers = ["Category", "Mean P(N>=1)", "StdDev P", "Mean E[N]", "StdDev E"]
                 for i, cat_name in enumerate(CATEGORIES):
                      avg_mp = np.mean(mean_probs[i]); avg_sp = np.mean(std_dev_probs[i])
                      avg_me = np.mean(mean_exps[i]); avg_se = np.mean(std_dev_exps[i])
                      summary_data.append([cat_name, f"{avg_mp:.5f}", f"{avg_sp:.5f}", f"{avg_me:.5f}", f"{avg_se:.5f}"])
                 if TABLES_AVAILABLE: print("\nSummary Table (Mean +/- Mean StdDev across cells):"); print(tabulate(summary_data, headers=headers, tablefmt="grid"))
                 else: print("\nSummary Stats (Mean +/- Mean StdDev across cells):"); [print(f" {r[0]}: P={r[1]}±{r[2]} | E={r[3]}±{r[4]}") for r in summary_data]

                 plot_replication_results_plotly(mean_probs, CATEGORIES, grid_info, CELL_SIZE_METERS, "Mean Probability P(N>=1)", "Mean P(N≥1)", MEAN_PROB_PLOT_HTML_FILE, cmap=CMAP_PROBABILITY)
                 plot_replication_results_plotly(std_dev_probs, CATEGORIES, grid_info, CELL_SIZE_METERS, "Std Dev of Probability P(N>=1)", "Std Dev P(N≥1)", STD_PROB_PLOT_HTML_FILE, cmap=CMAP_STDDEV)
                 plot_replication_results_plotly(mean_exps, CATEGORIES, grid_info, CELL_SIZE_METERS, "Mean Expected Count E[N]", "Mean E[N]", MEAN_EXP_PLOT_HTML_FILE, cmap=CMAP_MEAN)
                 plot_replication_results_plotly(std_dev_exps, CATEGORIES, grid_info, CELL_SIZE_METERS, "Std Dev of Expected Count E[N]", "Std Dev E[N]", STD_EXP_PLOT_HTML_FILE, cmap=CMAP_STDDEV)

        elif args.mode == 'realtime':
             run_realtime_mode(poly, prep, grid_info, WARMUP_DURATION_SEC)

    except FileNotFoundError as e: print(f"Error: {e}\nPlease ensure the boundary file '{MONTREAL_BOUNDARY_FILE}' exists.")
    except Exception as e: print(f"\nAn unexpected error occurred in main execution: {e}"); traceback.print_exc()

    print("\n--- Script Finished ---")
