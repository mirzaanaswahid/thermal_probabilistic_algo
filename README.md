# Thermal Probabilistic Algorithm

This repository contains research and implementation of thermal probabilistic algorithms for UAV (Unmanned Aerial Vehicle) operations in Montreal, Canada.

## Project Overview

This project focuses on developing and implementing probabilistic algorithms for thermal-aware UAV operations, including:

- Thermal data generation and analysis
- Probabilistic event simulation
- Montreal region mapping and path planning
- UAV fleet management and optimization
- Performance metrics and analysis

## Repository Structure

```
montreal_research/
├── event/                          # Event simulation modules
│   ├── monte_carlo_event_sim_real_time_v6_sizes.py
│   ├── monte_carlo_event_sim_real_time_v7_sizes.py
│   └── monte_carlo_event_sim_real_time_v8_sizes.py
├── montreal_map/                   # Montreal mapping utilities
│   ├── adj_regions.py
│   └── adj_regions_path.py
├── montreal_path/                  # Path planning algorithms
│   ├── all_path.py
│   ├── all_path_v3.0.py
│   └── all_pathv2.0.py
├── thermal_synthetic_data_generation/  # Thermal data generation
│   ├── figures/
│   ├── montreal_government_data/
│   ├── notebook/
│   └── synthetic_generated_data/
├── research_report/                # Research documentation
│   ├── Distributed Strategy.pdf
│   └── Thermal_Research.pdf
├── results/                        # Generated results and figures
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Key Features

- **Thermal Data Generation**: Synthetic thermal data generation based on Montreal land cover data
- **Probabilistic Event Simulation**: Monte Carlo simulations for real-time event processing
- **Path Planning**: Advanced algorithms for UAV path optimization in Montreal regions
- **Performance Analysis**: Comprehensive metrics and visualization tools
- **Research Documentation**: Detailed research reports and findings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thermal_probabilistic_algo.git
cd thermal_probabilistic_algo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Thermal Data Generation

Navigate to the thermal data generation module:
```bash
cd thermal_synthetic_data_generation/notebook
jupyter notebook
```

### Event Simulation

Run Monte Carlo event simulations:
```bash
cd event
python monte_carlo_event_sim_real_time_v8_sizes.py
```

### Path Planning

Execute path planning algorithms:
```bash
cd montreal_path
python all_path_v3.0.py
```

## Dependencies

Key dependencies include:
- numpy
- pandas
- matplotlib
- geopandas
- plotly
- jupyter
- scipy

See `requirements.txt` for the complete list.

## Research Areas

1. **Thermal Probabilistic Modeling**: Development of probabilistic models for thermal data analysis
2. **UAV Fleet Optimization**: Algorithms for optimal UAV fleet deployment and management
3. **Real-time Event Processing**: Monte Carlo simulations for real-time event handling
4. **Geospatial Analysis**: Montreal region mapping and spatial analysis
5. **Performance Metrics**: Comprehensive evaluation of algorithm performance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub.

## Acknowledgments

- Montreal government data sources
- Research collaborators and advisors
- Open source community contributors 