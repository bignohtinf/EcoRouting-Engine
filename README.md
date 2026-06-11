# EcoRouting Engine

A complete implementation of mathematical models, optimization algorithms, and AI for the full operational pipeline of electric vehicle fleets: from order batching and delivery routing, dynamic vehicle routing, driver coordination, EV-specific energy optimization, passenger ride-hailing, to multi-agent cooperative reinforcement learning.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Details](#module-details)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Technical Documentation](#technical-documentation)
7. [Algorithm Summary](#algorithm-summary)
8. [References](#references)

---

## Architecture Overview

The system is designed as a **Hierarchical Optimization Pipeline** with three decision layers:

| Layer | Scope | Update Frequency | Corresponding Modules |
|-------|-------|-------------------|-----------------------|
| **Strategic** | Charging network planning, fleet allocation, heterogeneous vehicle assignment | Daily / Weekly | EV-Specific |
| **Tactical** | Demand forecasting, driver coordination, surge pricing, SoC management | Hourly | DriverCoordination, Ride-hailing |
| **Operational** | Order batching, routing, dispatch, real-time charging scheduling | Milliseconds to Seconds | OrderBatching, DynamicVRP, Multi-AgentRL |


## Module Details

### 1. OrderBatchingPooling --- Order Batching & Delivery Routing

Solves the problem: **Group multiple orders with similar characteristics (location, time) into a single driver assignment** to minimize total transportation cost.

**Core Algorithms:**
- **ST-DBSCAN (Spatiotemporal DBSCAN):** Clusters orders by spatial proximity (Haversine distance) and temporal proximity of request times.
- **Greedy Insertion Router:** Constructs optimized Pickup-Delivery routes for each batch using feasibility-aware greedy insertion.
- **MILP McCormick:** Exact optimization model that linearizes the quadratic objective using McCormick envelopes: `w_ijk <= y_ik, w_ijk <= y_jk, w_ijk >= y_ik + y_jk - 1`.

**Constraints:**
- Vehicle capacity
- Time windows
- Maximum batch size

**Output:** Comparative report of MILP vs Heuristic solvers, route visualization saved as `route_visualization.png`.

---

### 2. DynamicVRP --- Dynamic Vehicle Routing

Solves the problem: **Real-time vehicle routing** under continuously arriving orders and changing traffic conditions.

**Core Algorithms:**
- **VRPTW MILP:** Mixed-Integer Linear Programming with Miller-Tucker-Zemlin (MTZ) subtour elimination constraints and Big-M time window enforcement.
- **PPO Actor-Critic:** Reinforcement learning with Generalized Advantage Estimator (GAE), truncation horizon, and variance reduction via advantage normalization.
- **ST-GNN ETA Predictor:** Travel time prediction using Spatio-Temporal Graph Neural Network message passing combined with multi-quantile loss (q = 0.1, 0.5, 0.9) and Huber Loss.

**Output:** Step-by-step routing logs, multi-quantile ETA predictions, trip map saved as `dynamic_vrp_plot.png`.

---

### 3. DriverCoordinationandOptimalMatching --- Driver Coordination & Optimal Matching

Solves the problem: **Select the best driver from hundreds of available drivers for each order batch**, ensuring income fairness and demand coverage.

**Core Algorithms:**
- **Hungarian (Kuhn-Munkres):** Classic bipartite matching with O(n^3) complexity, supporting dummy nodes for imbalanced supply-demand cases.
- **Auction Algorithm (Bertsekas):** Auction-based matching with O(n^2 log n) complexity, highly parallelizable for large-scale systems (n > 1000 drivers).
- **Gini Coefficient & Lexicographic Max-Min:** Measures and optimizes income fairness across the driver fleet.
- **Time-Expanded Transportation LP:** Proactive driver repositioning based on demand forecasts (ConvLSTM + GAT).
- **DRO Wasserstein & CVaR:** Robust optimization replacing the unrealistic Gaussian assumption for travel time distributions with distributionally robust and risk-averse formulations.
- **Crew Scheduling:** Maximum driving time constraints (8h/day), mandatory rest after 4 consecutive hours.

**Output:** Speed comparison of Hungarian vs Auction solvers, Gini coefficient before/after fairness adjustment, repositioning flow results, matching visualization saved as `matching_results.png`.

---

### 4. EV-Specific --- Electric Vehicle Optimization

Solves the problem: **Route electric vehicles under energy constraints**, optimize charging/swapping schedules, and plan the V-Green charging network.

**Core Algorithms:**
- **Energy Consumption Model:** Nonlinear model incorporating aerodynamic drag, elevation gradient, payload weight, and HVAC load. Parameters calibrated from VinFast telemetry data.
- **E-VRP-CS:** Extends VRPTW with State-of-Charge (SoC) variables, automatically inserting charging station visits when battery drops below 15%.
- **CC-CV Charging + SOS2:** Simulates nonlinear charging curves (Constant Current - Constant Voltage protocol) using piecewise linear approximation with SOS2 constraints.
- **M/M/c Erlang C:** Queuing model for charging stations, computing expected wait times and service probabilities.
- **Facility Location MILP:** Strategic planning of new charging station locations and capacities (p-Median, Capacitated Facility Location).
- **Battery Swap Station Location (BSSLP):** Optimizes battery swap station placement combined with Newsvendor inventory management for battery stock.
- **Vehicle-to-Grid (V2G):** Optimizes reverse power discharge schedules for idle vehicles, generating supplementary revenue.
- **CVaR-PPO:** Risk-averse reinforcement learning for energy safety, preventing aggressive policies that risk battery depletion.

**Output:** Energy consumption statistics, optimized charging schedules, Erlang C queue distributions, route map saved as `ev_route_map.png` and SoC profile saved as `ev_soc_profile.png`.

---

### 5. Ride-hailing --- Passenger Transport Optimization

Solves the problem: **Routing, ride-pooling, and dynamic pricing for passenger ride-hailing services**, with passenger-specific characteristics fundamentally different from delivery logistics.

**Core Algorithms:**
- **DARP MILP (Dial-a-Ride Problem):** Passenger routing with Precedence constraints (pickup before dropoff), Pairing, Maximum Ride Time, and Lexicographic VIP tier prioritization (Standard, Plus, Premium).
- **Shareability Network:** Trip compatibility graph following Santi et al. (PNAS 2014), solved via Maximum Weighted Matching.
- **RTV-graph:** High-capacity ride-sharing model from Alonso-Mora et al. (PNAS 2017), matching 2-4 passengers simultaneously.
- **Cancellation Prediction:** Classification model predicting cancellation rates (8-15% in Vietnam), integrating expected cancellation cost into the assignment cost matrix: `c_adj = (1 - p_cancel) * c + p_cancel * C_wasted`.
- **Detour Constraint:** Passenger detour acceptance threshold (10-25% over direct travel), modeled via Discrete Choice combined with discount incentives.
- **Surge Pricing:** Spatio-temporal surge multiplier for real-time supply-demand balancing.

**Delivery vs Ride-hailing Comparison:**

| Attribute | Delivery | Ride-hailing |
|-----------|----------|--------------|
| Batching | OBP, ST-DBSCAN | Ride-pooling, Shareability Network |
| Routing | VRPTW, DVRP | DARP, RTV-graph |
| Time window | 20-30 minutes | 3-10 minutes (high cancellation risk) |
| Capacity | Volume / Weight | Seats + Comfort |
| Cancellation rate | Low (< 5%) | High (8-15%) |
| Pricing | Fixed / Zone-based | Real-time surge |
| Detour | Not critical | Hard / Soft constraint |

**Output:** Surge coefficients, ride-pooling maps, DARP schedules, trip map saved as `ride_hailing_darp_map.png` and shareability graph saved as `shareability_network.png`.

---

### 6. Multi-AgentRL --- Multi-Agent Reinforcement Learning

Solves the problem: **Global coordination of multiple electric vehicles simultaneously** in a decentralized cooperative environment.

**Core Algorithms:**
- **QPLEX (ICLR 2021):** Dueling Bipartite Value Factorisation, solving the IGM (Individual-Global-Max) principle for CTDE (Centralized Training, Decentralized Execution).
- **QMIX:** Monotonic mixing network for value decomposition.
- **Prioritized Experience Replay (PER):** Priority-weighted experience buffer with Importance Sampling bias correction.

**Output:** Training logs, convergence plots saved as `qplex_cooperative_reward.png` and `qplex_advantage_factorization.png`.

---

## System Requirements

- **Python:** >= 3.8
- **Required packages:**
  - `numpy` --- Numerical computation
  - `scipy` --- Optimization (MILP solver, linear programming)
  - `matplotlib` --- Visualization (charts and maps)

All modules are implemented **without deep learning framework dependencies** (PyTorch, TensorFlow). RL and GNN models are built with pure numpy for lightweight portability across all environments.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/bignohtinf/EcoRouting-Engine.git
cd EcoRouting-Engine

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Each module operates independently. Navigate to the module directory and run `main.py`:

```bash
# 1. Order Batching & Delivery Routing
cd OrderBatchingPooling
python main.py

# 2. Dynamic Vehicle Routing
cd DynamicVRP
python main.py

# 3. Driver Coordination & Optimal Matching
cd DriverCoordinationandOptimalMatching
python main.py

# 4. Electric Vehicle Optimization
cd EV-Specific
python main.py

# 5. Passenger Ride-hailing
cd Ride-hailing
python main.py

# 6. Multi-Agent Reinforcement Learning
cd Multi-AgentRL
python main.py
```

Results are printed to the terminal (tables, statistics) and saved as image files (.png) in the respective module directories.

---

## Technical Documentation

- **`optimization_vehicle_service.tex`** --- Main LaTeX document containing all mathematical formulations, proofs, and theoretical analysis across 7 sections:
  1. Optimization Pipeline Overview
  2. Order Batching & Pooling
  3. Dynamic Vehicle Routing (DVRP)
  4. Driver Coordination & Optimal Matching
  5. Electric Vehicle System Optimization (EV-Specific)
  6. Ride-hailing Passenger Optimization
  7. Multi-Agent RL (QPLEX)

---

## Algorithm Summary

| Module | Algorithm / Model | Problem Type | Complexity |
|--------|-------------------|--------------|------------|
| OrderBatching | ST-DBSCAN | Clustering | O(n log n) |
| OrderBatching | Greedy Insertion | Routing heuristic | O(n^2) |
| OrderBatching | MILP McCormick | Exact optimization | NP-hard |
| DynamicVRP | VRPTW + MTZ | Exact routing | NP-hard |
| DynamicVRP | PPO + GAE | Reinforcement Learning | Polynomial (per episode) |
| DynamicVRP | ST-GNN + Quantile Loss | ETA prediction | O(V + E) per layer |
| DriverCoordination | Hungarian | Bipartite matching | O(n^3) |
| DriverCoordination | Auction (Bertsekas) | Bipartite matching | O(n^2 log n) |
| DriverCoordination | ConvLSTM + GAT | Demand forecasting | O(G * T) |
| DriverCoordination | DRO Wasserstein | Robust optimization | Convex program |
| EV-Specific | E-VRP-CS | EV routing + charging | NP-hard |
| EV-Specific | CC-CV + SOS2 | Charging scheduling | LP with SOS2 |
| EV-Specific | M/M/c Erlang C | Queuing analysis | Closed-form |
| EV-Specific | Facility Location | Network planning | NP-hard |
| EV-Specific | BSSLP + Newsvendor | Swap station planning | NP-hard |
| EV-Specific | CVaR-PPO | Risk-averse RL | Polynomial (per episode) |
| Ride-hailing | DARP MILP | Passenger routing | NP-hard |
| Ride-hailing | Shareability Network | Ride-pooling matching | O(n^2) |
| Ride-hailing | RTV-graph | High-capacity sharing | O(n^2 * V) |
| Ride-hailing | Surge Pricing | Dynamic pricing | O(G) per update |
| Multi-AgentRL | QPLEX | Cooperative MARL | O(n * d^2) per step |
| Multi-AgentRL | PER | Experience replay | O(log N) per sample |

---

## References

- Schneider, Stickel & Goeke (2014). *The E-VRPTW with Recharging Stations*. Transportation Science.
- Cordeau & Laporte (2007). *The dial-a-ride problem*. 4OR.
- Santi et al. (2014). *Quantifying the benefits of vehicle pooling with shareability networks*. PNAS.
- Alonso-Mora et al. (2017). *On-demand high-capacity ride-sharing*. PNAS.
- Castillo, Knoepfle & Weyl (2017). *Surge pricing solves the wild goose chase*. EC.
- Mohajerin Esfahani & Kuhn (2018). *Data-driven DRO using Wasserstein metric*.
- Rockafellar & Uryasev (2000). *Optimization of conditional value-at-risk*.
- Wang et al. (2021). *QPLEX: Duplex Dueling Multi-Agent Q-Learning*. ICLR.
