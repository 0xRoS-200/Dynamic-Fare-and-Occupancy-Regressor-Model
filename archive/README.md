# PUBLIC TRANSPORTATION USAGE DATASET

> Technical overview and specifications of dataset contents and design

---

---

### Dataset Overview:

| Component       | Detail                                   |
| --------------- | ---------------------------------------- |
| **Type**        | Tabular / Time-series / Graph-structured |
| **Rows**        | 569,400                                  |
| **Columns**     | 88                                       |
| **Cities**      | 6 major global cities                    |
| **Routes**      | ~6–16 routes per city                    |
| **Time range**  | 2020-01-01 → 2020-12-30                  |
| **Granularity** | Hourly observations                      |
| **Network**     | GTFS-inspired transit graph              |

---

### Dataset Description:

A large-scale public transportation usage dataset simulating urban transit systems across multiple global cities. It models passenger flow, delays, occupancy, and system anomalies across interconnected transport networks including metro, bus, rail, tram, ferry, and light rail systems.

The dataset is designed for forecasting, anomaly detection, and graph-based learning on real-world inspired mobility systems.

---

### Cities modeled:

| City      |
| --------- |
| London    |
| New York  |
| Berlin    |
| Tokyo     |
| Paris     |
| Singapore |

---

### Transit modes:

| Mode       |
| ---------- |
| metro      |
| bus        |
| rail       |
| tram       |
| lrt        |
| ferry      |
| light_rail |

---

### Fare system types:

| Type         |
| ------------ |
| standard     |
| concession   |
| child        |
| group        |
| tourist_pass |
| monthly_pass |

---

### Graph structure:

| Component       | Description                          |
| --------------- | ------------------------------------ |
| Nodes           | stations / stops                     |
| Edges           | routes between stops                 |
| Transfer edges  | inter-line connectivity              |
| Spatial linking | geohash-based proximity edges        |
| Flow direction  | directed weighted edges              |
| Dynamic weight  | passenger volume + delay sensitivity |

---

### Transit dynamics:

| Component              | Behaviour                               |
| ---------------------- | --------------------------------------- |
| Demand cycles          | daily + weekly commuting patterns       |
| Rush hours             | strong morning + evening peaks          |
| Delay propagation      | delays cascade across connected routes  |
| Congestion effects     | occupancy saturation reduces throughput |
| Seasonal shifts        | holidays + events affect ridership      |
| Network load balancing | uneven route stress distribution        |

---

### Feature engineering:

| Group             | Features                                  |
| ----------------- | ----------------------------------------- |
| Demand            | boarding_count, occupancy_rate            |
| Delay             | delay_minutes, delay_lag features         |
| Rolling stats     | 1h–24h rolling mean/std for ridership     |
| Temporal encoding | hour, day-of-week, month cyclic features  |
| Graph signals     | node centrality, route connectivity score |
| Anomaly flags     | delay_spike_flag, occupancy_overload_flag |
| Flow metrics      | passenger_flow_rate, transfer_intensity   |

---

### ML targets:

| Target                | Task                             |
| --------------------- | -------------------------------- |
| boarding_count        | Regression — demand forecasting  |
| delay_minutes         | Regression — delay prediction    |
| occupancy_rate        | Regression — crowding estimation |
| anomaly_flag          | Binary classification            |
| cancellation_flag     | Binary classification            |
| occupancy_probability | Regression                       |

---

### Dataset implementation notes:

| Component          | Detail                                    |
| ------------------ | ----------------------------------------- |
| Simulation         | GTFS-inspired probabilistic transit model |
| Graph model        | static topology + dynamic flow weights    |
| Temporal ordering  | strict hourly chronological simulation    |
| Split strategy     | 80/10/10 time-based split                 |
| Leakage prevention | no forward temporal leakage in features   |
| Scalability        | city-wise generation + merge pipeline     |

---

### Dataset Directory:

```
.
├── data/
│ ├── full/
│ │ └── transit_full.csv.gz
│ │
│ ├── splits/
│ │ ├── transit_train.csv.gz
│ │ ├── transit_val.csv.gz
│ │ └── transit_test.csv.gz
│ │
│ ├──subsets/
│ │ ├── transit_anomaly_detection.csv.gz
│ │ ├── transit_forecasting.csv.gz
│ │ └── Cities/
│ │   ├── transit__berlin.csv.gz
│ │   ├── transit_london.csv.gz
│ │   ├── transit_new_york.csv.gz
│ │   ├── transit_paris.csv.gz
│ │   ├── transit_singapore.csv.gz
│ │   └── transit_tokyo.csv.gz
│ │
│ └── features/
│   └── transit_features.csv.gz
│
├── LICENSE.txt
│
├── metadata/
│ └── metadata.json
│
├── notebooks/
│ └── demo.ipynb
│
└── README.md
```

---

### Subsets:

**Splits:** 80% train / 10% validation / 10% test

**Featured:**

- Demand forecasting subset
- Anomaly detection

---

### Possible applications:

- Public transport demand forecasting  
- Delay prediction and schedule optimization  
- Smart city mobility planning  
- Passenger flow and congestion modeling  
- Graph neural network research on transit systems  
- Real-time anomaly detection in transit networks  
- Route optimization and capacity planning  

---

### License:

Released under CC0 1.0 Universal.
