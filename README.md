# Health Anomaly Detector


The **Health Anomaly Detector** is a Python package for anomaly detection on health data. It offers:

- **Data Loading:** Import data from CSV files or SQL databases.
- **Data Preprocessing:** Convert date columns, clean string fields, etc.
- **Grouping:** Assign groups (e.g., regions or cohorts) using custom mappings.
- **Baseline Analysis:** Calculate baseline metrics using an IQR-based approach.
- **Anomaly Detection:** Apply statistical (IQR) or ML-based methods (IsolationForest, OneClassSVM, LocalOutlierFactor).
- **Regression Analysis:** Optionally run Ridge or Lasso regression analyses.
- **Visualization:** Generate plots, combine them into PDF reports, and plot geospatial data.
  
## Installation

### Prerequisites
Ensure you have Python 3.6+ installed. This package depends on several libraries which will be installed automatically. 

### Clone and Install

1. **Clone the Repository:**

   Open your terminal and run:

   ```bash
   git clone https://github.com/yourusername/health_anomaly_detector.git
   cd health_anomaly_detector

