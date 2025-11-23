# 🔆 Photovoltaic Knowledge-Informed Neural Network (PKINN)

This repository provides datasets and reference implementations for photovoltaic (PV) power forecasting under fluctuating environmental conditions (FECs).  
It accompanies our research on the **Photovoltaic Knowledge-Informed Neural Network (PKINN)** framework, which aims to improve forecasting robustness under abrupt ramps and stochastic fluctuations.

---

## 📁 Repository Structure

|-- Dataset/ # Real outdoor PV power and environmental measurements
|-- code/ # Example scripts for data loading, preprocessing, and baseline forecasting
|-- README.md

---

## 📊 Dataset Overview

The dataset contains synchronized measurements of:

- **PV power output** from multiple silicon-based PV modules  
- **Environmental variables**, including:  
  - Solar irradiance  
  - Module temperature  
  - Ambient temperature, humidity, wind information (if available)  
  - Electrical characteristics  

All data are recorded at **5-minute resolution** under real outdoor operating conditions.

These measurements capture a broad spectrum of natural fluctuation behaviors, including:

- Smooth diurnal patterns  
- **Abrupt power ramps** triggered by cloud movement or irradiance drops  
- Stochastic disturbances in irradiance and temperature  

This makes the dataset suitable for training and evaluating forecasting models under highly dynamic environmental scenarios.

---

## 🔍 Purpose of This Repository

The dataset and example code support research in:

- Short-term PV power forecasting  
- Data-driven and hybrid modeling  
- Fluctuation and ramp analysis  
- Forecasting under non-stationary meteorological conditions  
- Interpretable prediction frameworks  
- Benchmarking model **robustness under real-world FECs**

---

## 🧠 About PKINN

The PKINN framework integrates:

- **Quadratic Explicit Model (QEM)**  
  - Provides explicit analytical modeling of abrupt PV power fluctuations  
  - Describes deterministic physical responses to environmental and electrical factors  

- **Fluctuation Allocation Mechanism (FAM)**  
  - Performs adaptive modeling of stochastic disturbances  
  - Adjusts learning behavior based on fluctuation intensity  

Together, QEM and FAM enhance the **interpretability**, **robustness**, and **stability** of PV forecasting across diverse fluctuation scenarios.

---


