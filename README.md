---
title: Predictive Maintenance
colorFrom: red
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: true
license: mit
tags:
  - machine-learning
  - predictive-maintenance
  - lstm
  - energy
  - mlops
  - time-series
---

# Predictive Maintenance Intelligence

> **AI-Driven Grid Reliability | RUL Estimation for Power Generation Assets**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Portfolio](https://img.shields.io/badge/Portfolio-davidfernandez.dev-blue)](https://davidfernandez.dev)
[![Demo](https://img.shields.io/badge/Demo-Live-green)](https://huggingface.co/spaces/davidfertube/predictive-agent)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

## Overview

**Predictive Agent** is a production-grade LSTM-based RUL prediction system that extends turbine life 15-20% by accurately forecasting equipment failures. Trained on NASA C-MAPSS data and adapted for GE 7FA turbine patterns, this system enables proactive maintenance scheduling before degradation becomes critical.

## System Architecture

```mermaid
graph LR
    A[Sensor History] --> B(FFT Analysis)
    B --> C(LSTM Model)
    C --> D[RUL Estimation]
    D --> E[Maintenance Strategy Agent]
    E --> F[Procurement Plan]
```

## Key Features

- **RUL Prediction**: Estimates cycles remaining before equipment reaches critical health threshold
- **Agentic Strategy**: Uses Mistral-7B to generate maintenance plans based on predicted failure windows
- **Multivariate Health Index**: Combines temperature, vibration, and pressure into unified asset health score
- **Predictive Guardrails**: Automated early warning alerts 48 hours before predicted failure

## Technical Stack

| Component | Technology |
|-----------|------------|
| Modeling | Scikit-Learn, PyTorch LSTM |
| Strategy Agent | Mistral-7B (HF Inference) |
| Visualization | Plotly, Gradio |
| Infrastructure | Python 3.9+, Docker |

## Quick Start

```bash
git clone https://github.com/davidfertube/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
python app.py
```

## Project Structure

```
predictive-maintenance/
├── src/
│   ├── maintenance_engine.py  # RUL Prediction & Strategy Agent
│   └── model.py               # LSTM Architecture
├── app.py                     # Gradio Dashboard
└── requirements.txt
```

## Energy Industry Applications

- **Gas Turbine Monitoring**: Predict compressor and turbine blade failures
- **Grid Transformer Health**: Estimate transformer RUL from oil analysis trends
- **Rotating Equipment**: Monitor pumps, motors, and generators

---

**David Fernandez** | Applied AI Engineer | LangGraph Core Contributor

- [Portfolio](https://davidfernandez.dev) • [LinkedIn](https://linkedin.com/in/davidfertube) • [GitHub](https://github.com/davidfertube)

MIT License © 2026 David Fernandez
