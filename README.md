# FloodGuard – Real-Time Flood Risk Forecaster

FloodGuard is a Streamlit dashboard that predicts flood risk using real-time weather and terrain intelligence with a lightweight ML model.

## Features
- Real-time weather fetch via OpenWeatherMap (with simulated fallback)
- Terrain inputs from CSV (district-level) with fallback synthesis
- RandomForest flood risk classifier (synthetic training)
- KPIs, trend/gauge charts, risk zone heatmap, feature importance, and 72h timeline
- Forecast rainfall aggregation (next 24h) used in prediction
 - Tabbed UI with Overview, Forecast, Map, and Insights
 - Optional OpenWeather precipitation overlay on the risk map
- Auto Mode toggle, district selector, manual overrides
- Downloadable CSV report

## Requirements
- Python 3.9+

## Setup
```bash
pip install streamlit plotly scikit-learn pandas numpy requests
```

Optional: set your OpenWeatherMap API key as an environment variable (recommended):
```bash
# Windows PowerShell
$env:OWM_API_KEY = "3c7567ccdb5b204bec470f96f6115533"
```

Optional terrain CSV: place `data/terrain.csv` with columns:
```
district,elevation,drainage,urban
Chennai,100,0.4,0.9
Tiruchengodu,90,0.45,0.6
...
```

## Run
```bash
streamlit run FloodGuard_dashboard.py
```

## Models Used
- RandomForestClassifier (scikit-learn): multi-class classification of flood risk level using features: rainfall (observed+projected), elevation, drainage, soil saturation, urban density.
- Synthetic training data generator: rule-based function to produce realistic feature-label pairs for demonstration when labeled data is unavailable.

Why RF here: robust to nonlinearity, feature interactions, and mixed scales; fast to train; easy to inspect via feature importances.

Potential upgrades:
- Replace synthetic training with labeled historical flood events; add time-series features (antecedent rainfall indices), and try Gradient Boosted Trees (XGBoost/LightGBM, CatBoost) or calibrated classifiers for better probability estimates.
- Consider sequence models (Temporal Fusion Transformer) if high-quality hourly series and exogenous features are available.

## Interview Prep – What to Learn/Talk About
- Problem framing: target definition (risk tiers), features (weather, hydrology, terrain, urbanization), and forecast horizon.
- Data engineering: joining real-time weather (OpenWeather), terrain (Bhuvan/SRTM), administrative mapping; handling missing/lagged data; scaling to multiple districts.
- Modeling choices: RandomForest basics, class imbalance handling, calibration, cross-validation strategy by time and geography to avoid leakage.
- Evaluation: AUROC/PR for binary tiers, macro-F1 across risk levels, cost-sensitive metrics aligned with alerts; backtesting with sliding windows.
- Uncertainty and decisioning: thresholds for alerts, hysteresis to avoid flip-flop, combining model with simple safety rules.
- MLOps: caching, feature pipelines, monitoring data drift, A/B testing thresholds.
- Visualization/UX: KPIs, risk maps, timelines; communicating uncertainty.

Talking points specific to this app:
- Combined-rainfall feature: sums last-24h with projected next-24h rainfall to better capture near-term flood potential.
- Feature importance: shows rainfall and drainage prominence; supports stakeholder explanations.
- Map overlay: OpenWeather precipitation tiles over risk heatmap for situational awareness.

## Notes
- This project uses synthetic data for model training and as fallback for missing feeds; plug in authoritative sources (IMD/Bhuvan/SRTM) as available.
- For production use, replace synthetic training with labeled historical flood datasets.
