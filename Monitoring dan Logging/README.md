# Monitoring dan Logging - Iris Experiment

This component provides a complete monitoring stack for the Iris machine learning model.

## Components
- **2.prometheus.yml**: Configuration for scraping metrics from the local exporter on port **9090**.
- **3.prometheus_exporter.py**: A Python script that exposes custom metrics (Inference Count, Latency, Accuracy) for Prometheus.
- **7.Inference.py**: Simulation script that generates model predictions and triggers metric updates.

## Monitoring Access
- **Grafana Username**: `stefan_aprilio_xONr`
- **Prometheus Port**: `9090`

## Proof of Implementation
The folders `4.bukti monitoring Prometheus`, `5.bukti monitoring Grafana`, and `6.bukti alerting Grafana` contain visual evidence of the monitoring dashboards and alerting rules.
