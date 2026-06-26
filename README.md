# YieldGuard

A semiconductor-themed AI service, built solo over 60 days as a production-grade MLOps portfolio piece (SHIP/60 challenge). Predicts pass/fail on chip manufacturing process data (UCI SECOM dataset), served via FastAPI, containerized, deployed to Kubernetes on Azure, monitored with Prometheus/Grafana, and extended with a RAG "Explainer" head.

Model accuracy is not the point — the infrastructure around it is what's being graded.

## Progress

- [ ] **Level 0 — Base Camp**: baseline model, FastAPI `/predict` + stub `/ask`, Dockerized
- [ ] **Level 1 — Kubernetes**: Minikube, Pod/Deployment/Service, ConfigMap/Namespace, Helm chart
- [ ] **Level 2 — Azure + AZ-900**: ACR, AKS, AZ-900 certification
- [ ] **Level 3 — Observability**: Prometheus + Grafana, alert rule
- [ ] **Level 4 — LLMOps**: RAG on `/ask`, Langfuse tracing, eval metrics
- [ ] **Stretch — Terraform**: IaC for Azure ML + AKS, GitHub Actions CI/CD
- [ ] **Final — Ship & Document**: architecture diagram, CV/LinkedIn update

## Stack

Python · FastAPI · Docker · Kubernetes · Helm · Azure (AKS/ACR) · Prometheus · Grafana · RAG/Langfuse · Terraform (stretch)

## Data

UCI SECOM dataset, loaded via `ucimlrepo.fetch_ucirepo(id=179)` — 1567 rows of semiconductor process sensor data, pre-labelled pass/fail.
