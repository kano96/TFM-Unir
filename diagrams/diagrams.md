```mermaid
graph TD

  %% ------------------------
  %% Ingest & Observability
  %% ------------------------
  subgraph OBS["Ingest & Observability"]
    SIM[Simulator / Microservices]
    PROM[Prometheus Metrics]
    LOKI[Loki Logs]
    JAEGER[Jaeger Traces]

    SIM --> PROM
    SIM --> LOKI
    SIM --> JAEGER
  end

  %% ------------------------
  %% AIOps Layer
  %% ------------------------
  subgraph AIOPS["AIOps Layer"]
    FE[Feature Extractor]
    DET[Detector Service\nIsolationForest / Autoencoder]
    PRED[Predictor Service\nProphet / LSTM]
    RCA[RCA Service\nGraph Analysis + NLP]

    PROM --> FE
    LOKI --> FE
    JAEGER --> FE

    FE --> DET
    FE --> PRED
    FE --> RCA
  end

  %% ------------------------
  %% Visualization & Alerting
  %% ------------------------
  subgraph VIS["Visualization & Alerting"]
    GRAF[Grafana Dashboards]
    ALERT[Alert Router]
    SNS[AWS SNS or Slack]

    DET --> GRAF
    PRED --> GRAF
    RCA --> GRAF

    DET --> ALERT --> SNS
    PRED --> ALERT --> SNS
  end

  %% ------------------------
  %% Repo & CI/CD
  %% ------------------------
  subgraph CICD["CI/CD & Repository"]
    GH[GitHub Actions]
    ECR[AWS ECR Registry]
    TF[Terraform IaC]

    GH --> ECR
    GH --> TF
  end

  %% ------------------------
  %% AWS Infra
  %% ------------------------
  subgraph AWS["AWS Infrastructure"]
    EKS[EKS Cluster]
    S3[S3 Buckets\nDatasets + Models]
    CW[CloudWatch]

    EKS --> DET
    EKS --> PRED
    EKS --> RCA
    EKS --> FE

    FE --> S3
    EKS --> CW
  end

  GH --> EKS
