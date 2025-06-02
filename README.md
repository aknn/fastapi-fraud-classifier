# Privacy-Aware Fraud Detection API

## Overview
This microservice implements a privacy-first fraud detection workflow using FastAPI and Pydantic. It ingests transaction data, applies a cascading pipeline (rule-based → ML → optional LLM), enforces user consent, and provides full auditability.

## API Endpoints
* **GET /** – Service metadata (name, version, status)
* **GET /health** – Liveness and readiness check
* **GET /about** – About page
* **POST /consent** – Grant user consent with scopes and TTL
* **GET /consent/{user_id}** – Retrieve current consent status
* **DELETE /consent/{user_id}** – Revoke consent (optional scopes)
* **POST /risk-score** – Submit a transaction payload and receive a risk score response
* **GET /explanations/{transaction_id}?user_id={user_id}** – Retrieve SHAP explanation for a past transaction
* **GET /audit** – Fetch immutable audit log entries for compliance
* **POST /graph/query** – Query graph-based relationships and indirect risk propagation
* **GET /graph/fraud-rings/{entity_id}?entity_type={type}** – Detect suspicious rings in the entity graph
* **GET /status** – Aggregate service health and component statuses
* **GET /metrics** – System metrics (requests, alerts, processing time)

## Core Components
### 1. Consent Service
Manages user-level data sharing permissions with scopes (risk_scoring, graph_analytics, etc.) and issues consent tokens.

### 2. Risk Scoring Pipeline
- **Rule-Based Filters**: Quick checks (amount thresholds, payment type)
- **Machine Learning**: RandomForest/ONNX model for pattern detection
- **LLM Enhancement**: Optional deep analysis for mid-confidence cases

### 3. Explanations
SHAP-based transparency (`/explanations`) to expose feature contributions for each risk decision.

### 4. Audit Trail
Append-only, immutable log of every decision, timestamp, model version, and input features accessible via `/audit`.

### 5. Graph Service
Tracks user/device/merchant relationships in-memory and via Redis; supports `/graph/query` and fraud-ring detection.

### 6. Streaming Service
Kafka-backed (or mock) ingestion and alerting pipeline for real-time transaction processing and notifications.

### 7. Federated Learning (Stub)
Conceptual federated training endpoint to simulate weight aggregation across multiple clients.

### 8. Feature Store
Redis cache for entity-level features (velocity, frequency, average amount) to support rapid scoring.

## Tech Stack
* **Backend**: FastAPI, Uvicorn, Pydantic
* **ML**: scikit-learn, ONNX Runtime, SHAP
* **Graph**: NetworkX, Redis
* **Streaming**: Kafka or asyncio mocks
* **DB**: PostgreSQL (SQLAlchemy)
* **Caching**: Redis
* **Testing**: pytest, TestClient

## Project Layout
```
fastapi-fraud-service/
├── app/
│   ├── main.py          # FastAPI app and routers
│   ├── models.py        # Pydantic schemas
│   ├── services/        # Service modules (consent, graph, ml, streaming)
│   └── database/        # SQLAlchemy setup and service
├── src/config.py        # Configuration (env / BaseSettings)
├── tests/               # pytest suite
├── README.md            # This file
├── Dockerfile
└── requirements.txt
```

## Getting Started
1. Reopen in Dev Container (Docker)
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `uvicorn main:app --reload`
4. Explore Swagger UI at `http://localhost:8000/docs`

## Testing
Run the test suite with:
```bash
pytest --maxfail=1 --disable-warnings -q
```
