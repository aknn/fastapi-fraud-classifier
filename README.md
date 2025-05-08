# FinCrime Sentinel: A FastAPI-Powered AML Monitoring Service

## Overview
FinCrime Sentinel is a prototype AML (Anti-Money Laundering) monitoring API built with FastAPI and Pydantic. It ingests transaction messages, applies a cascading classification pipeline (rule-based → lightweight ML → LLM), and emits risk scores and alerts for suspicious activity. This project demonstrates end-to-end capabilities in data validation, model integration, scalable deployment, and real-time monitoring.

## Key Features

### 1. Robust Data Ingestion & Validation
*   **Pydantic Models**: `MsgPayload` (defined in `app/models.py`) defines the schema for incoming transaction messages, ensuring type safety and automatic API documentation.
*   **Error Handling**: Invalid payloads return clear 422 responses with field-level error details.

### 2. Cascading Classification Pipeline
*   **Level 1: Rule-Based Filters**: Quick rejection of clearly non-suspicious messages (e.g., amounts under threshold, known low-risk counterparties).
    *   *Toy example (conceptual)*:
        ```python
        # if payload.amount < 1000 and payload.country in LOW_RISK_COUNTRIES:
        #     return {"risk_score": 0.1, "alert": False}
        ```
*   **Level 2: Lightweight ML Model**: A fine-tuned Random Forest (or small neural net) trained on historic AML cases, potentially hosted via ONNX for fast inference. Captures more nuanced patterns (velocity, counterparty networks).
*   **Level 3: LLM Deep Analysis**: For mid-confidence cases, an LLM (e.g., GPT-4) inspects transaction metadata and free-text “purpose” fields to detect creative layering or obfuscation. Escalation only when ML confidence is within a defined range (e.g., 0.4–0.6).

### 3. Real-Time Alerting & Dashboard
*   **WebSocket Notifications**: Alerts can be pushed to a simple React dashboard (conceptual) the moment a transaction crosses a risk threshold.
*   **Audit Trail**: Every decision—rule result, ML score, LLM rationale—is logged (e.g., in PostgreSQL) for compliance review.

### 4. Scalable Deployment
*   **Containerized with Docker & Kubernetes**: Envisioned for separate pods for FastAPI, ML inference, and caching.
*   **Redis Caching**: Caches repeated lookups (e.g., same counterparty risk profile), aiming to cut inference calls.
*   **Serverless Endpoints**: Low-volume “explain this alert” endpoints could run on AWS Lambda to scale to zero.

### 5. Monitoring & Governance
*   **Prometheus + Grafana**: To track request volumes, endpoint latencies, per-model cost estimates, and anomaly counts.
*   **Cost Controls**: Alerts if LLM spend exceeds a defined daily budget or average inference latency surpasses a set threshold.

## Tech Stack (Target)
*   **Backend**: FastAPI, Uvicorn, Pydantic
*   **ML**: scikit-learn (Random Forest), ONNX Runtime, Hugging Face Transformers
*   **LLM**: OpenAI GPT-4 (or similar) via API
*   **Data**: PostgreSQL for logs, Redis for caching
*   **Infra**: Docker, Kubernetes (e.g., EKS), AWS Lambda
*   **Monitoring**: Prometheus, Grafana, Sentry

## Toy Workflow Example
1.  Client `POST` to `/api/v1/transactions` with JSON matching `MsgPayload`.
2.  FastAPI validates input, then passes to `classify_transaction(payload)` (currently a placeholder in `main.py`).
3.  `classify_transaction` applies rules → ML → (if needed) LLM.
4.  Result (example):
    ```json
    {
      "msg_id": 1234,
      "risk_score": 0.78,
      "alert": true,
      "explanation": "LLM detected mention of rapid layering across multiple jurisdictions."
    }
    ```
5.  Alert pushed over WebSocket (conceptual) and logged to the audit database (conceptual).

## Why This Project Matters
*   Demonstrates the ability to design modular, production-grade APIs.
*   Showcases expertise in both classical ML and cutting-edge LLM techniques.
*   Highlights cost-management strategies (cascading models, caching, serverless scaling) crucial in a banking/FinCrime context.
*   Illustrates end-to-end thinking: data validation → inference → alerting → compliance logging.

## Project Structure

The project follows a standard structure for FastAPI applications:

```
aml-ml-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app definition, main routes
│   ├── models.py               # Pydantic models (e.g., MsgPayload)
│   ├── routers/                # API routers for different modules
│   │   ├── __init__.py
│   │   └── inference.py        # Routes related to model inference
│   └── schemas/                # Pydantic schemas (potentially more detailed)
│       ├── __init__.py
│       └── transaction.py      # Schemas related to transactions
├── data/                       # Data files
│   ├── processed/              # Processed data for modeling
│   └── raw/                    # Raw input data
├── models/                     # Trained model artifacts
│   └── saved_models/
├── notebooks/                  # Jupyter notebooks for exploration
│   └── exploratory_data_analysis.ipynb
├── src/                        # Source code for ML pipeline
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── features/               # Feature engineering scripts
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── model/                  # Model training and prediction scripts
│   │   ├── __init__.py
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── preprocessing/          # Data preprocessing scripts
│       ├── __init__.py
│       └── preprocess_data.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_api.py             # API endpoint tests
│   └── test_model.py           # Model logic tests
├── .devcontainer/              # Dev container configuration
├── .vscode/                    # VS Code settings
├── Dockerfile                  # Docker configuration for the application
├── README.md                   # This file
└── requirements.txt            # Python package dependencies
```

## Data Acquisition

Acquire the synthetic transaction dataset from Kaggle:

```bash
kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml -p data/raw --unzip
```

Ensure the CSV file is located at `data/raw/synthetic_transactions.csv`.

## Set up instructions

This sample makes use of Dev Containers. To leverage this setup, make sure you have [Docker installed](https://www.docker.com/products/docker-desktop).

To successfully run this example, we recommend the following VS Code extensions:

- [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)

## Running the sample
- Open the project folder in VS Code.
- If prompted, or by using the Command Palette (**View > Command Palette...**), run the **Dev Containers: Reopen in Container** command.
- Once the container is built and your workspace is open inside the container, you can run the app using the Run and Debug view (select the "Python: FastAPI" configuration) or by pressing `F5`.
- `Ctrl + click` on the URL that shows up in the terminal (usually `http://127.0.0.1:8000`) to open the running application in your browser.
- Test the API functionality by navigating to the `/docs` URL (e.g., `http://127.0.0.1:8000/docs`) to view the Swagger UI and interact with the API endpoints.
- To run tests, open the Test Panel in VS Code or use the **Python: Configure Tests** command from the Command Palette (select `pytest`). Then run tests from the Test Panel or directly from your test files.
