from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from app.models import (
    TransactionPayload, ConsentRequest, ConsentScope, RiskScoreResponse,
    ExplanationResponse, AuditLogEntry, GraphQueryRequest, GraphQueryResponse,
    # Keep for backward compatibility
    FederatedTrainingUpdate, MsgPayload, PaymentType
)
from app.services.consent_service import ConsentService
from app.services.feature_store import FeatureStoreService
from app.services.graph_service import GraphService
from app.services.ml_service import MLModelService
from app.services.streaming_service import KafkaStreamingService
from app.database import get_db, create_tables
from app.database.service import DatabaseService
from src.config import settings, get_risk_level

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Privacy-aware fraud detection microservice for Trudenty",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on startup


@app.on_event("startup")
async def startup_event():
    """Initialize database tables and services"""
    create_tables()

# Dependency to get database service


def get_database_service(db: Session = Depends(get_db)) -> DatabaseService:
    """Get database service instance"""
    return DatabaseService(db)

# Service dependency functions


def get_consent_service(db_service: DatabaseService = Depends(get_database_service)) -> ConsentService:
    """Get consent service instance"""
    return ConsentService(db_service)


def get_feature_store_service(db_service: DatabaseService = Depends(get_database_service)) -> FeatureStoreService:
    """Get feature store service instance"""
    return FeatureStoreService(db_service)


def get_graph_service(db_service: DatabaseService = Depends(get_database_service)) -> GraphService:
    """Get graph service instance"""
    return GraphService(db_service)


def get_ml_service() -> MLModelService:
    """Get ML service instance"""
    return MLModelService()


def get_streaming_service() -> KafkaStreamingService:
    """Get streaming service instance"""
    return KafkaStreamingService()

# Dependencies


async def verify_consent(
    user_id: str,
    required_scopes: List[ConsentScope],
    consent_service: ConsentService = Depends(get_consent_service)
) -> Dict[str, bool]:
    """Verify user consent for required scopes"""
    consent_status = await consent_service.check_consent(user_id, required_scopes)

    # Check if all required scopes are granted
    missing_scopes = [scope for scope,
                      granted in consent_status.items() if not granted]

    if missing_scopes:
        raise HTTPException(
            status_code=403,
            detail=f"Missing consent for scopes: {missing_scopes}"
        )

    return consent_status


# Mock fraud detection model service
class FraudDetectionService:
    """Enhanced fraud detection with privacy-aware features"""

    def __init__(self):
        self.model_version = settings.model_version

    async def predict_risk(self, transaction: TransactionPayload, feature_store: FeatureStoreService, ml_service: MLModelService) -> RiskScoreResponse:
        """Predict fraud risk using multi-stage pipeline"""
        start_time = datetime.utcnow()

        # Stage 1: Rule-based filtering
        rule_score = await self._apply_rules(transaction)

        # Stage 2: Feature-based ML scoring
        ml_score = await self._ml_predict(transaction, feature_store, ml_service)

        # Stage 3: LLM enhancement (for mid-confidence cases)
        final_score = await self._llm_enhance(transaction, rule_score, ml_score)

        # Determine alert status
        risk_level = get_risk_level(final_score)
        alert = final_score >= settings.high_risk_threshold

        processing_time = (datetime.utcnow() -
                           start_time).total_seconds() * 1000

        return RiskScoreResponse(
            transaction_id=transaction.transaction_id or str(uuid.uuid4()),
            user_id=transaction.user_id,
            risk_score=final_score,
            risk_level=risk_level,
            alert=alert,
            confidence=0.85,  # Mock confidence
            model_version=self.model_version,
            processing_time_ms=processing_time
        )

    async def _apply_rules(self, transaction: TransactionPayload) -> float:
        """Rule-based risk assessment"""
        score = 0.0

        # High amount rule
        if transaction.amount > 10000:
            score += 0.3

        # Crypto payments
        if transaction.payment_type.value == "crypto":
            score += 0.2

        # Time-based rules (night transactions)
        if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 22:
            score += 0.1

        return min(score, 1.0)

    async def _ml_predict(self, transaction: TransactionPayload, feature_store: FeatureStoreService, ml_service: MLModelService) -> float:
        """ML-based prediction using cached features"""
        # Get user features
        user_features = await feature_store.get_user_features(transaction.user_id)

        # Handle optional device and merchant features safely
        device_features = {}
        if transaction.device_fingerprint:
            device_features = await feature_store.get_device_features(transaction.device_fingerprint)

        merchant_features = {}
        if transaction.merchant_id:
            merchant_features = await feature_store.get_merchant_features(transaction.merchant_id)

        # Use ML service for enhanced prediction
        fraud_probability, feature_importance = await ml_service.predict_fraud_probability(
            transaction, user_features, device_features, merchant_features
        )

        return fraud_probability

    async def _llm_enhance(self, transaction: TransactionPayload, rule_score: float, ml_score: float) -> float:
        """LLM enhancement for mid-confidence cases"""
        combined_score = (rule_score + ml_score) / 2

        # Only use LLM for mid-confidence cases (0.4-0.6 range)
        if 0.4 <= combined_score <= 0.6:
            # Mock LLM enhancement (in production, would call actual LLM API)
            llm_adjustment = 0.1 if transaction.description and "urgent" in transaction.description.lower() else - \
                0.05
            return max(0.0, min(1.0, combined_score + llm_adjustment))

        return combined_score


# Initialize fraud detection service
fraud_service = FraudDetectionService()


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/about")
def about() -> Dict[str, str]:
    """About page"""
    return {"message": "This is the about page."}


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "status": "active"
    }


# Privacy Consent Management
@app.post("/consent", response_model=ConsentRequest)
async def grant_consent(
    user_id: str,
    scopes: List[ConsentScope],
    expires_days: Optional[int] = None,
    consent_service: ConsentService = Depends(get_consent_service)
) -> ConsentRequest:
    """Grant user consent for data processing scopes"""
    expires_at = None
    if expires_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=expires_days)

    consent = await consent_service.grant_consent(user_id, scopes, expires_at)
    return consent


@app.get("/consent/{user_id}")
async def get_consent(
    user_id: str,
    consent_service: ConsentService = Depends(get_consent_service)
) -> List[Dict[str, Any]]:
    """Get current consent status for a user"""
    return await consent_service.get_user_consents(user_id)


@app.delete("/consent/{user_id}")
async def revoke_consent(
    user_id: str,
    scopes: Optional[List[ConsentScope]] = None,
    consent_service: ConsentService = Depends(get_consent_service)
) -> Dict[str, str]:
    """Revoke user consent"""
    success = await consent_service.revoke_consent(user_id, scopes)

    if success:
        return {"message": "Consent revoked successfully"}
    else:
        raise HTTPException(
            status_code=404, detail="No consent found to revoke")


# Enhanced Transaction Processing
@app.post("/risk-score", response_model=RiskScoreResponse)
async def calculate_risk_score(
    transaction: TransactionPayload,
    background_tasks: BackgroundTasks,
    db_service: DatabaseService = Depends(get_database_service),
    consent_service: ConsentService = Depends(get_consent_service),
    feature_store: FeatureStoreService = Depends(get_feature_store_service),
    ml_service: MLModelService = Depends(get_ml_service),
    graph_service: GraphService = Depends(get_graph_service),
    streaming_service: KafkaStreamingService = Depends(get_streaming_service)
) -> RiskScoreResponse:
    """Calculate fraud risk score for a transaction"""

    # Verify consent first
    await verify_consent(transaction.user_id, [ConsentScope.RISK_SCORING], consent_service)

    # Generate transaction ID if not provided
    if not transaction.transaction_id:
        transaction.transaction_id = str(uuid.uuid4())

    # Get risk prediction
    risk_response = await fraud_service.predict_risk(transaction, feature_store, ml_service)

    # Store transaction in database
    await db_service.create_transaction(transaction, risk_response)

    # Background tasks
    background_tasks.add_task(update_features_async,
                              transaction, feature_store)
    background_tasks.add_task(update_graph_async, transaction, graph_service)
    background_tasks.add_task(
        log_audit_async, transaction, risk_response, db_service)
    background_tasks.add_task(publish_to_kafka_async,
                              transaction, risk_response, streaming_service)

    # Create alert if needed
    if risk_response.alert:
        background_tasks.add_task(
            create_alert_async,
            transaction,
            risk_response,
            db_service
        )

    return risk_response


@app.get("/explanations/{transaction_id}", response_model=ExplanationResponse)
async def get_explanation(
    transaction_id: str,
    user_id: str,
    consent_service: ConsentService = Depends(get_consent_service),
    feature_store: FeatureStoreService = Depends(get_feature_store_service),
    ml_service: MLModelService = Depends(get_ml_service)
) -> ExplanationResponse:
    """Get model explanation for a transaction decision"""

    # Verify consent
    await verify_consent(user_id, [ConsentScope.BASIC_ANALYSIS], consent_service)

    # In a real implementation, you would retrieve the transaction from storage
    # For now, we'll create a mock transaction to demonstrate SHAP explanations
    mock_transaction = TransactionPayload(
        transaction_id=transaction_id,
        user_id=user_id,
        amount=5000.0,
        currency="USD",
        payment_type=PaymentType.CARD,
        merchant_id=None,
        device_fingerprint=None,
        ip_address=None,
        location=None,
        description="Mock transaction for explanation"
    )

    # Get features for explanation
    user_features = await feature_store.get_user_features(user_id)
    device_features = {}
    merchant_features = {}

    # Generate explanation using ML service
    fraud_probability, feature_importance = await ml_service.predict_fraud_probability(
        mock_transaction, user_features, device_features, merchant_features
    )

    explanation = await ml_service.generate_explanation(
        transaction_id, mock_transaction, feature_importance, fraud_probability
    )

    return explanation


# Graph-based Entity Relationships
@app.post("/graph/query", response_model=GraphQueryResponse)
async def query_graph(
    request: GraphQueryRequest,
    graph_service: GraphService = Depends(get_graph_service)
) -> GraphQueryResponse:
    """Query entity relationships and risk propagation"""
    return await graph_service.query_entity_relationships(request)


@app.get("/graph/fraud-rings/{entity_id}")
async def detect_fraud_rings(
    entity_id: str,
    entity_type: str = "user",
    graph_service: GraphService = Depends(get_graph_service)
) -> List[Dict[str, Any]]:
    """Detect potential fraud rings involving an entity"""
    return await graph_service.detect_fraud_rings(entity_type, entity_id)


# Fraud Alerts Management
@app.get("/alerts")
async def get_fraud_alerts(
    limit: int = 100,
    status: Optional[str] = "OPEN",
    db_service: DatabaseService = Depends(get_database_service)
) -> List[Dict[str, Any]]:
    """Get fraud alerts"""
    alerts = db_service.get_open_alerts(limit)

    return [
        {
            "alert_id": alert.alert_id,
            "transaction_id": alert.transaction_id,
            "user_id": alert.user_id,
            "risk_score": alert.risk_score,
            "risk_level": alert.risk_level,
            "alert_type": alert.alert_type,
            "priority": alert.priority,
            "status": alert.status,
            "alert_timestamp": alert.alert_timestamp.isoformat(),
            "model_version": alert.model_version,
            "confidence": alert.confidence
        }
        for alert in alerts
    ]


# Audit and Compliance
@app.get("/audit", response_model=List[AuditLogEntry])
async def get_audit_logs(
    user_id: Optional[str] = None,
    limit: int = 100,
    db_service: DatabaseService = Depends(get_database_service)
) -> List[AuditLogEntry]:
    """Retrieve audit logs for compliance"""
    db_logs = db_service.get_audit_logs(user_id, limit)

    # Convert database models to response models
    audit_entries = []
    for log in db_logs:
        # Access SQLAlchemy model attributes directly - they return actual values
        # Using type: ignore to bypass type checker confusion with SQLAlchemy instances
        audit_entries.append(AuditLogEntry(
            log_id=log.log_id,  # type: ignore
            transaction_id=log.transaction_id,  # type: ignore
            user_id=log.user_id,  # type: ignore
            timestamp=log.timestamp,  # type: ignore
            model_version=log.model_version,  # type: ignore
            input_features=log.input_features,  # type: ignore
            risk_score=log.risk_score,  # type: ignore
            decision=log.decision,  # type: ignore
            processing_pipeline=log.processing_pipeline,  # type: ignore
            consent_status=log.consent_status  # type: ignore
        ))

    return audit_entries


# Monitoring and Management Endpoints
@app.get("/status")
async def get_system_status(
    db_service: DatabaseService = Depends(get_database_service),
    streaming_service: KafkaStreamingService = Depends(get_streaming_service),
    ml_service: MLModelService = Depends(get_ml_service)
) -> Dict[str, Any]:
    """Get overall system status and health metrics"""
    # Get audit log count from database
    # Get all for count - fixed: removed await since get_audit_logs is synchronous
    audit_logs = db_service.get_audit_logs(limit=10000)

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "services": {
            "consent_service": "active",
            "feature_store": "active",
            "graph_service": "active",
            "ml_service": "active",
            "streaming_service": await streaming_service.get_consumer_status()
        },
        "audit_logs_count": len(audit_logs),
        "model_info": ml_service.get_model_info()
    }


@app.get("/metrics")
async def get_metrics(
    db_service: DatabaseService = Depends(get_database_service)
) -> Dict[str, Any]:
    """Get system metrics for monitoring"""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    # Get metrics from database - fixed: removed await since get_audit_logs is synchronous
    audit_logs = db_service.get_audit_logs(limit=10000)
    # Fixed: properly compare string values from SQLAlchemy model instances
    alerts = [log for log in audit_logs if getattr(
        log, 'decision', None) == "ALERT"]

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "transactions_processed": len(audit_logs),
        "alerts_generated": len(alerts),
        "average_processing_time_ms": 50.0,  # Mock metric
        "memory_usage_mb": 256,  # Mock metric
        "model_version": settings.model_version,
        "uptime_seconds": 3600  # Mock metric
    }


# Mock Federated Learning
@app.post("/federated/update")
async def federated_update(
    update: FederatedTrainingUpdate
) -> Dict[str, str]:
    """Accept federated learning model updates"""
    if not settings.federated_enabled:
        raise HTTPException(
            status_code=501, detail="Federated learning not enabled")

    # In production, this would update global model parameters
    return {
        "message": "Model update received",
        "client_id": update.client_id,
        "status": "processed"
    }


# Legacy endpoint for backward compatibility
@app.post("/api/v1/transactions")
async def process_legacy_transaction(payload: MsgPayload) -> Dict[str, Any]:
    """Legacy transaction processing endpoint"""
    # Simple classification for backward compatibility
    is_suspicious = "suspicious" in payload.msg_name.lower()

    return {
        "msg_id": payload.msg_id,
        "risk_score": 0.85 if is_suspicious else 0.1,
        "alert": is_suspicious,
        "explanation": "Legacy classification based on keyword detection"
    }


# Background task functions
async def update_features_async(transaction: TransactionPayload, feature_store: FeatureStoreService):
    """Update feature store in background"""
    # Use compute_user_features instead of update_transaction_features
    transaction_data = transaction.dict()
    await feature_store.compute_user_features(transaction.user_id, transaction_data)


async def update_graph_async(transaction: TransactionPayload, graph_service: GraphService):
    """Update graph relationships in background"""
    # Use add_transaction_relationship instead of add_transaction_to_graph
    await graph_service.add_transaction_relationship(
        transaction.user_id,
        transaction.merchant_id or "unknown_merchant",
        transaction.amount,
        transaction.timestamp
    )


async def publish_to_kafka_async(transaction: TransactionPayload, risk_response: RiskScoreResponse, streaming_service: KafkaStreamingService):
    """Publish transaction and alerts to Kafka in background"""
    await streaming_service.publish_transaction(transaction, risk_response)
    if risk_response.alert:
        await streaming_service.publish_alert(transaction, risk_response)


async def create_alert_async(
    transaction: TransactionPayload,
    risk_response: RiskScoreResponse,
    db_service: DatabaseService
):
    """Create fraud alert in database"""
    await db_service.create_fraud_alert(
        transaction_id=transaction.transaction_id or str(uuid.uuid4()),
        user_id=transaction.user_id,
        risk_score=risk_response.risk_score,
        risk_level=risk_response.risk_level,
        alert_type="FRAUD_SUSPECTED",
        model_version=risk_response.model_version
    )


async def log_audit_async(
    transaction: TransactionPayload,
    risk_response: RiskScoreResponse,
    db_service: DatabaseService
):
    """Log transaction for audit trail"""
    audit_entry = AuditLogEntry(
        log_id=str(uuid.uuid4()),
        transaction_id=transaction.transaction_id or str(uuid.uuid4()),
        user_id=transaction.user_id,
        timestamp=datetime.utcnow(),
        model_version=risk_response.model_version,
        input_features=transaction.dict(),
        risk_score=risk_response.risk_score,
        decision="ALERT" if risk_response.alert else "PASS",
        processing_pipeline=["rules", "ml", "llm"],
        consent_status={"risk_scoring": True, "basic_analysis": True}
    )
    await db_service.create_audit_log(audit_entry)
