"""
Inference-related API routes for fraud detection
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from app.models import (
    TransactionPayload, RiskScoreResponse, ExplanationResponse,
    ConsentScope, GraphQueryRequest, GraphQueryResponse
)
from app.services.consent_service import consent_service
from app.services.feature_store import feature_store
from app.services.graph_service import graph_service
from app.services.ml_service import ml_service
from app.services.streaming_service import streaming_service
from src.config import settings, get_risk_level

router = APIRouter(prefix="/api/v1", tags=["inference"])


# Dependencies
async def verify_consent_dependency(
    user_id: str,
    required_scopes: List[ConsentScope]
) -> Dict[str, bool]:
    """Verify user consent for required scopes"""
    consent_status = await consent_service.check_consent(user_id, required_scopes)

    missing_scopes = [scope for scope,
                      granted in consent_status.items() if not granted]

    if missing_scopes:
        raise HTTPException(
            status_code=403,
            detail=f"Missing consent for scopes: {missing_scopes}"
        )

    return consent_status


@router.post("/predict", response_model=RiskScoreResponse)
async def predict_fraud_risk(
    transaction: TransactionPayload,
    background_tasks: BackgroundTasks
) -> RiskScoreResponse:
    """Predict fraud risk for a transaction"""

    # Verify consent
    await verify_consent_dependency(transaction.user_id, [ConsentScope.RISK_SCORING])

    # Generate transaction ID if not provided
    if not transaction.transaction_id:
        transaction.transaction_id = str(uuid.uuid4())

    start_time = datetime.utcnow()

    # Get features
    user_features = await feature_store.get_user_features(transaction.user_id)
    device_features = {}
    if transaction.device_fingerprint:
        device_features = await feature_store.get_device_features(transaction.device_fingerprint)

    merchant_features = {}
    if transaction.merchant_id:
        merchant_features = await feature_store.get_merchant_features(transaction.merchant_id)

    # Get ML prediction
    fraud_probability, feature_importance = await ml_service.predict_fraud_probability(
        transaction, user_features, device_features, merchant_features
    )

    # Determine alert status
    risk_level = get_risk_level(fraud_probability)
    alert = fraud_probability >= settings.high_risk_threshold

    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

    risk_response = RiskScoreResponse(
        transaction_id=transaction.transaction_id,
        user_id=transaction.user_id,
        risk_score=fraud_probability,
        risk_level=risk_level,
        alert=alert,
        confidence=0.85,  # Mock confidence
        model_version=settings.model_version,
        processing_time_ms=processing_time
    )

    # Background tasks
    background_tasks.add_task(update_features_background, transaction)
    background_tasks.add_task(update_graph_background, transaction)
    background_tasks.add_task(
        publish_to_kafka_background, transaction, risk_response)

    return risk_response


@router.get("/explain/{transaction_id}", response_model=ExplanationResponse)
async def explain_prediction(
    transaction_id: str,
    user_id: str
) -> ExplanationResponse:
    """Get explanation for a fraud prediction"""

    # Verify consent
    await verify_consent_dependency(user_id, [ConsentScope.BASIC_ANALYSIS])

    # In production, retrieve the actual transaction from database
    # For now, create a mock transaction
    mock_transaction = TransactionPayload(
        transaction_id=transaction_id,
        user_id=user_id,
        amount=5000.0,
        currency="USD",
        payment_type="card",
        merchant_id=None,
        device_fingerprint=None,
        ip_address=None,
        location=None,
        description="Mock transaction for explanation"
    )

    # Get features
    user_features = await feature_store.get_user_features(user_id)
    device_features = {}
    merchant_features = {}

    # Generate explanation
    fraud_probability, feature_importance = await ml_service.predict_fraud_probability(
        mock_transaction, user_features, device_features, merchant_features
    )

    explanation = await ml_service.generate_explanation(
        transaction_id, mock_transaction, feature_importance, fraud_probability
    )

    return explanation


# Background task functions
async def update_features_background(transaction: TransactionPayload):
    """Update feature store in background"""
    await feature_store.update_transaction_features(transaction)


async def update_graph_background(transaction: TransactionPayload):
    """Update graph relationships in background"""
    await graph_service.add_transaction_to_graph(transaction)


async def publish_to_kafka_background(transaction: TransactionPayload, risk_response: RiskScoreResponse):
    """Publish to Kafka in background"""
    await streaming_service.publish_transaction(transaction)
    if risk_response.alert:
        await streaming_service.publish_alert(risk_response, transaction)
