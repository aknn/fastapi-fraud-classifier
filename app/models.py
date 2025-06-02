from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ConsentScope(str, Enum):
    """Privacy consent scopes for user data"""
    BASIC_ANALYSIS = "basic_analysis"
    RISK_SCORING = "risk_scoring"
    GRAPH_ANALYTICS = "graph_analytics"
    VELOCITY_TRACKING = "velocity_tracking"
    DEVICE_FINGERPRINTING = "device_fingerprinting"


class PaymentType(str, Enum):
    """Payment method types"""
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CRYPTO = "crypto"
    CASH = "cash"


class TransactionPayload(BaseModel):
    """Enhanced transaction payload for fraud detection"""
    transaction_id: Optional[str] = None
    user_id: str = Field(..., description="Unique user identifier (hashed)")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(..., description="Transaction currency code")
    payment_type: PaymentType
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    device_fingerprint: Optional[str] = Field(None, description="Device hash")
    ip_address: Optional[str] = Field(
        None, description="IP address (hashed for privacy)")
    location: Optional[str] = Field(None, description="Transaction location")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = Field(
        None, max_length=500, description="Transaction description")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v


class ConsentRequest(BaseModel):
    """User consent management"""
    user_id: str
    scopes: List[ConsentScope]
    granted: bool
    expires_at: Optional[datetime] = None
    consent_token: Optional[str] = None


class RiskScoreResponse(BaseModel):
    """Risk scoring response"""
    transaction_id: str
    user_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    alert: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    processing_time_ms: float


class ExplanationResponse(BaseModel):
    """Model explanation response using SHAP"""
    transaction_id: str
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None
    top_risk_factors: List[str]
    explanation_text: str


class AuditLogEntry(BaseModel):
    """Immutable audit log entry"""
    log_id: str
    transaction_id: str
    user_id: str
    timestamp: datetime
    model_version: str
    input_features: Dict[str, Any]
    risk_score: float
    decision: str
    processing_pipeline: List[str]
    consent_status: Dict[str, bool]


class GraphQueryRequest(BaseModel):
    """Graph-based entity relationship query"""
    entity_id: str
    entity_type: str = Field(..., description="user, device, merchant, etc.")
    relationship_depth: int = Field(default=2, ge=1, le=3)
    include_risk_propagation: bool = True
    query_type: str = Field(default="neighbors",
                            description="neighbors, paths, components")
    max_depth: int = Field(default=3, ge=1, le=5)
    target_entity_type: Optional[str] = None
    target_entity_id: Optional[str] = None


class GraphQueryResponse(BaseModel):
    """Graph query response"""
    entity_id: str
    direct_connections: List[Dict[str, Any]]
    indirect_risk_score: float
    suspicious_patterns: List[str]
    network_metrics: Dict[str, float]


class FederatedTrainingUpdate(BaseModel):
    """Mock federated learning update"""
    client_id: str
    model_weights: Dict[str, List[float]]
    training_samples: int
    privacy_budget_consumed: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FeatureStoreEntry(BaseModel):
    """Feature store cached entry"""
    entity_id: str
    entity_type: str
    features: Dict[str, Any]
    last_updated: datetime
    ttl_seconds: int = 3600


# Legacy model for backward compatibility
class MsgPayload(BaseModel):
    msg_id: Optional[int]
    msg_name: str
