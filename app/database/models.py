"""
SQLAlchemy Database Models for Privacy-Aware Fraud Detection
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Transaction(Base):
    """Transaction records for audit and analysis"""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(255), unique=True,
                            index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False)
    payment_type = Column(String(50), nullable=False)
    merchant_id = Column(String(255), index=True, nullable=True)
    device_fingerprint = Column(String(255), index=True, nullable=True)
    ip_address_hash = Column(String(255), nullable=True)  # Hashed for privacy
    location = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    # Fraud detection results
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)
    alert = Column(Boolean, default=False)
    model_version = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=True)


class AuditLog(Base):
    """Immutable audit logs for compliance"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(255), unique=True, index=True, nullable=False)
    transaction_id = Column(String(255), index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_version = Column(String(50), nullable=False)
    input_features = Column(JSON, nullable=False)
    risk_score = Column(Float, nullable=False)
    decision = Column(String(20), nullable=False)  # ALERT, PASS, REJECT
    # List of processing steps
    processing_pipeline = Column(JSON, nullable=False)
    # Consent verification results
    consent_status = Column(JSON, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class ConsentRecord(Base):
    """User consent records for privacy compliance"""
    __tablename__ = "consent_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True, nullable=False)
    scopes = Column(JSON, nullable=False)  # List of consent scopes
    granted = Column(Boolean, nullable=False)
    granted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    consent_token = Column(Text, nullable=True)
    ip_address_hash = Column(String(255), nullable=True)  # Hashed for privacy
    user_agent_hash = Column(String(255), nullable=True)  # Hashed for privacy
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(),
                        onupdate=func.now())


class FeatureCache(Base):
    """Cached feature computations"""
    __tablename__ = "feature_cache"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String(255), index=True, nullable=False)
    # user, device, merchant
    entity_type = Column(String(50), index=True, nullable=False)
    features = Column(JSON, nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    feature_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(),
                        onupdate=func.now())


class EntityRelationship(Base):
    """Graph-based entity relationships"""
    __tablename__ = "entity_relationships"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(50), index=True, nullable=False)
    source_id = Column(String(255), index=True, nullable=False)
    target_type = Column(String(50), index=True, nullable=False)
    target_id = Column(String(255), index=True, nullable=False)
    # uses_device, transacts_with, etc.
    relationship_type = Column(String(50), nullable=False)
    weight = Column(Float, default=1.0)  # Relationship strength
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    transaction_count = Column(Integer, default=1)
    total_amount = Column(Float, default=0.0)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(),
                        onupdate=func.now())


class ModelMetrics(Base):
    """Model performance metrics tracking"""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50), index=True, nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    # precision, recall, f1, auc, etc.
    metric_type = Column(String(50), nullable=False)
    evaluation_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    dataset_size = Column(Integer, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata
    created_at = Column(DateTime, server_default=func.now())


class FraudAlert(Base):
    """Fraud alerts for monitoring and response"""
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(255), unique=True, index=True, nullable=False)
    transaction_id = Column(String(255), index=True, nullable=False)
    user_id = Column(String(255), index=True, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    # FRAUD_SUSPECTED, VELOCITY_EXCEEDED, etc.
    alert_type = Column(String(50), nullable=False)
    # LOW, MEDIUM, HIGH, CRITICAL
    priority = Column(String(20), nullable=False)
    # OPEN, INVESTIGATING, RESOLVED, FALSE_POSITIVE
    status = Column(String(20), default="OPEN")
    assigned_to = Column(String(255), nullable=True)  # Investigator ID
    resolution_notes = Column(Text, nullable=True)
    alert_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    model_version = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(),
                        onupdate=func.now())


class SystemMetrics(Base):
    """System performance and health metrics"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), index=True, nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # ms, mb, count, etc.
    # api, ml_service, streaming, etc.
    service_name = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow,
                       index=True, nullable=False)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata
    created_at = Column(DateTime, server_default=func.now())
