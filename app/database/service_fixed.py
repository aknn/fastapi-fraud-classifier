"""
Database service layer for CRUD operations - Fixed version
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from .models import (
    Transaction, AuditLog, ConsentRecord, FeatureCache,
    EntityRelationship, FraudAlert, SystemMetrics
)
from ..models import (
    TransactionPayload, ConsentRequest,
    RiskScoreResponse, AuditLogEntry
)


class DatabaseService:
    """Database service for fraud detection operations"""

    def __init__(self, db: Session):
        self.db = db

    # Transaction operations
    def create_transaction(
        self,
        transaction: TransactionPayload,
        risk_response: Optional[RiskScoreResponse] = None
    ) -> Transaction:
        """Create a new transaction record"""
        db_transaction = Transaction(
            transaction_id=transaction.transaction_id or str(uuid.uuid4()),
            user_id=transaction.user_id,
            amount=transaction.amount,
            currency=transaction.currency,
            payment_type=transaction.payment_type.value,
            merchant_id=transaction.merchant_id,
            device_fingerprint=transaction.device_fingerprint,
            ip_address_hash=transaction.ip_address,  # Assume already hashed
            location=transaction.location,
            description=transaction.description,
            extra_data=transaction.metadata,  # Fixed: use extra_data instead of metadata
            timestamp=transaction.timestamp
        )

        if risk_response:
            # These are direct attribute assignments to the SQLAlchemy model
            # The values will be properly handled by SQLAlchemy ORM
            db_transaction.risk_score = risk_response.risk_score
            db_transaction.risk_level = risk_response.risk_level
            db_transaction.alert = risk_response.alert
            db_transaction.model_version = risk_response.model_version
            db_transaction.confidence = risk_response.confidence
            db_transaction.processing_time_ms = risk_response.processing_time_ms

        self.db.add(db_transaction)
        self.db.commit()
        self.db.refresh(db_transaction)
        return db_transaction

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        return self.db.query(Transaction).filter(
            Transaction.transaction_id == transaction_id
        ).first()

    def get_user_transactions(
        self,
        user_id: str,
        limit: int = 100,
        hours_back: Optional[int] = None
    ) -> List[Transaction]:
        """Get recent transactions for a user"""
        query = self.db.query(Transaction).filter(
            Transaction.user_id == user_id)

        if hours_back:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            query = query.filter(Transaction.timestamp >= cutoff_time)

        return query.order_by(desc(Transaction.timestamp)).limit(limit).all()

    # Consent operations
    def create_consent_record(self, consent: ConsentRequest) -> ConsentRecord:
        """Create a new consent record"""
        db_consent = ConsentRecord(
            user_id=consent.user_id,
            scopes=[scope.value for scope in consent.scopes],
            granted=consent.granted,
            granted_at=datetime.utcnow(),
            expires_at=consent.expires_at,
            consent_token=consent.consent_token
        )

        self.db.add(db_consent)
        self.db.commit()
        self.db.refresh(db_consent)
        return db_consent

    def get_user_consent(self, user_id: str) -> Optional[ConsentRecord]:
        """Get active consent for a user"""
        return self.db.query(ConsentRecord).filter(
            and_(
                ConsentRecord.user_id == user_id,
                # Fixed: use is_() for boolean comparison
                ConsentRecord.granted.is_(True),
                or_(
                    ConsentRecord.expires_at.is_(None),
                    ConsentRecord.expires_at > datetime.utcnow()
                ),
                ConsentRecord.revoked_at.is_(None)
            )
        ).order_by(desc(ConsentRecord.granted_at)).first()

    def revoke_consent(self, user_id: str, scopes: Optional[List[str]] = None) -> bool:
        """Revoke user consent"""
        query = self.db.query(ConsentRecord).filter(
            and_(
                ConsentRecord.user_id == user_id,
                # Fixed: use is_() for boolean comparison
                ConsentRecord.granted.is_(True),
                ConsentRecord.revoked_at.is_(None)
            )
        )

        records = query.all()
        if not records:
            return False

        for record in records:
            # Direct assignment works in SQLAlchemy ORM
            record.revoked_at = datetime.utcnow()

        self.db.commit()
        return True

    # Feature cache operations
    def cache_features(
        self,
        entity_id: str,
        entity_type: str,
        features: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> FeatureCache:
        """Cache computed features"""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        # Check if record exists and update or create new
        existing = self.db.query(FeatureCache).filter(
            and_(
                FeatureCache.entity_id == entity_id,
                FeatureCache.entity_type == entity_type
            )
        ).first()

        if existing:
            # Direct assignments work in SQLAlchemy ORM
            existing.features = features
            existing.computed_at = datetime.utcnow()
            existing.expires_at = expires_at
            db_cache = existing
        else:
            db_cache = FeatureCache(
                entity_id=entity_id,
                entity_type=entity_type,
                features=features,
                computed_at=datetime.utcnow(),
                expires_at=expires_at,
                feature_version="1.0"
            )
            self.db.add(db_cache)

        self.db.commit()
        self.db.refresh(db_cache)
        return db_cache

    def get_cached_features(
        self,
        entity_id: str,
        entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached features if not expired"""
        cache_record = self.db.query(FeatureCache).filter(
            and_(
                FeatureCache.entity_id == entity_id,
                FeatureCache.entity_type == entity_type,
                FeatureCache.expires_at > datetime.utcnow()
            )
        ).first()

        # When accessing data from SQLAlchemy models, the attributes are automatically converted
        return cache_record.features if cache_record else None

    # Graph operations
    def create_entity_relationship(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        extra_data: Optional[Dict[str, Any]] = None  # Fixed: use extra_data
    ) -> EntityRelationship:
        """Create or update entity relationship"""
        existing = self.db.query(EntityRelationship).filter(
            and_(
                EntityRelationship.source_type == source_type,
                EntityRelationship.source_id == source_id,
                EntityRelationship.target_type == target_type,
                EntityRelationship.target_id == target_id,
                EntityRelationship.relationship_type == relationship_type
            )
        ).first()

        if existing:
            # Direct assignments work in SQLAlchemy ORM
            existing.weight = weight
            existing.last_seen = datetime.utcnow()
            existing.transaction_count = existing.transaction_count + \
                1  # Fixed: explicit addition
            if extra_data:
                existing.extra_data = extra_data
            db_relationship = existing
        else:
            db_relationship = EntityRelationship(
                source_type=source_type,
                source_id=source_id,
                target_type=target_type,
                target_id=target_id,
                relationship_type=relationship_type,
                weight=weight,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                transaction_count=1,
                extra_data=extra_data or {}
            )
            self.db.add(db_relationship)

        self.db.commit()
        self.db.refresh(db_relationship)
        return db_relationship

    def get_entity_relationships(
        self,
        entity_type: str,
        entity_id: str,
        depth: int = 1
    ) -> List[EntityRelationship]:
        """Get entity relationships up to specified depth"""
        # For now, just return direct relationships (depth=1)
        # In production, would implement recursive graph traversal
        return self.db.query(EntityRelationship).filter(
            or_(
                and_(
                    EntityRelationship.source_type == entity_type,
                    EntityRelationship.source_id == entity_id
                ),
                and_(
                    EntityRelationship.target_type == entity_type,
                    EntityRelationship.target_id == entity_id
                )
            )
        ).all()

    # Audit operations
    def create_audit_log(self, audit_entry: AuditLogEntry) -> AuditLog:
        """Create audit log entry"""
        db_audit = AuditLog(
            log_id=audit_entry.log_id,
            transaction_id=audit_entry.transaction_id,
            user_id=audit_entry.user_id,
            timestamp=audit_entry.timestamp,
            model_version=audit_entry.model_version,
            input_features=audit_entry.input_features,
            risk_score=audit_entry.risk_score,
            decision=audit_entry.decision,
            processing_pipeline=audit_entry.processing_pipeline,
            consent_status=audit_entry.consent_status
        )

        self.db.add(db_audit)
        self.db.commit()
        self.db.refresh(db_audit)
        return db_audit

    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs"""
        query = self.db.query(AuditLog)

        if user_id:
            query = query.filter(AuditLog.user_id == user_id)

        return query.order_by(desc(AuditLog.timestamp)).limit(limit).all()

    # Alert operations
    def create_fraud_alert(
        self,
        transaction_id: str,
        user_id: str,
        risk_score: float,
        risk_level: str,
        alert_type: str = "FRAUD_SUSPECTED",
        model_version: str = "1.0"
    ) -> FraudAlert:
        """Create fraud alert"""
        db_alert = FraudAlert(
            alert_id=str(uuid.uuid4()),
            transaction_id=transaction_id,
            user_id=user_id,
            risk_score=risk_score,
            risk_level=risk_level,
            alert_type=alert_type,
            priority=risk_level,  # Use risk level as priority
            status="OPEN",
            alert_timestamp=datetime.utcnow(),
            model_version=model_version
        )

        self.db.add(db_alert)
        self.db.commit()
        self.db.refresh(db_alert)
        return db_alert

    def get_open_alerts(self, limit: int = 100) -> List[FraudAlert]:
        """Get open fraud alerts"""
        return self.db.query(FraudAlert).filter(
            FraudAlert.status == "OPEN"
        ).order_by(desc(FraudAlert.alert_timestamp)).limit(limit).all()

    # Metrics operations
    def record_system_metric(
        self,
        metric_name: str,
        metric_value: float,
        service_name: str,
        metric_unit: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None  # Fixed: use extra_data
    ) -> SystemMetrics:
        """Record system metric"""
        db_metric = SystemMetrics(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            service_name=service_name,
            timestamp=datetime.utcnow(),
            extra_data=extra_data or {}
        )

        self.db.add(db_metric)
        self.db.commit()
        self.db.refresh(db_metric)
        return db_metric

    def get_system_metrics(  # Fixed: removed async - not needed for sync SQLAlchemy
        self,
        service_name: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 1000
    ) -> List[SystemMetrics]:
        """Get system metrics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        query = self.db.query(SystemMetrics).filter(
            SystemMetrics.timestamp >= cutoff_time
        )

        if service_name:
            query = query.filter(SystemMetrics.service_name == service_name)

        return query.order_by(desc(SystemMetrics.timestamp)).limit(limit).all()
