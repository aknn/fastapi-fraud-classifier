"""
Feature Store Service for Privacy-Aware Fraud Detection
Manages cached features for velocity checks, device fingerprinting, and entity relationships
"""
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import redis
import pandas as pd
from collections import defaultdict

from app.models import FeatureStoreEntry, TransactionPayload
from src.config import settings, VELOCITY_FEATURES, DEVICE_FEATURES


class FeatureStoreService:
    """Manages feature caching and computation for fraud detection"""

    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)
        self.feature_prefix = "features:"
        self.velocity_prefix = "velocity:"
        self.device_prefix = "device:"

    async def get_user_features(self, user_id: str) -> Dict[str, Any]:
        """Get cached features for a user"""
        feature_key = f"{self.feature_prefix}user:{user_id}"
        cached_data = self.redis_client.get(feature_key)

        if cached_data:
            feature_entry = FeatureStoreEntry(**json.loads(cached_data))

            # Check if cache is still valid
            if (datetime.utcnow() - feature_entry.last_updated).seconds < feature_entry.ttl_seconds:
                return feature_entry.features

        # Compute fresh features if cache miss or expired
        features = await self._compute_user_features(user_id)
        await self._cache_features("user", user_id, features)

        return features

    async def get_device_features(self, device_fingerprint: str) -> Dict[str, Any]:
        """Get cached features for a device"""
        if not device_fingerprint:
            return {}

        feature_key = f"{self.feature_prefix}device:{device_fingerprint}"
        cached_data = self.redis_client.get(feature_key)

        if cached_data:
            feature_entry = FeatureStoreEntry(**json.loads(cached_data))

            if (datetime.utcnow() - feature_entry.last_updated).seconds < feature_entry.ttl_seconds:
                return feature_entry.features

        features = await self._compute_device_features(device_fingerprint)
        await self._cache_features("device", device_fingerprint, features)

        return features

    async def get_merchant_features(self, merchant_id: str) -> Dict[str, Any]:
        """Get cached features for a merchant"""
        if not merchant_id:
            return {}

        feature_key = f"{self.feature_prefix}merchant:{merchant_id}"
        cached_data = self.redis_client.get(feature_key)

        if cached_data:
            feature_entry = FeatureStoreEntry(**json.loads(cached_data))

            if (datetime.utcnow() - feature_entry.last_updated).seconds < feature_entry.ttl_seconds:
                return feature_entry.features

        features = await self._compute_merchant_features(merchant_id)
        await self._cache_features("merchant", merchant_id, features)

        return features

    async def update_transaction_features(self, transaction: TransactionPayload) -> None:
        """Update features based on new transaction"""
        current_time = datetime.utcnow()

        # Update velocity features
        await self._update_velocity_features(transaction, current_time)

        # Update device features
        if transaction.device_fingerprint:
            await self._update_device_features(transaction, current_time)

        # Update merchant features
        if transaction.merchant_id:
            await self._update_merchant_features(transaction, current_time)

        # Invalidate cached features to force recomputation
        await self._invalidate_related_features(transaction)

    async def compute_velocity_features(self, user_id: str, timestamp: datetime) -> Dict[str, float]:
        """Compute velocity features for fraud detection"""
        features = {}

        # Define time windows
        windows = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7)
        }

        for window_name, window_duration in windows.items():
            start_time = timestamp - window_duration

            # Get transactions in window
            txn_data = await self._get_user_transactions_in_window(user_id, start_time, timestamp)

            if txn_data:
                features[f"txn_count_{window_name}"] = len(txn_data)
                features[f"amount_sum_{window_name}"] = sum(
                    t["amount"] for t in txn_data)
                features[f"amount_avg_{window_name}"] = features[f"amount_sum_{window_name}"] / len(
                    txn_data)
                features[f"unique_merchants_{window_name}"] = len(
                    set(t.get("merchant_id") for t in txn_data if t.get("merchant_id")))
                features[f"amount_std_{window_name}"] = self._calculate_std(
                    [t["amount"] for t in txn_data])
            else:
                features[f"txn_count_{window_name}"] = 0
                features[f"amount_sum_{window_name}"] = 0.0
                features[f"amount_avg_{window_name}"] = 0.0
                features[f"unique_merchants_{window_name}"] = 0
                features[f"amount_std_{window_name}"] = 0.0

        return features

    async def _compute_user_features(self, user_id: str) -> Dict[str, Any]:
        """Compute comprehensive features for a user"""
        current_time = datetime.utcnow()

        # Velocity features
        velocity_features = await self.compute_velocity_features(user_id, current_time)

        # Historical features
        historical_features = await self._compute_historical_features(user_id)

        # Risk patterns
        risk_features = await self._compute_risk_pattern_features(user_id)

        return {
            **velocity_features,
            **historical_features,
            **risk_features,
            "feature_computation_time": current_time.isoformat()
        }

    async def _compute_device_features(self, device_fingerprint: str) -> Dict[str, Any]:
        """Compute features for a device"""
        # Get device transaction history
        device_data = await self._get_device_history(device_fingerprint)

        if not device_data:
            return {
                "device_age_days": 0,
                "device_txn_count": 0,
                "device_avg_amount": 0.0,
                "device_unique_users": 0,
                "device_risk_score": 0.5  # neutral risk for new devices
            }

        # Calculate device features
        first_seen = min(d["timestamp"] for d in device_data)
        device_age = (datetime.utcnow() -
                      datetime.fromisoformat(first_seen)).days

        amounts = [d["amount"] for d in device_data]
        unique_users = len(set(d["user_id"] for d in device_data))

        # Device risk score based on usage patterns
        risk_indicators = [
            unique_users > 5,  # Many users on same device
            len(amounts) > 100,  # High transaction volume
            max(amounts) > 10000 if amounts else False,  # Large transactions
            device_age < 7  # Very new device
        ]
        risk_score = 0.2 + (sum(risk_indicators) * 0.2)  # 0.2 to 1.0 scale

        return {
            "device_age_days": device_age,
            "device_txn_count": len(device_data),
            "device_avg_amount": sum(amounts) / len(amounts) if amounts else 0.0,
            "device_unique_users": unique_users,
            "device_risk_score": min(risk_score, 1.0)
        }

    async def _compute_merchant_features(self, merchant_id: str) -> Dict[str, Any]:
        """Compute features for a merchant"""
        merchant_data = await self._get_merchant_history(merchant_id)

        if not merchant_data:
            return {
                "merchant_txn_count": 0,
                "merchant_avg_amount": 0.0,
                "merchant_unique_users": 0,
                "merchant_risk_score": 0.5
            }

        amounts = [d["amount"] for d in merchant_data]
        unique_users = len(set(d["user_id"] for d in merchant_data))

        return {
            "merchant_txn_count": len(merchant_data),
            "merchant_avg_amount": sum(amounts) / len(amounts) if amounts else 0.0,
            "merchant_unique_users": unique_users,
            "merchant_risk_score": 0.3  # Placeholder risk score
        }

    async def _compute_historical_features(self, user_id: str) -> Dict[str, Any]:
        """Compute long-term historical features"""
        # Get last 30 days of data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)

        txn_data = await self._get_user_transactions_in_window(user_id, start_time, end_time)

        if not txn_data:
            return {
                "avg_daily_transactions": 0.0,
                "avg_daily_amount": 0.0,
                "preferred_payment_type": "unknown",
                "weekend_transaction_ratio": 0.0
            }

        # Group by day
        daily_stats = defaultdict(lambda: {"count": 0, "amount": 0.0})
        payment_types = defaultdict(int)
        weekend_count = 0

        for txn in txn_data:
            date_key = datetime.fromisoformat(txn["timestamp"]).date()
            daily_stats[date_key]["count"] += 1
            daily_stats[date_key]["amount"] += txn["amount"]

            payment_types[txn.get("payment_type", "unknown")] += 1

            # Check if weekend
            weekday = datetime.fromisoformat(txn["timestamp"]).weekday()
            if weekday >= 5:  # Saturday or Sunday
                weekend_count += 1

        days_active = len(daily_stats)
        avg_daily_txns = len(txn_data) / max(days_active, 1)
        avg_daily_amount = sum(d["amount"]
                               for d in daily_stats.values()) / max(days_active, 1)

        preferred_payment = max(payment_types.items(), key=lambda x: x[1])[
            0] if payment_types else "unknown"
        weekend_ratio = weekend_count / len(txn_data) if txn_data else 0.0

        return {
            "avg_daily_transactions": avg_daily_txns,
            "avg_daily_amount": avg_daily_amount,
            "preferred_payment_type": preferred_payment,
            "weekend_transaction_ratio": weekend_ratio
        }

    async def _compute_risk_pattern_features(self, user_id: str) -> Dict[str, Any]:
        """Compute risk pattern indicators"""
        # This is a simplified version - in production, these would be more sophisticated

        return {
            "has_high_value_transactions": False,  # Placeholder
            "unusual_timing_pattern": False,  # Placeholder
            "geographic_anomaly": False,  # Placeholder
            "rapid_succession_transactions": False  # Placeholder
        }

    async def _cache_features(self, entity_type: str, entity_id: str, features: Dict[str, Any]) -> None:
        """Cache computed features"""
        feature_entry = FeatureStoreEntry(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            last_updated=datetime.utcnow(),
            ttl_seconds=settings.feature_store_ttl
        )

        feature_key = f"{self.feature_prefix}{entity_type}:{entity_id}"

        self.redis_client.setex(
            feature_key,
            settings.feature_store_ttl,
            json.dumps(feature_entry.dict(), default=str)
        )

    async def _invalidate_related_features(self, transaction: TransactionPayload) -> None:
        """Invalidate cached features that need recomputation"""
        keys_to_delete = [
            f"{self.feature_prefix}user:{transaction.user_id}"
        ]

        if transaction.device_fingerprint:
            keys_to_delete.append(
                f"{self.feature_prefix}device:{transaction.device_fingerprint}")

        if transaction.merchant_id:
            keys_to_delete.append(
                f"{self.feature_prefix}merchant:{transaction.merchant_id}")

        for key in keys_to_delete:
            self.redis_client.delete(key)

    async def _update_velocity_features(self, transaction: TransactionPayload, timestamp: datetime) -> None:
        """Update velocity tracking for a user"""
        velocity_key = f"{self.velocity_prefix}{transaction.user_id}"

        # Store transaction data for velocity computation
        txn_data = {
            "timestamp": timestamp.isoformat(),
            "amount": transaction.amount,
            "merchant_id": transaction.merchant_id,
            "payment_type": transaction.payment_type.value
        }

        # Add to sorted set with timestamp as score for easy range queries
        self.redis_client.zadd(
            velocity_key,
            {json.dumps(txn_data): timestamp.timestamp()}
        )

        # Keep only last 7 days of data
        cutoff_time = timestamp - timedelta(days=7)
        self.redis_client.zremrangebyscore(
            velocity_key, 0, cutoff_time.timestamp())

        # Set expiration
        self.redis_client.expire(velocity_key, 7 * 24 * 3600)  # 7 days

    async def _update_device_features(self, transaction: TransactionPayload, timestamp: datetime) -> None:
        """Update device tracking"""
        device_key = f"{self.device_prefix}{transaction.device_fingerprint}"

        device_data = {
            "user_id": transaction.user_id,
            "timestamp": timestamp.isoformat(),
            "amount": transaction.amount
        }

        self.redis_client.lpush(device_key, json.dumps(device_data))
        # Keep last 1000 transactions
        self.redis_client.ltrim(device_key, 0, 999)
        self.redis_client.expire(device_key, 30 * 24 * 3600)  # 30 days

    async def _update_merchant_features(self, transaction: TransactionPayload, timestamp: datetime) -> None:
        """Update merchant tracking"""
        merchant_key = f"merchant:{transaction.merchant_id}"

        merchant_data = {
            "user_id": transaction.user_id,
            "timestamp": timestamp.isoformat(),
            "amount": transaction.amount
        }

        self.redis_client.lpush(merchant_key, json.dumps(merchant_data))
        # Keep last 10k transactions
        self.redis_client.ltrim(merchant_key, 0, 9999)
        self.redis_client.expire(merchant_key, 90 * 24 * 3600)  # 90 days

    async def _get_user_transactions_in_window(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get user transactions within a time window"""
        velocity_key = f"{self.velocity_prefix}{user_id}"

        # Get transactions in time range
        raw_data = self.redis_client.zrangebyscore(
            velocity_key,
            start_time.timestamp(),
            end_time.timestamp()
        )

        return [json.loads(data) for data in raw_data]

    async def _get_device_history(self, device_fingerprint: str) -> List[Dict[str, Any]]:
        """Get device transaction history"""
        device_key = f"{self.device_prefix}{device_fingerprint}"
        raw_data = self.redis_client.lrange(device_key, 0, -1)

        return [json.loads(data) for data in raw_data]

    async def _get_merchant_history(self, merchant_id: str) -> List[Dict[str, Any]]:
        """Get merchant transaction history"""
        merchant_key = f"merchant:{merchant_id}"
        raw_data = self.redis_client.lrange(merchant_key, 0, -1)

        return [json.loads(data) for data in raw_data]

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


# Global feature store service instance
feature_store = FeatureStoreService()
