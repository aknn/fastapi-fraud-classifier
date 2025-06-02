"""
Feature Store Service for Privacy-Aware Fraud Detection
Manages feature computation, caching, and retrieval with privacy controls
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import aioredis
import json

from app.models import FeatureStoreEntry
from src.config import settings
from app.database.service import DatabaseService


class FeatureStoreService:
    """Manages feature computation and storage for fraud detection"""

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.redis_client: Optional[aioredis.Redis] = None
        self.feature_prefix = "features:"
        self.velocity_prefix = "velocity:"

    async def _get_redis_client(self) -> aioredis.Redis:
        """Get or create async Redis client"""
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(settings.redis_url)
        return self.redis_client

    async def compute_user_features(
        self,
        user_id: str,
        transaction_data: Dict[str, Any],
        historical_transactions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive user features for fraud detection"""

        if historical_transactions is None:
            # Get recent transactions from database
            db_transactions = self.db_service.get_user_transactions(
                user_id, limit=100, hours_back=24*30)  # Last 30 days
            historical_transactions = [
                {
                    "amount": t.amount,
                    "timestamp": t.timestamp,
                    "payment_type": t.payment_type,
                    "merchant_id": t.merchant_id,
                    "location": t.location,
                }
                for t in db_transactions
            ]

        features = {}

        # Basic transaction features
        features.update(self._compute_transaction_features(transaction_data))

        # Historical user behavior features
        features.update(self._compute_user_behavior_features(
            user_id, historical_transactions))

        # Velocity features
        features.update(await self._compute_velocity_features(
            user_id, transaction_data))

        # Device and location features
        features.update(
            self._compute_device_location_features(transaction_data))

        # Time-based features
        features.update(self._compute_temporal_features(transaction_data))

        return features

    def _compute_transaction_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features from current transaction"""
        features = {
            "transaction_amount": transaction_data.get("amount", 0.0),
            "transaction_currency": transaction_data.get("currency", "USD"),
            "payment_type_encoded": self._encode_payment_type(
                transaction_data.get("payment_type", "unknown")),
            # Log-like transform
            "amount_log": max(0.01, transaction_data.get("amount", 0.01))**0.5,
            "is_weekend": datetime.now().weekday() >= 5,
            "hour_of_day": datetime.now().hour,
        }

        return features

    def _compute_user_behavior_features(
        self,
        user_id: str,
        historical_transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute user behavior features from historical data"""

        if not historical_transactions:
            return {
                "avg_transaction_amount": 0.0,
                "transaction_count_30d": 0,
                "unique_merchants_30d": 0,
                "spending_variance": 0.0,
                "days_since_last_transaction": 999,
            }

        amounts = [t.get("amount", 0) for t in historical_transactions]
        merchants = set(t.get("merchant_id")
                        for t in historical_transactions if t.get("merchant_id"))

        # Calculate variance safely
        if len(amounts) > 1:
            mean_amount = sum(amounts) / len(amounts)
            variance = sum((x - mean_amount) **
                           2 for x in amounts) / len(amounts)
        else:
            variance = 0.0

        # Days since last transaction
        if historical_transactions:
            last_transaction = max(
                t.get("timestamp", datetime.min)
                for t in historical_transactions
            )
            if isinstance(last_transaction, datetime):
                days_since_last = (datetime.utcnow() - last_transaction).days
            else:
                days_since_last = 0
        else:
            days_since_last = 999

        features = {
            "avg_transaction_amount": sum(amounts) / len(amounts) if amounts else 0.0,
            "transaction_count_30d": len(historical_transactions),
            "unique_merchants_30d": len(merchants),
            "spending_variance": variance,
            "days_since_last_transaction": days_since_last,
        }

        return features

    async def _compute_velocity_features(
        self,
        user_id: str,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute velocity-based features using Redis for real-time counting"""

        redis_client = await self._get_redis_client()
        current_time = datetime.utcnow()

        # Time windows for velocity computation
        time_windows = {
            "1h": 3600,
            "6h": 21600,
            "24h": 86400,
        }

        velocity_features = {}

        for window_name, window_seconds in time_windows.items():
            # Count transactions in time window
            velocity_key = f"{self.velocity_prefix}{user_id}:{window_name}"

            # Add current transaction
            await redis_client.lpush(velocity_key, current_time.timestamp())
            await redis_client.expire(velocity_key, window_seconds)

            # Count transactions in window
            all_timestamps = await redis_client.lrange(velocity_key, 0, -1)
            cutoff_time = current_time.timestamp() - window_seconds

            recent_count = sum(
                1 for ts_bytes in all_timestamps
                if float(ts_bytes.decode()) >= cutoff_time
            )

            velocity_features[f"transaction_count_{window_name}"] = recent_count

        return velocity_features

    def _compute_device_location_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute device and location features"""
        features: Dict[str, Any] = {
            "has_device_fingerprint": bool(transaction_data.get("device_fingerprint")),
            "location_provided": bool(transaction_data.get("location")),
            "ip_address_provided": bool(transaction_data.get("ip_address")),
        }

        # Simple location encoding (in production, would use more sophisticated methods)
        location = transaction_data.get("location", "")
        if location:
            # Simple hash for categorical encoding
            features["location_hash"] = hash(location) % 1000

        return features

    def _compute_temporal_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute time-based features"""
        timestamp = transaction_data.get("timestamp", datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        features = {
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_weekend": timestamp.weekday() >= 5,
            "is_night": timestamp.hour < 6 or timestamp.hour > 22,
            "month": timestamp.month,
        }

        return features

    def _encode_payment_type(self, payment_type: str) -> int:
        """Simple encoding of payment types"""
        encoding_map = {
            "credit_card": 1,
            "debit_card": 2,
            "bank_transfer": 3,
            "digital_wallet": 4,
            "crypto": 5,
            "cash": 6,
            "unknown": 0
        }
        return encoding_map.get(payment_type.lower(), 0)

    async def cache_features(
        self,
        entity_id: str,
        entity_type: str,
        features: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> FeatureStoreEntry:
        """Cache computed features in both Redis and database"""

        # Cache in database
        # Remove unused variable
        self.db_service.cache_features(
            entity_id, entity_type, features, ttl_seconds)

        # Cache in Redis for fast access
        redis_client = await self._get_redis_client()
        cache_key = f"{self.feature_prefix}{entity_type}:{entity_id}"

        feature_entry = FeatureStoreEntry(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            last_updated=datetime.utcnow(),
            ttl_seconds=ttl_seconds
        )

        await redis_client.setex(
            cache_key,
            ttl_seconds,
            json.dumps(feature_entry.dict())
        )

        return feature_entry

    async def get_cached_features(
        self,
        entity_id: str,
        entity_type: str
    ) -> Optional[FeatureStoreEntry]:
        """Get cached features from Redis first, then database"""

        # Try Redis first
        redis_client = await self._get_redis_client()
        cache_key = f"{self.feature_prefix}{entity_type}:{entity_id}"

        cached_data = await redis_client.get(cache_key)
        if cached_data:
            try:
                feature_data = json.loads(cached_data)
                feature_entry = FeatureStoreEntry(**feature_data)
                return feature_entry
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback to database
        db_features = self.db_service.get_cached_features(
            entity_id, entity_type)
        if db_features:
            return FeatureStoreEntry(
                entity_id=entity_id,
                entity_type=entity_type,
                features=db_features,
                last_updated=datetime.utcnow(),
                ttl_seconds=3600
            )

        return None

    async def get_user_features(self, user_id: str) -> Dict[str, Any]:
        """Get cached user features or compute new ones"""
        cached_features = await self.get_cached_features(user_id, "user")

        if cached_features:
            return cached_features.features

        # Compute new features if not cached
        features = await self.compute_user_features(user_id, {})

        # Cache the computed features
        await self.cache_features(user_id, "user", features)

        return features

    async def get_device_features(self, device_fingerprint: str) -> Dict[str, Any]:
        """Get cached device features or return default"""
        cached_features = await self.get_cached_features(device_fingerprint, "device")

        if cached_features:
            return cached_features.features

        # Return default device features if not cached
        default_features = {
            "device_age_days": 30,
            "is_mobile": True,
            "browser_type": "unknown",
            "operating_system": "unknown",
            "screen_resolution": "unknown",
            "timezone": "UTC",
            "language": "en",
            "is_vpn": False,
            "device_risk_score": 0.3
        }

        # Cache the default features
        await self.cache_features(device_fingerprint, "device", default_features)

        return default_features

    async def get_merchant_features(self, merchant_id: str) -> Dict[str, Any]:
        """Get cached merchant features or return default"""
        cached_features = await self.get_cached_features(merchant_id, "merchant")

        if cached_features:
            return cached_features.features

        # Return default merchant features if not cached
        default_features = {
            "merchant_age_days": 365,
            "merchant_category": "general",
            "merchant_risk_score": 0.2,
            "average_transaction_amount": 100.0,
            "transaction_volume_30d": 1000,
            "chargeback_rate": 0.01,
            "merchant_location": "unknown",
            "is_verified": True,
            "fraud_complaints": 0
        }

        # Cache the default features
        await self.cache_features(merchant_id, "merchant", default_features)

        return default_features

    async def batch_compute_features(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[FeatureStoreEntry]:
        """Batch compute features for multiple entities"""

        results = []
        for request in requests:
            entity_id = request.get("entity_id")
            entity_type = request.get("entity_type", "user")
            transaction_data = request.get("transaction_data", {})

            if entity_type == "user" and entity_id:
                features = await self.compute_user_features(
                    entity_id, transaction_data)

                feature_entry = await self.cache_features(
                    entity_id, entity_type, features)
                results.append(feature_entry)

        return results

    async def cleanup_expired_features(self) -> int:
        """Clean up expired feature cache entries"""
        # This would be implemented as a background task
        # For now, return 0 as placeholder
        return 0
