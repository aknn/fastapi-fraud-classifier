"""
Privacy-aware consent management service
Handles user consent for data processing and privacy compliance
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import aioredis
from jose import jwt, JWTError
import json

from app.models import ConsentRequest, ConsentScope
from src.config import settings
from app.database.service import DatabaseService


class ConsentService:
    """Manages user consent for privacy-aware fraud detection"""

    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.redis_client: Optional[aioredis.Redis] = None
        self.consent_prefix = "consent:"
        self.user_prefix = "user_consents:"

    async def _get_redis_client(self) -> aioredis.Redis:
        """Get or create async Redis client"""
        if self.redis_client is None:
            self.redis_client = await aioredis.from_url(settings.redis_url)
        return self.redis_client

    async def grant_consent(
        self,
        user_id: str,
        scopes: List[ConsentScope],
        expires_at: Optional[datetime] = None
    ) -> ConsentRequest:
        """Grant consent for specific data processing scopes"""

        if expires_at is None:
            expires_at = datetime.utcnow() + timedelta(days=settings.consent_default_ttl_days)

        # Generate consent token
        consent_token = self._generate_consent_token(
            user_id, scopes, expires_at)

        consent_request = ConsentRequest(
            user_id=user_id,
            scopes=scopes,
            granted=True,
            expires_at=expires_at,
            consent_token=consent_token
        )

        # Store in database
        self.db_service.create_consent_record(consent_request)

        # Store in Redis with expiration
        redis_client = await self._get_redis_client()
        consent_key = f"{self.consent_prefix}{user_id}"
        user_consents_key = f"{self.user_prefix}{user_id}"

        # Store consent with expiration
        consent_data = {
            "user_id": user_id,
            "scopes": [scope.value for scope in scopes],
            "granted": True,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "consent_token": consent_token
        }

        ttl_seconds = int((expires_at - datetime.utcnow()).total_seconds()
                          ) if expires_at else 86400 * settings.consent_default_ttl_days
        await redis_client.setex(consent_key, ttl_seconds, json.dumps(consent_data))

        # Add to user consent list
        await redis_client.lpush(user_consents_key, json.dumps(consent_data))
        await redis_client.expire(user_consents_key, ttl_seconds)

        return consent_request

    async def revoke_consent(
        self,
        user_id: str,
        scopes: Optional[List[ConsentScope]] = None
    ) -> bool:
        """Revoke user consent for specified scopes or all scopes"""

        # Revoke in database
        scope_values = [scope.value for scope in scopes] if scopes else None
        db_result = self.db_service.revoke_consent(user_id, scope_values)

        # Revoke in Redis
        redis_client = await self._get_redis_client()
        consent_key = f"{self.consent_prefix}{user_id}"

        result = await redis_client.delete(consent_key)
        return result > 0 or db_result

    async def check_consent(
        self,
        user_id: str,
        required_scopes: List[ConsentScope]
    ) -> Dict[str, bool]:
        """Check if user has granted consent for required scopes"""

        # First check Redis for fast access
        redis_client = await self._get_redis_client()
        consent_key = f"{self.consent_prefix}{user_id}"

        consent_data = await redis_client.get(consent_key)

        if consent_data:
            try:
                consent_info = json.loads(consent_data)
                granted_scopes = set(consent_info.get("scopes", []))

                return {
                    scope.value: scope.value in granted_scopes
                    for scope in required_scopes
                }
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback to database
        db_consent = self.db_service.get_user_consent(user_id)
        if db_consent and db_consent.granted:
            granted_scopes = set(db_consent.scopes or [])
            return {
                scope.value: scope.value in granted_scopes
                for scope in required_scopes
            }

        # No consent found
        return {scope.value: False for scope in required_scopes}

    async def get_user_consents(self, user_id: str) -> List[Dict[str, any]]:
        """Get all consent records for a user"""

        redis_client = await self._get_redis_client()
        user_consents_key = f"{self.user_prefix}{user_id}"

        # Try Redis first
        consent_list = await redis_client.lrange(user_consents_key, 0, -1)

        if consent_list:
            try:
                return [json.loads(consent_data) for consent_data in consent_list]
            except json.JSONDecodeError:
                pass

        # Fallback to database (simplified - would need query for all user consents)
        db_consent = self.db_service.get_user_consent(user_id)
        if db_consent:
            return [{
                "user_id": db_consent.user_id,
                "scopes": db_consent.scopes,
                "granted": db_consent.granted,
                "expires_at": db_consent.expires_at.isoformat() if db_consent.expires_at else None,
                "consent_token": db_consent.consent_token
            }]

        return []

    def _generate_consent_token(
        self,
        user_id: str,
        scopes: List[ConsentScope],
        expires_at: Optional[datetime]
    ) -> str:
        """Generate JWT token for consent"""
        payload = {
            "user_id": user_id,
            "scopes": [scope.value for scope in scopes],
            "iat": datetime.utcnow(),
            "exp": expires_at or datetime.utcnow() + timedelta(days=settings.consent_default_ttl_days)
        }

        return jwt.encode(
            payload,
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm
        )

    def verify_consent_token(self, token: str) -> Optional[Dict[str, any]]:
        """Verify and decode consent token"""
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )
            return payload
        except JWTError:
            return None

    async def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records"""
        # This would be run as a background task
        # For now, just return 0 as placeholder
        return 0
