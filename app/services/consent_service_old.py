"""
Privacy-aware consent management service
Handles user consent for data processing and privacy compliance
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import redis
from jose import jwt, JWTError
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import json

from app.models import ConsentRequest, ConsentScope
from src.config import settings


class ConsentService:
    """Manages user consent for privacy-aware fraud detection"""

    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)
        self.consent_prefix = "consent:"
        self.user_prefix = "user_consents:"

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

        # Store in Redis with expiration
        consent_key = f"{self.consent_prefix}{user_id}"
        user_consents_key = f"{self.user_prefix}{user_id}"

        consent_data = {
            "scopes": [scope.value for scope in scopes],
            "granted": True,
            "expires_at": expires_at.isoformat(),
            "consent_token": consent_token,
            "created_at": datetime.utcnow().isoformat()
        }

        # Store consent data
        self.redis_client.setex(
            consent_key,
            int((expires_at - datetime.utcnow()).total_seconds()),
            json.dumps(consent_data)
        )

        # Track user consents for audit
        self.redis_client.lpush(user_consents_key, json.dumps(consent_data))
        self.redis_client.expire(
            user_consents_key, settings.data_retention_days * 24 * 3600)

        return consent_request

    async def revoke_consent(self, user_id: str, scopes: Optional[List[ConsentScope]] = None) -> bool:
        """Revoke consent for specific scopes or all scopes"""
        consent_key = f"{self.consent_prefix}{user_id}"

        if scopes is None:
            # Revoke all consent
            result = self.redis_client.delete(consent_key)
            return result > 0

        # Revoke specific scopes
        consent_data = self.redis_client.get(consent_key)
        if not consent_data:
            return False

        consent_info = json.loads(consent_data)
        current_scopes = set(consent_info.get("scopes", []))
        revoked_scopes = set(scope.value for scope in scopes)

        remaining_scopes = current_scopes - revoked_scopes

        if not remaining_scopes:
            # No scopes left, delete entirely
            self.redis_client.delete(consent_key)
        else:
            # Update with remaining scopes
            consent_info["scopes"] = list(remaining_scopes)
            consent_info["updated_at"] = datetime.utcnow().isoformat()

            expires_at = datetime.fromisoformat(consent_info["expires_at"])
            ttl = int((expires_at - datetime.utcnow()).total_seconds())

            if ttl > 0:
                self.redis_client.setex(
                    consent_key, ttl, json.dumps(consent_info))
            else:
                self.redis_client.delete(consent_key)

        return True

    async def check_consent(self, user_id: str, required_scopes: List[ConsentScope]) -> Dict[str, bool]:
        """Check if user has granted consent for required scopes"""
        consent_key = f"{self.consent_prefix}{user_id}"
        consent_data = self.redis_client.get(consent_key)

        if not consent_data:
            return {scope.value: False for scope in required_scopes}

        consent_info = json.loads(consent_data)

        # Check if consent has expired
        expires_at = datetime.fromisoformat(consent_info["expires_at"])
        if expires_at < datetime.utcnow():
            self.redis_client.delete(consent_key)
            return {scope.value: False for scope in required_scopes}

        granted_scopes = set(consent_info.get("scopes", []))

        return {
            scope.value: scope.value in granted_scopes
            for scope in required_scopes
        }

    async def get_user_consents(self, user_id: str) -> Optional[ConsentRequest]:
        """Get current consent status for a user"""
        consent_key = f"{self.consent_prefix}{user_id}"
        consent_data = self.redis_client.get(consent_key)

        if not consent_data:
            return None

        consent_info = json.loads(consent_data)

        # Check if expired
        expires_at = datetime.fromisoformat(consent_info["expires_at"])
        if expires_at < datetime.utcnow():
            self.redis_client.delete(consent_key)
            return None

        return ConsentRequest(
            user_id=user_id,
            scopes=[ConsentScope(scope) for scope in consent_info["scopes"]],
            granted=consent_info["granted"],
            expires_at=expires_at,
            consent_token=consent_info.get("consent_token")
        )

    def _generate_consent_token(
        self,
        user_id: str,
        scopes: List[ConsentScope],
        expires_at: datetime
    ) -> str:
        """Generate a JWT token for consent verification"""
        payload = {
            "user_id": user_id,
            "scopes": [scope.value for scope in scopes],
            "exp": expires_at.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "jti": str(uuid.uuid4())  # JWT ID for uniqueness
        }

        return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)

    def verify_consent_token(self, token: str) -> Optional[Dict]:
        """Verify and decode a consent token"""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[
                                 settings.jwt_algorithm])
            return payload
        except JWTError:
            return None

    async def anonymize_user_data(self, user_id: str) -> str:
        """Anonymize user ID for privacy compliance"""
        if not settings.anonymization_enabled:
            return user_id

        # Simple hash-based anonymization (in production, use more sophisticated methods)
        import hashlib

        # Add salt for security
        salt = settings.secret_key.encode('utf-8')
        user_data = f"{user_id}{salt.decode('utf-8')}".encode('utf-8')

        anonymized_id = hashlib.sha256(user_data).hexdigest()[:16]
        return f"anon_{anonymized_id}"

    async def cleanup_expired_consents(self) -> int:
        """Clean up expired consent records (for batch processing)"""
        cleaned_count = 0

        # This would typically be run as a scheduled job
        # For now, it's a placeholder for the concept

        pattern = f"{self.consent_prefix}*"
        for key in self.redis_client.scan_iter(match=pattern):
            consent_data = self.redis_client.get(key)
            if consent_data:
                consent_info = json.loads(consent_data)
                expires_at = datetime.fromisoformat(consent_info["expires_at"])

                if expires_at < datetime.utcnow():
                    self.redis_client.delete(key)
                    cleaned_count += 1

        return cleaned_count


# Global consent service instance
consent_service = ConsentService()
