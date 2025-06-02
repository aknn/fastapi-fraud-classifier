from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings for privacy-aware fraud detection"""

    # Application Settings
    app_name: str = "Privacy-Aware Fraud Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # Database Settings
    database_url: str = "postgresql://user:password@localhost:5432/fraud_detection"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis Settings
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    # Kafka Settings
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_transactions: str = "transactions"
    kafka_topic_alerts: str = "fraud_alerts"
    kafka_consumer_group: str = "fraud_detection_group"

    # ML Model Settings
    model_version: str = "v1.0.0"
    model_confidence_threshold: float = 0.5
    high_risk_threshold: float = 0.7
    critical_risk_threshold: float = 0.9

    # Feature Store Settings
    feature_store_ttl: int = 3600  # 1 hour
    velocity_window_minutes: int = 60
    max_graph_depth: int = 3

    # Privacy Settings
    consent_default_ttl_days: int = 365
    anonymization_enabled: bool = True
    data_retention_days: int = 2555  # ~7 years for compliance

    # Security Settings
    secret_key: str = "your-secret-key-change-in-production"
    jwt_secret: str = "your-jwt-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Monitoring Settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"

    # Rate Limiting
    rate_limit_per_minute: int = 100
    burst_rate_limit: int = 200

    # Federated Learning Settings
    federated_enabled: bool = False
    federated_round_duration_minutes: int = 60
    min_clients_per_round: int = 3

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# Risk level mapping
RISK_LEVELS = {
    (0.0, 0.3): "LOW",
    (0.3, 0.5): "MEDIUM",
    (0.5, 0.7): "HIGH",
    (0.7, 1.0): "CRITICAL"
}


def get_risk_level(score: float) -> str:
    """Convert risk score to risk level string"""
    for (min_score, max_score), level in RISK_LEVELS.items():
        if min_score <= score < max_score:
            return level
    return "CRITICAL"  # Default for edge cases


# Low risk countries for rule-based filtering
LOW_RISK_COUNTRIES = [
    "US", "CA", "GB", "DE", "FR", "AU", "JP", "SG", "CH", "NL",
    "SE", "NO", "DK", "FI", "NZ", "IE", "LU", "AT", "BE"
]


# High risk merchant categories
HIGH_RISK_MERCHANT_CATEGORIES = [
    "gambling", "cryptocurrency", "money_transfer", "forex",
    "adult_entertainment", "telemarketing", "debt_collection"
]


# Feature engineering constants
VELOCITY_FEATURES = [
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_1h", "amount_sum_24h", "amount_sum_7d",
    "unique_merchants_1h", "unique_merchants_24h"
]


DEVICE_FEATURES = [
    "device_age_days", "device_txn_count", "device_avg_amount",
    "device_unique_users", "device_risk_score"
]


GRAPH_FEATURES = [
    "direct_connections", "indirect_risk_score", "network_centrality",
    "suspicious_neighbor_count", "avg_neighbor_risk"
]
