"""
Kafka Streaming Service for Real-time Transaction Processing
Handles real-time transaction ingestion and alert publishing
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

from app.models import TransactionPayload, RiskScoreResponse
from src.config import settings

logger = logging.getLogger(__name__)


class KafkaStreamingService:
    """Manages Kafka streaming for real-time fraud detection"""

    def __init__(self):
        self.producer = None
        self.consumer = None
        self.running = False
        self._initialize_kafka()

    def _initialize_kafka(self):
        """Initialize Kafka producer and consumer"""
        try:
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(
                    v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=3,
                acks='all',
                enable_idempotence=True
            )

            # Initialize consumer
            self.consumer = KafkaConsumer(
                settings.kafka_topic_transactions,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_consumer_group,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True
            )

            logger.info("Kafka streaming service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            # In development, continue without Kafka
            self.producer = None
            self.consumer = None

    async def publish_transaction(self, transaction: TransactionPayload) -> bool:
        """Publish transaction to Kafka topic"""
        if not self.producer:
            logger.warning("Kafka producer not available")
            return False

        try:
            transaction_data = {
                "transaction_id": transaction.transaction_id,
                "user_id": transaction.user_id,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "payment_type": transaction.payment_type.value,
                "merchant_id": transaction.merchant_id,
                "device_fingerprint": transaction.device_fingerprint,
                "timestamp": transaction.timestamp.isoformat(),
                "metadata": transaction.metadata or {}
            }

            # Use user_id as partition key for consistent routing
            key = transaction.user_id

            future = self.producer.send(
                settings.kafka_topic_transactions,
                key=key,
                value=transaction_data
            )

            # Wait for confirmation (with timeout)
            record_metadata = future.get(timeout=10)

            logger.info(
                f"Transaction published to Kafka: topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, offset={record_metadata.offset}"
            )

            return True

        except KafkaError as e:
            logger.error(f"Failed to publish transaction to Kafka: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing transaction: {e}")
            return False

    async def publish_alert(self, risk_response: RiskScoreResponse, transaction: TransactionPayload) -> bool:
        """Publish fraud alert to Kafka topic"""
        if not self.producer or not risk_response.alert:
            return False

        try:
            alert_data = {
                "alert_id": f"alert_{risk_response.transaction_id}",
                "transaction_id": risk_response.transaction_id,
                "user_id": risk_response.user_id,
                "risk_score": risk_response.risk_score,
                "risk_level": risk_response.risk_level,
                "alert_timestamp": datetime.utcnow().isoformat(),
                "transaction_amount": transaction.amount,
                "transaction_currency": transaction.currency,
                "payment_type": transaction.payment_type.value,
                "merchant_id": transaction.merchant_id,
                "model_version": risk_response.model_version,
                "confidence": risk_response.confidence,
                "priority": self._calculate_alert_priority(risk_response.risk_score),
                "metadata": {
                    "processing_time_ms": risk_response.processing_time_ms,
                    "device_fingerprint": transaction.device_fingerprint,
                    "location": transaction.location
                }
            }

            # Use user_id as partition key
            key = risk_response.user_id

            future = self.producer.send(
                settings.kafka_topic_alerts,
                key=key,
                value=alert_data
            )

            record_metadata = future.get(timeout=10)

            logger.warning(
                f"FRAUD ALERT published: user={risk_response.user_id}, "
                f"score={risk_response.risk_score:.3f}, "
                f"level={risk_response.risk_level}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            return False

    def _calculate_alert_priority(self, risk_score: float) -> str:
        """Calculate alert priority based on risk score"""
        if risk_score >= settings.critical_risk_threshold:
            return "CRITICAL"
        elif risk_score >= settings.high_risk_threshold:
            return "HIGH"
        else:
            return "MEDIUM"

    async def start_consumer(self, callback_func=None):
        """Start consuming transactions from Kafka"""
        if not self.consumer:
            logger.warning("Kafka consumer not available")
            return

        self.running = True
        logger.info("Starting Kafka consumer...")

        try:
            while self.running:
                # Poll for messages with timeout
                message_batch = self.consumer.poll(timeout_ms=1000)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Process the transaction message
                            await self._process_transaction_message(message, callback_func)

                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            # Continue processing other messages

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.running = False
            logger.info("Kafka consumer stopped")

    async def _process_transaction_message(self, message, callback_func=None):
        """Process a single transaction message from Kafka"""
        try:
            transaction_data = message.value

            # Convert to TransactionPayload
            transaction = TransactionPayload(
                transaction_id=transaction_data.get("transaction_id"),
                user_id=transaction_data["user_id"],
                amount=transaction_data["amount"],
                currency=transaction_data["currency"],
                payment_type=transaction_data["payment_type"],
                merchant_id=transaction_data.get("merchant_id"),
                device_fingerprint=transaction_data.get("device_fingerprint"),
                ip_address=transaction_data.get("ip_address"),
                location=transaction_data.get("location"),
                timestamp=datetime.fromisoformat(
                    transaction_data["timestamp"]),
                description=transaction_data.get("description"),
                metadata=transaction_data.get("metadata", {})
            )

            # Call the provided callback function (e.g., fraud detection pipeline)
            if callback_func:
                await callback_func(transaction)

            logger.debug(
                f"Processed transaction: {transaction.transaction_id}")

        except Exception as e:
            logger.error(f"Failed to process transaction message: {e}")

    def stop_consumer(self):
        """Stop the Kafka consumer"""
        self.running = False
        logger.info("Stopping Kafka consumer...")

    async def get_consumer_status(self) -> Dict[str, Any]:
        """Get consumer status and lag information"""
        if not self.consumer:
            return {"status": "disabled", "reason": "Consumer not initialized"}

        try:
            # Get topic partitions
            partitions = self.consumer.partitions_for_topic(
                settings.kafka_topic_transactions)

            status = {
                "status": "running" if self.running else "stopped",
                "topic": settings.kafka_topic_transactions,
                "group_id": settings.kafka_consumer_group,
                "partitions": list(partitions) if partitions else [],
                "last_poll": datetime.utcnow().isoformat()
            }

            return status

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def publish_batch_transactions(self, transactions: List[TransactionPayload]) -> Dict[str, int]:
        """Publish multiple transactions in batch"""
        if not self.producer:
            return {"published": 0, "failed": len(transactions)}

        published = 0
        failed = 0

        for transaction in transactions:
            success = await self.publish_transaction(transaction)
            if success:
                published += 1
            else:
                failed += 1

        # Flush to ensure all messages are sent
        self.producer.flush()

        return {"published": published, "failed": failed}

    def __del__(self):
        """Cleanup Kafka connections"""
        try:
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
        except Exception as e:
            logger.error(f"Error closing Kafka connections: {e}")


# Global streaming service instance
streaming_service = KafkaStreamingService()
