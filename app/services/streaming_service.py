"""
Kafka Streaming Service for Real-time Fraud Detection
Handles real-time transaction processing and alerting
"""
from typing import Dict, Any, List, Union, Optional
import asyncio
import json
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer

from app.models import TransactionPayload, RiskScoreResponse
from src.config import settings


class KafkaStreamingService:
    """Manages Kafka streaming for real-time fraud detection"""

    def __init__(self):
        self.producer = None
        self.consumer = None
        self.running = False
        self.topics = {
            "transactions": "fraud_detection_transactions",
            "alerts": "fraud_detection_alerts",
            "audit": "fraud_detection_audit"
        }

    def _get_producer(self) -> Union[KafkaProducer, "MockKafkaProducer"]:
        """Get or create Kafka producer"""
        if self.producer is None:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=settings.kafka_bootstrap_servers,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                    key_serializer=lambda x: x.encode('utf-8') if x else None,
                    acks='all',  # Wait for all replicas to acknowledge
                    retries=3,
                    max_in_flight_requests_per_connection=1
                )
            except Exception as e:
                print(f"Failed to create Kafka producer: {e}")
                # Return a mock producer for development
                self.producer = MockKafkaProducer()

        return self.producer

    def _get_consumer(self, topic: str, group_id: str) -> Union[KafkaConsumer, "MockKafkaConsumer"]:
        """Get or create Kafka consumer"""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            return consumer
        except Exception as e:
            print(f"Failed to create Kafka consumer: {e}")
            # Return a mock consumer for development
            return MockKafkaConsumer()

    async def publish_transaction(
        self,
        transaction: TransactionPayload,
        risk_response: RiskScoreResponse
    ) -> bool:
        """Publish transaction and risk assessment to Kafka"""

        producer = self._get_producer()

        # Prepare message payload
        message = {
            "transaction": transaction.dict(),
            "risk_assessment": risk_response.dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": "transaction_processed"
        }

        try:
            # Publish to transactions topic
            future = producer.send(
                self.topics["transactions"],
                key=transaction.transaction_id or "unknown",
                value=message
            )

            # Wait for confirmation (with timeout)
            future.get(timeout=10)

            # If high risk, also publish to alerts topic
            if risk_response.alert:
                await self.publish_alert(transaction, risk_response)

            return True

        except Exception as e:
            print(f"Failed to publish transaction: {e}")
            return False

    async def publish_alert(
        self,
        transaction: TransactionPayload,
        risk_response: RiskScoreResponse
    ) -> bool:
        """Publish fraud alert to Kafka"""

        producer = self._get_producer()

        alert_message = {
            "alert_id": f"alert_{transaction.transaction_id}",
            "transaction_id": transaction.transaction_id,
            "user_id": transaction.user_id,
            "risk_score": risk_response.risk_score,
            "risk_level": risk_response.risk_level,
            "amount": transaction.amount,
            "currency": transaction.currency,
            "merchant_id": transaction.merchant_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": "fraud_alert",
            "priority": "HIGH" if risk_response.risk_score > 0.8 else "MEDIUM"
        }

        try:
            future = producer.send(
                self.topics["alerts"],
                key=f"alert_{transaction.user_id}",
                value=alert_message
            )

            future.get(timeout=10)
            return True

        except Exception as e:
            print(f"Failed to publish alert: {e}")
            return False

    async def publish_audit_log(self, audit_data: Dict[str, Any]) -> bool:
        """Publish audit log entry to Kafka"""

        producer = self._get_producer()

        audit_message = {
            **audit_data,
            "timestamp": datetime.utcnow().isoformat(),
            "message_type": "audit_log"
        }

        try:
            future = producer.send(
                self.topics["audit"],
                key=audit_data.get("transaction_id") or "unknown",
                value=audit_message
            )

            future.get(timeout=10)
            return True

        except Exception as e:
            print(f"Failed to publish audit log: {e}")
            return False

    async def start_alert_consumer(self) -> None:
        """Start consuming fraud alerts for real-time processing"""

        consumer = self._get_consumer(
            self.topics["alerts"],
            "fraud_alert_processor"
        )

        self.running = True

        try:
            while self.running:
                # Poll for messages with timeout
                message_pack = consumer.poll(timeout_ms=1000)

                if message_pack:
                    for topic_partition, messages in message_pack.items():
                        for message in messages:
                            await self._process_alert_message(message.value)

                # Allow other coroutines to run
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Alert consumer error: {e}")
        finally:
            consumer.close()

    async def _process_alert_message(self, alert_data: Dict[str, Any]) -> None:
        """Process incoming fraud alert"""

        print(f"Processing fraud alert: {alert_data.get('alert_id')}")

        # In a real implementation, this would:
        # 1. Send notifications to security team
        # 2. Update risk models with new alert data
        # 3. Trigger additional security checks
        # 4. Log to monitoring systems

        # For now, just log the alert
        risk_score = alert_data.get('risk_score', 0)
        user_id = alert_data.get('user_id')
        amount = alert_data.get('amount', 0)

        print(
            f"FRAUD ALERT: User {user_id} - Risk Score: {risk_score:.2f} - Amount: ${amount}")

    async def get_consumer_status(self) -> str:
        """Get status of Kafka consumers"""
        if self.running:
            return "active"
        else:
            return "inactive"

    async def stop_consumers(self) -> None:
        """Stop all Kafka consumers"""
        self.running = False

        if self.consumer:
            self.consumer.close()

    def close(self) -> None:
        """Close Kafka connections"""
        if self.producer:
            self.producer.close()

        if self.consumer:
            self.consumer.close()

    async def get_topic_stats(self) -> Dict[str, Any]:
        """Get statistics about Kafka topics"""
        # In a real implementation, this would query Kafka metadata
        # For now, return mock statistics

        return {
            "topics": list(self.topics.values()),
            "producer_active": self.producer is not None,
            "consumer_active": self.running,
            "messages_sent_today": 1250,  # Mock data
            "alerts_generated_today": 45,  # Mock data
            "last_message_timestamp": datetime.utcnow().isoformat()
        }

    async def batch_publish_transactions(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Publish multiple transactions in batch"""

        producer = self._get_producer()
        successful = 0
        failed = 0

        for item in batch_data:
            try:
                future = producer.send(
                    self.topics["transactions"],
                    key=item.get("transaction_id") or "unknown",
                    value=item
                )

                future.get(timeout=5)  # Shorter timeout for batch
                successful += 1

            except Exception as e:
                print(f"Failed to publish batch item: {e}")
                failed += 1

        return {
            "total": len(batch_data),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(batch_data) if batch_data else 0
        }


class MockKafkaProducer:
    """Mock Kafka producer for development/testing"""

    def send(self, topic: str, key: Optional[str] = None, value: Any = None):
        """Mock send method"""
        print(f"Mock Kafka: Publishing to {topic} - Key: {key}")
        return MockFuture()

    def close(self):
        """Mock close method"""
        pass


class MockKafkaConsumer:
    """Mock Kafka consumer for development/testing"""

    def poll(self, timeout_ms: int = 1000):
        """Mock poll method"""
        return {}  # No messages

    def close(self):
        """Mock close method"""
        pass


class MockFuture:
    """Mock Kafka future for development/testing"""

    def get(self, timeout: int = 10):
        """Mock get method"""
        return True  # Success
