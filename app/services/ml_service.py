"""
Machine Learning Service for Privacy-Aware Fraud Detection
Handles model inference, explanations, and federated learning concepts
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

from app.models import RiskScoreResponse, ExplanationResponse, FederatedTrainingUpdate, TransactionPayload
from src.config import settings


class MLModelService:
    """Manages ML models for fraud detection with privacy awareness"""

    def __init__(self):
        # Initialize with mock models - in production these would be loaded from files
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.model_version = settings.model_version
        self.model_metadata = {
            "trained_at": datetime.utcnow(),
            "training_samples": 10000,  # Mock data
            "model_type": "RandomForest",
            "features": []
        }

        # Initialize mock model
        self._initialize_mock_model()

    def _initialize_mock_model(self):
        """Initialize a mock model for demonstration"""
        # Create a simple mock model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        self.scaler = StandardScaler()

        # Mock feature names (these would match the actual features from feature store)
        self.feature_names = [
            "transaction_amount", "amount_log", "hour_of_day", "day_of_week",
            "is_weekend", "is_night", "payment_type_encoded", "avg_transaction_amount",
            "transaction_count_30d", "unique_merchants_30d", "spending_variance",
            "days_since_last_transaction", "transaction_count_1h", "transaction_count_6h",
            "transaction_count_24h", "has_device_fingerprint", "location_provided",
            "ip_address_provided", "location_hash"
        ]

        # Create mock training data to fit the model
        np.random.seed(42)
        n_samples = 1000
        X_mock = np.random.randn(n_samples, len(self.feature_names))

        # Create mock labels (fraud/not fraud)
        # Higher transaction amounts and unusual hours have higher fraud probability
        fraud_probability = (
            (X_mock[:, 0] > 2) |  # High transaction amount
            (X_mock[:, 2] < 2) |  # Very early hours
            (X_mock[:, 2] > 22)   # Very late hours
        ).astype(float)
        y_mock = (np.random.random(n_samples) <
                  fraud_probability * 0.3).astype(int)

        # Fit the mock model
        X_scaled = self.scaler.fit_transform(X_mock)
        self.model.fit(X_scaled, y_mock)

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    def predict_risk(
        self,
        features: Dict[str, Any],
        user_id: str,
        transaction_id: str
    ) -> RiskScoreResponse:
        """Predict fraud risk for a transaction"""

        start_time = datetime.utcnow()

        # Ensure all expected features are present
        feature_vector = self._prepare_feature_vector(features)

        if self.model is None or self.scaler is None:
            # Fallback to rule-based scoring
            return self._rule_based_scoring(features, user_id, transaction_id, start_time)

        try:
            # Scale features
            feature_vector_scaled = self.scaler.transform(
                feature_vector.reshape(1, -1))

            # Predict fraud probability
            fraud_probability = self.model.predict_proba(
                feature_vector_scaled)[0, 1]

            # Calculate processing time
            processing_time = (datetime.utcnow() -
                               start_time).total_seconds() * 1000

            # Determine risk level and alert status
            risk_level = self._determine_risk_level(fraud_probability)
            alert = fraud_probability >= settings.high_risk_threshold

            return RiskScoreResponse(
                transaction_id=transaction_id,
                user_id=user_id,
                risk_score=float(fraud_probability),
                risk_level=risk_level,
                confidence=0.85,  # Mock confidence score
                alert=alert,
                model_version=self.model_version,
                processing_time_ms=processing_time
            )

        except Exception as e:
            # Fallback to rule-based scoring on error
            print(f"ML model error: {e}")
            return self._rule_based_scoring(features, user_id, transaction_id, start_time)

    def explain_prediction(
        self,
        features: Dict[str, Any],
        transaction_id: str
    ) -> ExplanationResponse:
        """Generate explanation for fraud prediction using SHAP"""

        feature_vector = self._prepare_feature_vector(features)

        if self.explainer is None or self.scaler is None:
            # Return mock explanation
            return self._mock_explanation(features, transaction_id)

        try:
            # Scale features
            feature_vector_scaled = self.scaler.transform(
                feature_vector.reshape(1, -1))

            # Compute SHAP values
            shap_values = self.explainer.shap_values(feature_vector_scaled)

            # For binary classification, use the positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class

            # Create feature importance explanations
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(shap_values[0]):
                    feature_importance[feature_name] = float(shap_values[0][i])

            # Sort by absolute importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Generate human-readable explanations
            explanations = []
            for feature, importance in sorted_features[:5]:  # Top 5 features
                direction = "increases" if importance > 0 else "decreases"
                explanations.append(
                    f"Feature '{feature}' {direction} fraud risk by {abs(importance):.3f}"
                )

            return ExplanationResponse(
                transaction_id=transaction_id,
                feature_importance=dict(sorted_features),
                top_risk_factors=[feature for feature,
                                  _ in sorted_features[:5]],
                explanation_text="; ".join(explanations)
            )

        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return self._mock_explanation(features, transaction_id)

    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy array in expected order"""
        feature_vector = np.zeros(len(self.feature_names))

        for i, feature_name in enumerate(self.feature_names):
            if feature_name in features:
                value = features[feature_name]
                # Handle different data types
                if isinstance(value, bool):
                    feature_vector[i] = float(value)
                elif isinstance(value, (int, float)):
                    feature_vector[i] = float(value)
                else:
                    feature_vector[i] = 0.0  # Default for non-numeric features
            else:
                feature_vector[i] = 0.0  # Default value for missing features

        return feature_vector

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on risk score"""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.5:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "VERY_LOW"

    def _rule_based_scoring(
        self,
        features: Dict[str, Any],
        user_id: str,
        transaction_id: str,
        start_time: datetime
    ) -> RiskScoreResponse:
        """Fallback rule-based fraud scoring"""

        risk_score = 0.0

        # High amount transactions are riskier
        amount = features.get("transaction_amount", 0)
        if amount > 10000:
            risk_score += 0.4
        elif amount > 1000:
            risk_score += 0.2

        # Night time transactions are riskier
        if features.get("is_night", False):
            risk_score += 0.2

        # High velocity is risky
        if features.get("transaction_count_1h", 0) > 5:
            risk_score += 0.3

        # No device fingerprint is risky
        if not features.get("has_device_fingerprint", False):
            risk_score += 0.1

        # Cap at 1.0
        risk_score = min(risk_score, 1.0)

        processing_time = (datetime.utcnow() -
                           start_time).total_seconds() * 1000

        return RiskScoreResponse(
            transaction_id=transaction_id,
            user_id=user_id,
            risk_score=risk_score,
            risk_level=self._determine_risk_level(risk_score),
            confidence=0.7,  # Lower confidence for rule-based
            alert=risk_score >= settings.high_risk_threshold,
            model_version="rule_based_v1.0",
            processing_time_ms=processing_time
        )

    def _mock_explanation(self, features: Dict[str, Any], transaction_id: str) -> ExplanationResponse:
        """Generate mock explanation when SHAP is not available"""

        explanations = [
            "Transaction amount contributes to risk assessment",
            "Time of day affects fraud probability",
            "User transaction history influences scoring",
            "Device information impacts risk calculation"
        ]

        # Create mock feature importance
        feature_importance = {}
        for key in features.keys():
            # Assign random but consistent importance
            importance = hash(key + transaction_id) % 100 / 100.0
            feature_importance[key] = importance

        return ExplanationResponse(
            transaction_id=transaction_id,
            feature_importance=feature_importance,
            top_risk_factors=list(sorted(feature_importance.keys(), key=lambda k: abs(
                feature_importance[k]), reverse=True)[:5]),
            explanation_text="; ".join(explanations)
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_version": self.model_version,
            "model_type": self.model_metadata.get("model_type", "unknown"),
            "features_count": len(self.feature_names),
            "trained_at": self.model_metadata.get("trained_at", datetime.utcnow()).isoformat(),
            "is_trained": self.model is not None
        }

    async def update_model_federated(
        self,
        update: FederatedTrainingUpdate
    ) -> Dict[str, Any]:
        """Mock federated learning update"""
        # In a real implementation, this would:
        # 1. Validate the federated update
        # 2. Aggregate with other updates
        # 3. Update the global model
        # 4. Preserve privacy through differential privacy or secure aggregation

        return {
            "update_id": update.client_id,  # Use client_id as update_id
            "status": "accepted",
            "new_model_version": f"{self.model_version}_federated",
            # Use training_samples as participant count
            "participants": update.training_samples,
            "accuracy_improvement": 0.02  # Mock improvement
        }

    def batch_predict(
        self,
        batch_features: List[Dict[str, Any]],
        transaction_ids: List[str],
        user_ids: List[str]
    ) -> List[RiskScoreResponse]:
        """Batch prediction for multiple transactions"""

        results = []
        for features, tx_id, user_id in zip(batch_features, transaction_ids, user_ids):
            result = self.predict_risk(features, user_id, tx_id)
            results.append(result)

        return results

    async def predict_fraud_probability(
        self,
        transaction: 'TransactionPayload',
        user_features: Dict[str, Any],
        device_features: Dict[str, Any],
        merchant_features: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Predict fraud probability and return feature importance"""

        # Combine all features
        combined_features = {
            **user_features,
            **device_features,
            **merchant_features,
            "amount": transaction.amount,
            "payment_type": transaction.payment_type.value,
            "hour_of_day": transaction.timestamp.hour,
            "is_weekend": transaction.timestamp.weekday() >= 5
        }

        # Use existing predict_risk method
        risk_response = self.predict_risk(
            combined_features, transaction.user_id, str(transaction.transaction_id))

        # Generate mock feature importance for SHAP explanations
        feature_importance = {
            "amount": 0.25,
            "payment_type": 0.15,
            "hour_of_day": 0.10,
            "user_avg_amount": 0.20,
            "merchant_risk_score": 0.15,
            "device_risk_score": 0.10,
            "is_weekend": 0.05
        }

        # Add importance for available features
        for key in combined_features.keys():
            if key not in feature_importance:
                feature_importance[key] = 0.02

        return risk_response.risk_score, feature_importance

    async def generate_explanation(
        self,
        transaction_id: str,
        transaction: 'TransactionPayload',
        feature_importance: Dict[str, float],
        fraud_probability: float
    ) -> 'ExplanationResponse':
        """Generate explanation for a fraud prediction"""

        # Use existing explain_prediction method
        combined_features = {
            "amount": transaction.amount,
            "payment_type": transaction.payment_type.value,
            "hour_of_day": transaction.timestamp.hour,
            "user_id": transaction.user_id
        }

        explanation = self.explain_prediction(
            combined_features, transaction_id)

        # Update the explanation with the provided feature importance
        explanation.feature_importance = feature_importance
        # Note: ExplanationResponse doesn't have risk_score field, so we update explanation_text instead
        explanation.explanation_text = f"Fraud probability: {fraud_probability:.2%}. " + \
            explanation.explanation_text

        return explanation
