"""
Machine Learning Service for Privacy-Aware Fraud Detection
Handles model inference, SHAP explanations, and model versioning
"""
import json
import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

from app.models import TransactionPayload, ExplanationResponse
from src.config import settings


class MLModelService:
    """Manages ML model inference and explanations"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.explainer = None
        self.feature_names = []
        self.model_version = settings.model_version
        self._initialize_model()

    def _initialize_model(self):
        """Initialize a simple fraud detection model"""
        # In production, this would load a trained model from storage
        # For now, we'll create a simple mock model

        self.feature_names = [
            "amount", "hour", "day_of_week", "payment_type_encoded",
            "txn_count_1h", "txn_count_24h", "amount_sum_1h", "amount_sum_24h",
            "device_age_days", "device_risk_score", "merchant_risk_score",
            "velocity_score", "location_risk", "time_risk"
        ]

        # Create a simple Random Forest model
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()

        # Generate some mock training data to fit the model
        np.random.seed(42)
        X_mock = np.random.randn(1000, len(self.feature_names))
        # Create realistic fraud patterns
        y_mock = (
            (X_mock[:, 0] > 2) |  # High amounts
            (X_mock[:, 4] > 1.5) |  # High velocity
            (X_mock[:, 9] > 1.0)   # High device risk
        ).astype(int)

        # Fit the model
        X_scaled = self.scaler.fit_transform(X_mock)
        self.model.fit(X_scaled, y_mock)

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

    async def predict_fraud_probability(
        self,
        transaction: TransactionPayload,
        user_features: Dict[str, Any],
        device_features: Dict[str, Any],
        merchant_features: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Predict fraud probability and return feature importance"""

        # Extract features from transaction and cached features
        features = self._extract_features(
            transaction, user_features, device_features, merchant_features
        )

        # Convert to numpy array
        feature_vector = np.array([features]).reshape(1, -1)

        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Get prediction probability
        fraud_probability = self.model.predict_proba(
            feature_vector_scaled)[0][1]

        # Get SHAP values for explanation
        shap_values = self.explainer.shap_values(feature_vector_scaled)

        # Convert SHAP values to feature importance dict
        if isinstance(shap_values, list):
            # For binary classification, take the positive class SHAP values
            shap_values_positive = shap_values[1][0]
        else:
            shap_values_positive = shap_values[0]

        feature_importance = {
            name: float(value)
            for name, value in zip(self.feature_names, shap_values_positive)
        }

        return fraud_probability, feature_importance

    def _extract_features(
        self,
        transaction: TransactionPayload,
        user_features: Dict[str, Any],
        device_features: Dict[str, Any],
        merchant_features: Dict[str, Any]
    ) -> List[float]:
        """Extract feature vector from transaction and cached features"""

        # Basic transaction features
        amount = float(transaction.amount)
        hour = float(transaction.timestamp.hour)
        day_of_week = float(transaction.timestamp.weekday())

        # Encode payment type
        payment_type_mapping = {
            "card": 1.0, "bank_transfer": 2.0, "digital_wallet": 3.0,
            "crypto": 4.0, "cash": 5.0
        }
        payment_type_encoded = payment_type_mapping.get(
            transaction.payment_type.value, 0.0)

        # User velocity features
        txn_count_1h = float(user_features.get("txn_count_1h", 0))
        txn_count_24h = float(user_features.get("txn_count_24h", 0))
        amount_sum_1h = float(user_features.get("amount_sum_1h", 0))
        amount_sum_24h = float(user_features.get("amount_sum_24h", 0))

        # Device features
        device_age_days = float(device_features.get("device_age_days", 0))
        device_risk_score = float(
            device_features.get("device_risk_score", 0.5))

        # Merchant features
        merchant_risk_score = float(
            merchant_features.get("merchant_risk_score", 0.3))

        # Derived features
        velocity_score = (txn_count_1h * 0.6 + txn_count_24h * 0.4) / 10.0
        location_risk = self._calculate_location_risk(transaction.location)
        time_risk = self._calculate_time_risk(hour, day_of_week)

        return [
            amount, hour, day_of_week, payment_type_encoded,
            txn_count_1h, txn_count_24h, amount_sum_1h, amount_sum_24h,
            device_age_days, device_risk_score, merchant_risk_score,
            velocity_score, location_risk, time_risk
        ]

    def _calculate_location_risk(self, location: Optional[str]) -> float:
        """Calculate risk score based on location"""
        if not location:
            return 0.5  # Neutral risk for unknown location

        # High-risk locations (simplified)
        high_risk_locations = ["unknown", "tor", "vpn", "offshore"]

        for risk_location in high_risk_locations:
            if risk_location in location.lower():
                return 0.8

        return 0.2  # Low risk for normal locations

    def _calculate_time_risk(self, hour: float, day_of_week: float) -> float:
        """Calculate risk score based on time patterns"""
        # Higher risk for night hours (11 PM - 5 AM)
        if hour >= 23 or hour <= 5:
            return 0.7

        # Higher risk for weekends (Friday evening, Saturday, Sunday)
        if day_of_week >= 5 or (day_of_week == 4 and hour >= 18):
            return 0.6

        return 0.3  # Normal business hours

    async def generate_explanation(
        self,
        transaction_id: str,
        transaction: TransactionPayload,
        feature_importance: Dict[str, float],
        fraud_probability: float
    ) -> ExplanationResponse:
        """Generate human-readable explanation using SHAP values"""

        # Sort features by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Get top risk factors
        top_risk_factors = []
        for feature_name, importance in sorted_features[:5]:
            if importance > 0.1:  # Only significant positive contributions
                explanation = self._get_feature_explanation(
                    feature_name, importance, transaction)
                if explanation:
                    top_risk_factors.append(explanation)

        # Generate overall explanation text
        if fraud_probability > 0.7:
            risk_level = "HIGH"
        elif fraud_probability > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        explanation_text = f"{risk_level} risk transaction (score: {fraud_probability:.2f}). "

        if top_risk_factors:
            explanation_text += f"Key factors: {', '.join(top_risk_factors[:3])}."
        else:
            explanation_text += "No significant risk factors identified."

        return ExplanationResponse(
            transaction_id=transaction_id,
            feature_importance=feature_importance,
            shap_values=feature_importance,  # Same as feature importance for simplicity
            top_risk_factors=top_risk_factors,
            explanation_text=explanation_text
        )

    def _get_feature_explanation(
        self,
        feature_name: str,
        importance: float,
        transaction: TransactionPayload
    ) -> Optional[str]:
        """Convert feature importance to human-readable explanation"""

        explanations = {
            "amount": f"Transaction amount (${transaction.amount:,.2f}) is {'high' if importance > 0 else 'low'}",
            "hour": f"Transaction time ({transaction.timestamp.hour}:00) is {'unusual' if importance > 0 else 'normal'}",
            "day_of_week": f"Day of week pattern is {'suspicious' if importance > 0 else 'normal'}",
            "txn_count_1h": f"Recent transaction frequency is {'high' if importance > 0 else 'normal'}",
            "txn_count_24h": f"Daily transaction frequency is {'high' if importance > 0 else 'normal'}",
            "amount_sum_1h": f"Recent transaction volume is {'high' if importance > 0 else 'normal'}",
            "amount_sum_24h": f"Daily transaction volume is {'high' if importance > 0 else 'normal'}",
            "device_age_days": f"Device age is {'very new' if importance > 0 else 'established'}",
            "device_risk_score": f"Device risk profile is {'high' if importance > 0 else 'low'}",
            "merchant_risk_score": f"Merchant risk profile is {'elevated' if importance > 0 else 'normal'}",
            "velocity_score": f"Transaction velocity is {'suspicious' if importance > 0 else 'normal'}",
            "location_risk": f"Location risk is {'elevated' if importance > 0 else 'normal'}",
            "time_risk": f"Timing pattern is {'unusual' if importance > 0 else 'normal'}"
        }

        return explanations.get(feature_name)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance info"""
        return {
            "model_version": self.model_version,
            "model_type": "RandomForestClassifier",
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "last_updated": datetime.utcnow().isoformat(),
            "performance_metrics": {
                "precision": 0.85,  # Mock metrics
                "recall": 0.82,
                "f1_score": 0.83,
                "auc_roc": 0.89
            }
        }


# Global ML service instance
ml_service = MLModelService()
