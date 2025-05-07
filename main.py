from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.models import MsgPayload  # Corrected import path

app = FastAPI()

# Define the response model for transaction classification


class TransactionResponse(BaseModel):
    msg_id: Optional[int]
    risk_score: float
    alert: bool
    explanation: str

# Placeholder for the classification pipeline


def classify_transaction(payload: MsgPayload) -> TransactionResponse:
    # Toy logic: if msg_name contains "suspicious", flag it.
    # This will be replaced by the rule-based -> ML -> LLM pipeline.
    is_suspicious = "suspicious" in payload.msg_name.lower()

    if is_suspicious:
        return TransactionResponse(
            msg_id=payload.msg_id,
            risk_score=0.85,
            alert=True,
            explanation="High risk: Message name contained suspicious keyword."
        )
    else:
        return TransactionResponse(
            msg_id=payload.msg_id,
            risk_score=0.1,
            alert=False,
            explanation="Low risk: No suspicious keywords detected in message name."
        )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hello FinCrime Sentinel"}


# About page route
@app.get("/about")
def about() -> dict[str, str]:
    return {"message": "This is the FinCrime Sentinel AML Monitoring API."}


# Route to process a transaction
@app.post("/api/v1/transactions", response_model=TransactionResponse)
def process_transaction(payload: MsgPayload) -> TransactionResponse:
    # In a real scenario, if payload.msg_id is None, you might generate one here
    # and update the payload or pass it separately to classify_transaction.
    # For now, classify_transaction handles the msg_id from the payload.

    classification_result = classify_transaction(payload)

    # Here you would also log the transaction and result to PostgreSQL
    # and push alerts via WebSockets if classification_result.alert is True.

    return classification_result
