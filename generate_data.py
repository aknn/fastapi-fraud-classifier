import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
NUM_RECORDS = 100_000
OUTPUT_PATH = Path("data/raw/synthetic_transactions.csv")

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# DATA GENERATION FUNCTIONS
# -----------------------------------------------------------------------------
fake = Faker()


def generate_timestamps(n: int, start: datetime, end: datetime) -> pd.DataFrame:
    total_seconds = int((end - start).total_seconds())
    offsets = np.random.randint(0, total_seconds, size=n)
    dates = [start + timedelta(seconds=int(off)) for off in offsets]
    return pd.DataFrame({
        "DateTime": dates,
        "Date": [dt.date().isoformat() for dt in dates],
        "Time": [dt.time().isoformat() for dt in dates],
    })


def generate_accounts(n: int):
    senders = np.random.randint(1_000_000, 9_999_999, size=n)
    receivers = np.random.randint(1_000_000, 9_999_999, size=n)
    return senders, receivers


def generate_amounts(n: int, fraud_ratio: float = 0.05):
    is_laundering = np.random.rand(n) < fraud_ratio
    amounts = np.where(
        is_laundering,
        np.random.lognormal(mean=10, sigma=1.2, size=n),
        np.random.lognormal(mean=8, sigma=1.0, size=n),
    )
    return amounts, is_laundering.astype(int)


def sample_categories(n: int, choices: list[str]) -> np.ndarray:
    return np.random.choice(choices, size=n)


def build_dataframe(n: int) -> pd.DataFrame:
    logger.info("Generating timestamps...")
    ts = generate_timestamps(n, start=datetime(
        2020, 1, 1), end=datetime(2024, 12, 31))
    logger.info("Generating account IDs...")
    senders, receivers = generate_accounts(n)
    logger.info("Generating amounts and flags...")
    amounts, flags = generate_amounts(n)

    currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]
    banks = ["London", "New York", "Tokyo", "Zurich", "Toronto", "Frankfurt"]
    payment_types = ["SWIFT", "WIRE", "ACH", "SEPA", "RTGS"]

    logger.info("Sampling categorical fields...")
    return pd.DataFrame({
        "Date": ts["Date"],
        "Time": ts["Time"],
        "Sender_account": senders,
        "Receiver_account": receivers,
        "Amount": amounts,
        "Payment_currency": sample_categories(n, currencies),
        "Received_currency": sample_categories(n, currencies),
        "Sender_bank_location": sample_categories(n, banks),
        "Receiver_bank_location": sample_categories(n, banks),
        "Payment_type": sample_categories(n, payment_types),
        "Is_laundering": flags,
        "Laundering_type": np.where(
            flags == 1,
            sample_categories(
                n, ["structuring", "smurfing", "layering", "shell"]),
            "none"
        ),
    })

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main():
    logger.info(f"Building synthetic dataset ({NUM_RECORDS} records)...")
    df = build_dataframe(NUM_RECORDS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved synthetic data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
