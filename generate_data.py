import logging
from datetime import datetime, timedelta, time
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
    # weighted by hour-of-day with business hours peak
    total_days = (end.date() - start.date()).days
    days = np.random.randint(0, total_days + 1, size=n)
    dates = [start.date() + timedelta(days=int(d)) for d in days]
    # business hours 8-18 have higher weight
    hour_weights = np.array([0.02] * 24)
    hour_weights[8:19] = 0.04
    hour_weights /= hour_weights.sum()
    hours = np.random.choice(np.arange(24), size=n, p=hour_weights)
    minutes = np.random.randint(0, 60, size=n)
    seconds = np.random.randint(0, 60, size=n)
    datetimes = [datetime.combine(d, time(h, m, s))
                 for d, h, m, s in zip(dates, hours, minutes, seconds)]
    return pd.DataFrame({
        "DateTime": datetimes,
        "Date": [dt.date().isoformat() for dt in datetimes],
        "Time": [dt.time().isoformat() for dt in datetimes],
    })


def generate_accounts(n: int):
    senders = np.random.randint(1_000_000, 9_999_999, size=n)
    receivers = np.random.randint(1_000_000, 9_999_999, size=n)
    # network hubs: a small set of accounts with higher connectivity
    hub_count = max(1, n // 2000)
    hub_senders = np.random.choice(senders, size=hub_count, replace=False)
    hub_receivers = np.random.choice(receivers, size=hub_count, replace=False)
    hub_mask = np.random.rand(n) < 0.1
    senders = np.where(hub_mask, np.random.choice(
        hub_senders, size=n), senders)
    receivers = np.where(hub_mask, np.random.choice(
        hub_receivers, size=n), receivers)
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
    # realistic merchant categories
    merchants = ["retail", "electronics", "legal services",
                 "shell corporation", "pharma", "luxury goods", "crypto exchange"]
    merchant_category = sample_categories(n, merchants)
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
        "Merchant_category": merchant_category,
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
