import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial preprocessing on the transaction data.
    - Combines Date and Time into DateTime.
    - Extracts Hour and DayOfWeek.
    - Creates Log_Amount.
    - Drops original and leaky columns.
    """
    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # 1. Feature Engineering
    # Combine Date and Time into DateTime
    df_processed['DateTime'] = pd.to_datetime(
        df_processed['Date'] + ' ' + df_processed['Time'])

    # Extract Hour and DayOfWeek
    df_processed['Hour'] = df_processed['DateTime'].dt.hour
    df_processed['DayOfWeek'] = df_processed['DateTime'].dt.day_name()

    # Create Log_Amount
    df_processed['Log_Amount'] = np.log1p(df_processed['Amount'])

    # 2. Dropping Columns
    columns_to_drop = [
        'Date',
        'Time',
        'DateTime',  # Original DateTime after feature extraction
        'Amount',    # Original Amount after creating Log_Amount
        'Laundering_type',  # Target leakage
        'Sender_account',  # High cardinality identifier
        'Receiver_account'  # High cardinality identifier
    ]
    df_processed = df_processed.drop(columns=columns_to_drop)

    return df_processed


# Example usage (optional, for testing):
if __name__ == '__main__':
    # Create a sample DataFrame similar to the synthetic data
    sample_data = {
        'Date': ['2023-01-01', '2023-01-01'],
        'Time': ['10:00:00', '11:00:00'],
        'Sender_account': [123, 456],
        'Receiver_account': [789, 101],
        'Amount': [100.0, 15000.0],
        'Payment_currency': ['USD', 'EUR'],
        'Received_currency': ['USD', 'EUR'],
        'Sender_bank_location': ['New York', 'London'],
        'Receiver_bank_location': ['London', 'New York'],
        'Payment_type': ['WIRE', 'SWIFT'],
        'Is_laundering': [0, 1],
        'Laundering_type': ['none', 'structuring']
    }
    sample_df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(sample_df.head())

    processed_df = preprocess_data(sample_df)
    print("\nProcessed DataFrame:")
    print(processed_df.head())
    print("\nProcessed DataFrame columns:")
    print(processed_df.columns)
