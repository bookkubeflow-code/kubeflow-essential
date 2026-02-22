# tests/test_components.py
import pandas as pd


def test_clean_data_removes_nulls():
    """Test that clean_data removes null values."""
    from components.preprocessing import clean_data

    # Create test data with nulls
    test_df = pd.DataFrame({
        'feature1': [1, 2, None, 4],
        'feature2': [1, None, 3, 4],
        'target': [0, 1, 1, 0]
    })

    cleaned = clean_data(test_df)

    # Verify no nulls remain
    assert cleaned.isnull().sum().sum() == 0

    # Verify we kept non-null rows
    assert len(cleaned) == 2  # Only rows 0 and 3 have no nulls


def test_clean_data_preserves_schema():
    """Test that clean_data doesn't change column names."""
    from components.preprocessing import clean_data

    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })

    cleaned = clean_data(test_df)

    assert list(cleaned.columns) == ['feature1', 'feature2', 'target']
