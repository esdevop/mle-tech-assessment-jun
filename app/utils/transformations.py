import numpy as np
from numpy.typing import NDArray

def _apply_halflife(series: NDArray[np.float64], halflife: np.float32, rounding: int=4) -> NDArray[np.float32]:
    """
    Apply exponential decay to a pandas Series based on the given half-life.

    Parameters:
    - series: NDArray[np.float64]
        The input time series data.
    - halflife: np.float64
        The half-life period for the exponential decay.

    Returns:
    - NDArray[np.float32]
        The transformed series with exponential decay applied.
    """
    adstocked_series = series.copy()

    # TODO: Enter your calculation here
    # Implement the exponential decay logic
    # Hint: Use the formula mentioned in the background sectionies[i] += adstocked_series[i - 1] * np.exp(decay_factor)

    return np.round(adstocked_series, rounding).astype(np.float32)

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    # Example usage
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "raw_data.csv"
    example_series = pd.read_csv(data_path, parse_dates=["date_week"])["tv_ad_executions"].values
    halflife = 2.5
    transformed_series = _apply_halflife(example_series, halflife)
    print(transformed_series)