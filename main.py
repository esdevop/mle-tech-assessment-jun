import pandas as pd
from pathlib import Path
from app.utils.transformations import _apply_halflife

def main():
    data_path = Path(__file__).parent / "data" / "raw" / "raw_data.csv"
    example_series = pd.read_csv(data_path, parse_dates=["date_week"])["tv_ad_executions"].values
    halflife = 2.5
    transformed_series = _apply_halflife(example_series, halflife)
    print(transformed_series)


if __name__ == "__main__":
    main()
