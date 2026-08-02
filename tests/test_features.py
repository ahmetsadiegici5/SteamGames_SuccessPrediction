import numpy as np
import pandas as pd
import pytest

from streamlit_app import FEATURE_COLS, engineer_features, parse_owners


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0-20000", 10000.0),
        ("20000-50000", 35000.0),
        ("10k-20k", 15000.0),
        ("1,000,000-2,000,000", 1500000.0),
    ],
)
def test_parse_owners_returns_range_midpoint(raw, expected):
    assert parse_owners(raw) == expected


def test_parse_owners_invalid_values():
    assert np.isnan(parse_owners("unknown"))
    assert np.isnan(parse_owners(None))


def make_games():
    return pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "release_date": ["2018-01-15", "2018-07-01", "2018-10-20"],
            "english": [1, 0, 1],
            "developer": ["Studio", "Indie", "Big"],
            "publisher": ["Studio", "Indie", "Publisher Inc"],
            "platforms": ["windows", "windows;mac", "windows;mac;linux"],
            "required_age": [0, 18, 0],
            "categories": ["Single-player", "Multi-player;Co-op", "Single-player"],
            "owners": ["0-20000", "20000-50000", "50000-100000"],
            "price": [0.0, 9.99, 19.99],
        }
    )


def test_engineer_features_builds_expected_columns():
    df, median = engineer_features(make_games())

    assert median == 35000.0
    assert set(FEATURE_COLS) <= set(df.columns)
    assert df["success"].tolist() == [0, 0, 1]
    assert df["has_publisher"].tolist() == [0, 0, 1]
    assert df["has_multiplayer"].tolist() == [0, 1, 0]
    assert df["platform_count"].tolist() == [1, 2, 3]
    assert df["is_free"].tolist() == [1, 0, 0]
    assert df["is_mature"].tolist() == [0, 1, 0]
    assert df[["season_Winter", "season_Summer", "season_Fall"]].values.diagonal().tolist() == [1, 1, 1]


def test_real_dataset_has_no_missing_prices():
    df = pd.read_csv("data/steam.csv")
    assert len(df) == 27075
    assert pd.to_numeric(df["price"], errors="coerce").notna().all()
