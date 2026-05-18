"""Tests d'intégration légers : bootstrap → train → predict."""
import pytest
import pandas as pd
from src.training.bootstrap import generate
from src.features.build import build_features, feature_matrix


def test_synthetic_dataset_shape():
    df = generate(n_rows=200, seed=1)
    assert len(df) == 200
    expected_cols = {
        "tribunal", "city", "region", "property_type",
        "surface", "rooms", "initial_price", "adjudicated_price",
        "adjudication_date",
    }
    assert expected_cols.issubset(df.columns)
    # Sanity sur les ordres de grandeur
    assert df["surface"].between(1, 5000).all()
    assert df["initial_price"].gt(0).all()
    assert df["adjudicated_price"].gt(0).all()


def test_build_features_training():
    df = generate(n_rows=100, seed=2)
    feats = build_features(df, training=True)
    # Doit avoir les colonnes attendues
    assert "price_per_sqm_initial" in feats.columns
    assert "year" in feats.columns
    assert "month" in feats.columns
    # Pas de NaN sur les features clés
    assert not feats["surface"].isna().any()
    assert not feats["initial_price"].isna().any()


def test_build_features_inference():
    """Sans target, must still work."""
    df = pd.DataFrame([{
        "tribunal": "TJ Paris", "city": "Paris", "region": "Île-de-France",
        "property_type": "Appartement", "surface": 68, "rooms": 3,
        "initial_price": 185000,
    }])
    feats = build_features(df, training=False)
    X = feature_matrix(feats)
    assert len(X) == 1
    assert X.iloc[0]["surface"] == 68
