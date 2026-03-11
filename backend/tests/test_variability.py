from __future__ import annotations

import numpy as np
import pandas as pd

from app.stats.variability import compute_variability_scores, gini_impurity, shannon_entropy


def test_entropy_and_gini_known_distribution() -> None:
    series = pd.Series(["A", "A", "B", "B"])
    entropy = shannon_entropy(series)
    gini = gini_impurity(series)

    assert np.isclose(entropy, 1.0)
    assert np.isclose(gini, 0.5)


def test_variability_custom_freq_only_non_informative() -> None:
    df = pd.DataFrame({"Unnamed: 0": [0, 1, 2, 3], "cat": ["x", "y", "x", "z"], "num": [1, 2, 3, 4]})
    payload = compute_variability_scores(df, custom_mode="freq_only")

    cat_row = next(row for row in payload["rows"] if row["column"] == "cat")
    num_row = next(row for row in payload["rows"] if row["column"] == "num")
    assert all(row["column"] != "Unnamed: 0" for row in payload["rows"])

    assert cat_row["custom_index"] == 1.0
    assert any(w["code"] == "custom_non_informative" for w in cat_row["warnings"])
    assert num_row["coefficient_variation"] is not None
