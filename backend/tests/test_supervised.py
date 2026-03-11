from __future__ import annotations

import pandas as pd

from app.stats.supervised import compute_supervised_overview


def test_supervised_overview_handles_only_target_plus_technical_id() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": list(range(10)),
            "DaysInDeposit": [5, 7, 10, 9, 8, 12, 15, 11, 6, 7],
        }
    )

    payload = compute_supervised_overview(df)
    assert payload["target_present"] is True
    assert payload["pearson_correlations"] == []
    assert payload["mutual_information"] == []

