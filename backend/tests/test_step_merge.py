from __future__ import annotations

import pandas as pd

from app.etl.step_merge import MergeStep
from app.etl.types import PipelineContext


def test_merge_step_by_unnamed_index() -> None:
    in_df = pd.DataFrame(
        {
            "Unnamed: 0": [1, 2, 3],
            "Owner": ["A", "B", "C"],
            "Type": ["X", "Y", "Z"],
        }
    )
    out_df = pd.DataFrame(
        {
            "Unnamed: 0": [2, 3, 4],
            "Owner": ["B", "C", "D"],
            "DaysInDeposit": [10, 12, 8],
        }
    )

    ctx = PipelineContext(in_df=in_df, out_df=out_df)
    merged_ctx = MergeStep().run(ctx)

    assert merged_ctx.merged_df is not None
    assert merged_ctx.merged_df.shape[0] == 2
    assert set(merged_ctx.merged_df["Unnamed: 0"].tolist()) == {2, 3}
    assert merged_ctx.step_metadata["step_merge"]["unmatched_in_rows"] == 1
    assert merged_ctx.step_metadata["step_merge"]["unmatched_out_rows"] == 1
