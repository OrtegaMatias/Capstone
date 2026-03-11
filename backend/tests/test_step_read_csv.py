from __future__ import annotations

from app.etl.step_clean_columns import CleanColumnsStep
from app.etl.step_read_csv import robust_read_csv
from app.etl.types import DatasetInput, PipelineContext


def test_robust_read_csv_semicolon_latin1() -> None:
    content = "Unnamed: 0;Owner;Type\n1;Año;X\n2;B;Y\n".encode("latin-1")
    df, meta = robust_read_csv(content, "sample.csv")

    assert meta["separator"] == ";"
    assert meta["encoding"] == "latin-1"
    assert list(df.columns) == ["Unnamed: 0", "Owner", "Type"]
    assert df.iloc[0]["Owner"] == "Año"


def test_clean_columns_normalizes_double_spaces() -> None:
    content = "  Unnamed: 0 ; Owner  ;Type\n1;A;X\n".encode("utf-8")
    df, _ = robust_read_csv(content, "spaces.csv")

    ctx = PipelineContext(in_input=DatasetInput(filename="spaces.csv", content=content), in_df=df)
    clean = CleanColumnsStep().run(ctx)

    assert "Owner" in clean.in_df.columns
    assert "Unnamed: 0" in clean.in_df.columns
