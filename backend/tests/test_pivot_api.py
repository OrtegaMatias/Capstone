from __future__ import annotations


def _upload_pair(client):
    in_csv = (
        "Unnamed: 0,Condition,Owner,Size,Type,Quality,week\n"
        "1,O,1,2,DRY,CLASE B-C,5\n"
        "2,O,1,2,DRY,CLASE D,5\n"
        "3,D,2,2,DRY,CLASE B-C,5\n"
        "4,O,2,1,RF,CLASE A,5\n"
    )
    out_csv = (
        "Unnamed: 0,Owner,Size,Type,Quality,DaysInDeposit,week\n"
        "1,1,2,DRY,CLASE B-C,28,5\n"
        "2,1,2,DRY,CLASE D,29,5\n"
        "3,2,2,DRY,CLASE B-C,6,5\n"
        "4,2,1,RF,CLASE A,14,5\n"
    )

    response = client.post(
        "/api/v1/upload",
        files={
            "in_file": ("Grupo1_in.csv", in_csv, "text/csv"),
            "out_file": ("Grupo1_out.csv", out_csv, "text/csv"),
        },
    )
    assert response.status_code == 200
    return response.json()["dataset_id"]


def test_pivot_sources_and_metadata(client) -> None:
    dataset_id = _upload_pair(client)

    sources_res = client.get(f"/api/v1/datasets/{dataset_id}/pivot/sources")
    assert sources_res.status_code == 200
    sources = {item["source"]: item["available"] for item in sources_res.json()["sources"]}
    assert sources["in"] is True
    assert sources["out"] is True

    metadata_res = client.get(f"/api/v1/datasets/{dataset_id}/pivot/metadata", params={"source": "in"})
    assert metadata_res.status_code == 200
    metadata = metadata_res.json()
    assert "Owner" in metadata["dimensions"]
    assert "Unnamed: 0" not in metadata["dimensions"]
    assert "sum" in metadata["agg_functions"]
    assert "Owner" in metadata["filter_options"]
    assert "1" in metadata["filter_options"]["Owner"]
    assert "Unnamed: 0" not in metadata["field_agg_functions"]


def test_pivot_query_rejects_technical_id_column(client) -> None:
    dataset_id = _upload_pair(client)

    request_payload = {
        "source": "in",
        "row_dim": "Unnamed: 0",
        "col_dim": "Quality",
        "value_field": "Size",
        "agg_func": "sum",
        "filters": {},
        "include_blank": True,
        "top_k": 10,
        "small_n_threshold": 5,
    }

    res = client.post(f"/api/v1/datasets/{dataset_id}/pivot/query", json=request_payload)
    assert res.status_code == 400
    assert "technical/id-like" in res.json()["message"]


def test_pivot_query_sum_size_in(client) -> None:
    dataset_id = _upload_pair(client)

    request_payload = {
        "source": "in",
        "row_dim": "Owner",
        "col_dim": "Quality",
        "value_field": "Size",
        "agg_func": "sum",
        "filters": {},
        "include_blank": True,
        "top_k": 10,
        "small_n_threshold": 5,
    }

    res = client.post(f"/api/v1/datasets/{dataset_id}/pivot/query", json=request_payload)
    assert res.status_code == 200
    payload = res.json()

    assert payload["matrix"]["grand_total"]["value"] == 7.0
    owner_1 = next(row for row in payload["matrix"]["rows"] if row["row_key"] == "1")
    assert owner_1["row_total"]["value"] == 4.0


def test_pivot_query_rejects_invalid_rate_field(client) -> None:
    dataset_id = _upload_pair(client)

    request_payload = {
        "source": "out",
        "row_dim": "Owner",
        "col_dim": "Size",
        "value_field": "Size",
        "agg_func": "rate_gt_14",
        "filters": {},
        "include_blank": True,
        "top_k": 10,
        "small_n_threshold": 5,
    }

    res = client.post(f"/api/v1/datasets/{dataset_id}/pivot/query", json=request_payload)
    assert res.status_code == 400
    assert "DaysInDeposit" in res.json()["message"]


def test_pivot_query_with_filters(client) -> None:
    dataset_id = _upload_pair(client)

    request_payload = {
        "source": "out",
        "row_dim": "Owner",
        "col_dim": "Size",
        "value_field": "DaysInDeposit",
        "agg_func": "mean",
        "filters": {"Owner": ["1"]},
        "include_blank": True,
        "top_k": 10,
        "small_n_threshold": 5,
    }

    res = client.post(f"/api/v1/datasets/{dataset_id}/pivot/query", json=request_payload)
    assert res.status_code == 200
    payload = res.json()
    assert payload["matrix"]["grand_total"]["count"] == 2
    assert payload["matrix"]["grand_total"]["value"] == 28.5
