from __future__ import annotations


def test_upload_and_eda_smoke(client) -> None:
    in_csv = (
        "Unnamed: 0,Condition,Owner,Size,Type,Quality,week\n"
        "1,New,1001,12.5,A,High,1\n"
        "2,Used,1002,10.2,B,Medium,1\n"
    )

    response = client.post(
        "/api/v1/upload",
        files={"in_file": ("Grupo1_in.csv", in_csv, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_id"]
    assert payload["has_target"] is False
    assert "schema" in payload

    dataset_id = payload["dataset_id"]
    eda_res = client.get(f"/api/v1/datasets/{dataset_id}/eda")
    assert eda_res.status_code == 200
    eda_payload = eda_res.json()
    assert "shape" in eda_payload
    assert "missingness" in eda_payload


def test_upload_both_files_enables_target(client) -> None:
    in_csv = (
        "Unnamed: 0,Condition,Owner,Size,Type,Quality,week\n"
        "1,New,1001,12.5,A,High,1\n"
        "2,Used,1002,10.2,B,Medium,1\n"
    )
    out_csv = (
        "Unnamed: 0,Owner,Size,Type,Quality,DaysInDeposit,week\n"
        "1,1001,12.5,A,High,20,1\n"
        "2,1002,10.2,B,Medium,30,1\n"
    )

    response = client.post(
        "/api/v1/upload",
        files={
            "in_file": ("Grupo1_in.csv", in_csv, "text/csv"),
            "out_file": ("Grupo1_out.csv", out_csv, "text/csv"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_target"] is True
    assert "schema" in payload

    dataset_id = payload["dataset_id"]
    supervised = client.get(f"/api/v1/datasets/{dataset_id}/supervised/overview")
    assert supervised.status_code == 200
    assert supervised.json()["target_present"] is True


def test_eda_warnings_are_normalized_for_technical_and_week_noise(client) -> None:
    in_csv = (
        "Unnamed: 0,Condition,Owner,Size,Type,Quality,week\n"
        "1,O,1,2,DRY,A,5\n"
        "2,O,2,2,DRY,B,5\n"
    )
    out_csv = (
        "Unnamed: 0,Owner,Size,Type,Quality,DaysInDeposit,week\n"
        "1,9,2,DRY,A,10,5\n"
        "2,8,2,DRY,B,20,5\n"
    )
    upload = client.post(
        "/api/v1/upload",
        files={
            "in_file": ("Grupo1_in.csv", in_csv, "text/csv"),
            "out_file": ("Grupo1_out.csv", out_csv, "text/csv"),
        },
    )
    dataset_id = upload.json()["dataset_id"]

    eda = client.get(f"/api/v1/datasets/{dataset_id}/eda")
    assert eda.status_code == 200
    warnings = eda.json()["warnings"]

    assert not any(w.get("column") == "Unnamed: 0" for w in warnings)
    assert not any(w.get("code") == "high_cardinality" and w.get("column") == "DaysInDeposit" for w in warnings)
    assert not any(w.get("code") == "constant_column" and w.get("column") == "week" for w in warnings)
    assert not any(w.get("code") == "near_constant_column" and w.get("column") == "week" for w in warnings)
    assert any(w.get("code") == "week_constant" and w.get("column") == "week" for w in warnings)


def test_multiple_regression_endpoint_with_out_source(client) -> None:
    in_csv = (
        "Unnamed: 0,Condition,Owner,Size,Type,Quality,week\n"
        "1,O,1,1,DRY,A,5\n"
        "2,O,1,2,RF,A,5\n"
        "3,D,2,3,DRY,B,5\n"
        "4,O,2,4,RF,B,5\n"
        "5,D,3,5,DRY,C,5\n"
        "6,O,3,6,RF,C,5\n"
    )
    out_csv = (
        "Unnamed: 0,Owner,Size,Type,Quality,DaysInDeposit,week\n"
        "1,1,1,DRY,A,8,5\n"
        "2,1,2,RF,A,13,5\n"
        "3,2,3,DRY,B,14,5\n"
        "4,2,4,RF,B,20,5\n"
        "5,3,5,DRY,C,20,5\n"
        "6,3,6,RF,C,27,5\n"
    )

    upload = client.post(
        "/api/v1/upload",
        files={
            "in_file": ("Grupo1_in.csv", in_csv, "text/csv"),
            "out_file": ("Grupo1_out.csv", out_csv, "text/csv"),
        },
    )
    assert upload.status_code == 200
    dataset_id = upload.json()["dataset_id"]

    response = client.get(f"/api/v1/datasets/{dataset_id}/supervised/multiple-regression")
    assert response.status_code == 200
    payload = response.json()
    assert payload["target_present"] is True
    assert payload["model_built"] is True
    assert payload["n_obs"] >= 6
    assert isinstance(payload["anova_rows"], list)
    assert isinstance(payload["conclusions"], list)


def test_multiple_regression_endpoint_without_out_source(client) -> None:
    in_csv = (
        "Unnamed: 0,Condition,Owner,Size,Type,Quality,week\n"
        "1,New,1001,12.5,A,High,1\n"
        "2,Used,1002,10.2,B,Medium,1\n"
    )
    upload = client.post(
        "/api/v1/upload",
        files={"in_file": ("Grupo1_in.csv", in_csv, "text/csv")},
    )
    assert upload.status_code == 200
    dataset_id = upload.json()["dataset_id"]

    response = client.get(f"/api/v1/datasets/{dataset_id}/supervised/multiple-regression")
    assert response.status_code == 200
    payload = response.json()
    assert payload["target_present"] is False
    assert payload["model_built"] is False
    assert any(w["code"] == "missing_out_source" for w in payload["warnings"])
