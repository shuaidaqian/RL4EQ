import csv
import json
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest

import env.eme_reference as eme_reference
from env.eme_reference import (
    EME_FULL_RADAR_DEPTH_SECONDS,
    load_evans_1965_envelope,
    physical_delay_samples,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "eme"
VALID_CSV_TEXT = """delay_ms,power_db_3p6cm,power_db_68cm,point_kind,source_figure,digitization_note
0.0,0.0,0.0,digitized,Evans1965-Fig8,前沿归一化
10.0,-22.5,-35.0,digitized,Evans1965-Fig8,人工读图锚点
11.6,-23.0,-40.0,support_extension,Evans1965-Text-SupportBoundary,物理支撑延伸
"""


def _valid_manifest():
    return {
        "schema_version": "eme-reference-v1",
        "doi": "10.6028/jres.069d.195",
        "source_figure": "Evans 1965 Fig. 8",
        "endpoint_extension_policy": {
            "kind": "support_extension",
            "delay_seconds": 0.0116,
            "upper": {
                "mode": "hold_last_observed",
                "value_db": -23.0,
                "observed": False,
            },
            "lower": {
                "mode": "right_censored",
                "censoring_limit_db": -40.0,
                "observed": False,
            },
        },
    }


def _load_fixture(tmp_path, csv_text=VALID_CSV_TEXT, manifest=None):
    csv_path = tmp_path / "envelope.csv"
    manifest_path = tmp_path / "manifest.json"
    csv_path.write_text(csv_text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(_valid_manifest() if manifest is None else manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return eme_reference._load_evans_1965_envelope_from_paths(csv_path, manifest_path)


def _reconstruct(envelope, **changes):
    payload = {field.name: getattr(envelope, field.name) for field in fields(envelope)}
    payload.update(changes)
    return type(envelope)(**payload)


def test_eme_reference_has_provenance_and_full_radar_depth():
    envelope = load_evans_1965_envelope()

    assert EME_FULL_RADAR_DEPTH_SECONDS == 0.0116
    assert envelope.source_doi == "10.6028/jres.069d.195"
    assert envelope.delay_seconds[0] == 0.0
    assert envelope.delay_seconds[-1] == pytest.approx(0.0116)
    assert np.all(np.diff(envelope.delay_seconds) > 0.0)
    assert np.all(envelope.lower_power_db <= envelope.upper_power_db)


def test_eme_reference_result_has_immutable_semantics():
    envelope = load_evans_1965_envelope()

    assert is_dataclass(envelope)
    with pytest.raises(FrozenInstanceError):
        envelope.source_doi = "被篡改的 DOI"
    assert envelope.delay_seconds.flags.writeable is False
    assert envelope.upper_power_db.flags.writeable is False
    assert envelope.lower_power_db.flags.writeable is False
    with pytest.raises(ValueError):
        envelope.delay_seconds[0] = 1.0


@pytest.mark.parametrize(
    "field_name",
    ["delay_seconds", "upper_power_db", "lower_power_db", "observed_mask"],
)
def test_eme_reference_arrays_cannot_be_made_writeable(field_name):
    envelope = load_evans_1965_envelope()
    array = getattr(envelope, field_name)

    with pytest.raises(ValueError):
        array.setflags(write=True)
    assert array.flags.writeable is False


@pytest.mark.parametrize(
    "case",
    [
        "two_dimensional",
        "empty",
        "unequal_length",
        "non_finite_delay",
        "non_finite_upper",
        "non_finite_lower",
        "non_increasing_delay",
        "non_zero_start",
        "wrong_end",
        "inverted_bounds",
    ],
)
def test_eme_echo_envelope_rejects_invalid_direct_construction(case):
    envelope = load_evans_1965_envelope()
    delay = np.array(envelope.delay_seconds, copy=True)
    upper = np.array(envelope.upper_power_db, copy=True)
    lower = np.array(envelope.lower_power_db, copy=True)

    if case == "two_dimensional":
        delay = delay.reshape(1, -1)
    elif case == "empty":
        delay = np.array([], dtype=np.float64)
        upper = np.array([], dtype=np.float64)
        lower = np.array([], dtype=np.float64)
    elif case == "unequal_length":
        upper = upper[:-1]
    elif case == "non_finite_delay":
        delay[1] = np.nan
    elif case == "non_finite_upper":
        upper[1] = np.inf
    elif case == "non_finite_lower":
        lower[1] = -np.inf
    elif case == "non_increasing_delay":
        delay[1] = delay[0]
    elif case == "non_zero_start":
        delay[0] = 0.0001
    elif case == "wrong_end":
        delay[-1] = 0.0115
    elif case == "inverted_bounds":
        lower[1] = upper[1] + 1.0

    with pytest.raises(ValueError):
        _reconstruct(
            envelope,
            delay_seconds=delay,
            upper_power_db=upper,
            lower_power_db=lower,
        )


def test_eme_echo_envelope_rejects_invalid_point_metadata():
    envelope = load_evans_1965_envelope()
    bad_point_kind = (*envelope.point_kind[:-1], "digitized")
    bad_observed_mask = np.array(envelope.observed_mask, copy=True)
    bad_observed_mask[-1] = True

    with pytest.raises(ValueError):
        _reconstruct(envelope, point_kind=bad_point_kind)
    with pytest.raises(ValueError):
        _reconstruct(envelope, observed_mask=bad_observed_mask)


def test_reference_files_are_sparse_auditable_standard_formats():
    with (DATA_DIR / "evans_1965_fig8_envelope.csv").open(
        "r", encoding="utf-8", newline=""
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))
    with (DATA_DIR / "reference_manifest.json").open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    assert 2 <= len(rows) <= 12
    assert float(rows[0]["delay_ms"]) == 0.0
    assert float(rows[-1]["delay_ms"]) == 11.6
    assert manifest["doi"] == "10.6028/jres.069d.195"
    assert manifest["public_pdf_url"].startswith("https://")
    assert manifest["source_figure"] == "Evans 1965 Fig. 8"
    assert manifest["digitization_date"] == "2026-08-25"
    assert manifest["axes"]["x"] == "相对月面前沿的时延（ms）"
    assert manifest["axes"]["y"] == "归一化对数回波功率（dB）"
    assert manifest["normalization"] == "两条曲线均在前沿（零时延）归一化为 0 dB"
    assert "人工数字化" in manifest["digitization_method"]
    assert "读图不确定性" in manifest["uncertainty"]
    assert "只作为 EME 上下包络约束" in manifest["scope_limitation"]
    assert "不宣称是 1.296 GHz 精确 PDP" in manifest["scope_limitation"]


def test_reference_csv_distinguishes_digitized_points_from_support_extension():
    with (DATA_DIR / "evans_1965_fig8_envelope.csv").open(
        "r", encoding="utf-8", newline=""
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert "point_kind" in reader.fieldnames
    assert {row["point_kind"] for row in rows[:-1]} == {"digitized"}
    assert rows[-1]["point_kind"] == "support_extension"
    assert {row["source_figure"] for row in rows[:-1]} == {"Evans1965-Fig8"}
    assert rows[-1]["source_figure"] == "Evans1965-Text-SupportBoundary"
    assert rows[-1]["digitization_note"] == "论文正文给出的物理雷达深度边界，非 Fig.8 定量读图"
    assert float(rows[-1]["power_db_3p6cm"]) == -23.0
    assert float(rows[-1]["power_db_68cm"]) == -40.0


def test_reference_manifest_separates_digitization_precision_from_endpoint_extension():
    with (DATA_DIR / "reference_manifest.json").open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    assert "仅适用于 point_kind=digitized" in manifest["digitization_method"]
    assert "support_extension" in manifest["uncertainty"]
    assert "0.5 至 1 dB 读图精度不适用" in manifest["uncertainty"]
    policy = manifest["endpoint_extension_policy"]
    assert policy == _valid_manifest()["endpoint_extension_policy"]


def test_loaded_reference_preserves_observation_and_endpoint_policy_metadata():
    envelope = load_evans_1965_envelope()

    assert envelope.point_kind == (
        "digitized",
        "digitized",
        "digitized",
        "digitized",
        "digitized",
        "digitized",
        "digitized",
        "digitized",
        "support_extension",
    )
    assert envelope.observed_mask.tolist() == [True] * 8 + [False]
    policy = envelope.endpoint_extension_policy
    assert is_dataclass(policy)
    assert policy.kind == "support_extension"
    assert policy.delay_seconds == 0.0116
    assert policy.upper.mode == "hold_last_observed"
    assert policy.upper.value_db == -23.0
    assert policy.upper.observed is False
    assert policy.lower.mode == "right_censored"
    assert policy.lower.censoring_limit_db == -40.0
    assert policy.lower.observed is False
    with pytest.raises(FrozenInstanceError):
        policy.kind = "digitized"


@pytest.mark.parametrize(
    "header",
    [
        "delay_ms,power_db_3p6cm,power_db_68cm,source_figure,digitization_note",
        (
            "delay_ms,power_db_3p6cm,power_db_68cm,point_kind,source_figure,"
            "digitization_note,source_figure"
        ),
    ],
)
def test_loader_rejects_missing_or_duplicate_csv_columns(tmp_path, header):
    csv_lines = VALID_CSV_TEXT.splitlines()
    malformed_csv = "\n".join([header, *csv_lines[1:]]) + "\n"

    with pytest.raises(ValueError, match="CSV"):
        _load_fixture(tmp_path, csv_text=malformed_csv)


@pytest.mark.parametrize(
    "malformed_csv",
    [
        VALID_CSV_TEXT.replace(",digitized,", ",estimated,", 1),
        VALID_CSV_TEXT.replace("support_extension", "digitized"),
        VALID_CSV_TEXT.replace(",digitized,", ",support_extension,", 1),
    ],
)
def test_loader_rejects_invalid_support_extension_layout(tmp_path, malformed_csv):
    with pytest.raises(ValueError, match="point_kind|support_extension"):
        _load_fixture(tmp_path, csv_text=malformed_csv)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema_version", "eme-reference-v0"),
        ("doi", "10.0000/invalid"),
        ("source_figure", "Evans 1965 Fig. 7"),
    ],
)
def test_loader_rejects_invalid_manifest_identity(
    tmp_path, field_name, invalid_value
):
    manifest = _valid_manifest()
    manifest[field_name] = invalid_value

    with pytest.raises(ValueError, match="manifest"):
        _load_fixture(tmp_path, manifest=manifest)


@pytest.mark.parametrize(
    "case",
    ["kind", "delay", "upper_value", "lower_limit", "upper_observed", "lower_observed"],
)
def test_loader_rejects_manifest_endpoint_policy_inconsistent_with_csv(
    tmp_path, case
):
    manifest = _valid_manifest()
    policy = manifest["endpoint_extension_policy"]
    if case == "kind":
        policy["kind"] = "digitized"
    elif case == "delay":
        policy["delay_seconds"] = 0.0115
    elif case == "upper_value":
        policy["upper"]["value_db"] = -22.0
    elif case == "lower_limit":
        policy["lower"]["censoring_limit_db"] = -39.0
    elif case == "upper_observed":
        policy["upper"]["observed"] = True
    elif case == "lower_observed":
        policy["lower"]["observed"] = True

    with pytest.raises(ValueError, match="manifest|端点"):
        _load_fixture(tmp_path, manifest=manifest)


def test_reference_loading_does_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    envelope = load_evans_1965_envelope()

    assert envelope.delay_seconds[-1] == pytest.approx(0.0116)


def test_physical_delay_is_derived_from_sample_rate():
    assert physical_delay_samples(2_000.0, 0.0116) == 24
    assert physical_delay_samples(10_000.0, 0.0116) == 116
    assert physical_delay_samples(1e-200, 1e-100) == 1


@pytest.mark.parametrize(
    ("sample_rate_hz", "max_delay_seconds"),
    [
        (0.0, 0.0116),
        (-1.0, 0.0116),
        (2_000.0, 0.0),
        (2_000.0, -0.0116),
    ],
)
def test_physical_delay_rejects_non_positive_parameters(sample_rate_hz, max_delay_seconds):
    with pytest.raises(ValueError, match="必须为正"):
        physical_delay_samples(sample_rate_hz, max_delay_seconds)


@pytest.mark.parametrize(
    ("sample_rate_hz", "max_delay_seconds"),
    [
        (np.nan, 0.0116),
        (np.inf, 0.0116),
        (2_000.0, np.nan),
        (2_000.0, np.inf),
        (sys.float_info.max, 2.0),
        (float.fromhex("0x0.0000000000001p-1022"), 0.5),
    ],
)
def test_physical_delay_rejects_non_finite_or_overflowed_product(
    sample_rate_hz, max_delay_seconds
):
    with pytest.raises(ValueError):
        physical_delay_samples(sample_rate_hz, max_delay_seconds)
