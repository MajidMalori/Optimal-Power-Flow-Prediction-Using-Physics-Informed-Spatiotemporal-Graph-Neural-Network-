from src.visualization.plot_ws_common import ensure_case_dir, METHOD_COLORS


def test_ensure_case_dir_creates_case_folder(tmp_path):
    base = tmp_path / "reports" / "warmstart" / "speed"
    out = ensure_case_dir(str(base), "case33")
    assert out.endswith("case33")
    assert (base / "case33").exists()


def test_method_colors_have_required_keys():
    assert set(METHOD_COLORS.keys()) == {"flat", "dc", "warmstart"}
