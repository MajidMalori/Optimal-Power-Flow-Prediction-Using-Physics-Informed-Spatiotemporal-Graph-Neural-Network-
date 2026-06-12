import os

from src.visualization.plot_training import (
    build_case_training_metrics_csv,
    plot_case_final_metrics,
    plot_case_loss_overlay,
    plot_loss_curves,
    plot_publication_summary,
    plot_lr_curve,
    plot_timing,
    plot_test_metrics,
    write_summary_index,
)


def test_training_plotters_smoke(tmp_path):
    metrics_csv = tmp_path / "metrics_epoch.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                "epoch,epoch_time_s,train_loss,val_loss,lr",
                "0,1.0,2.0,3.0,0.001",
                "1,1.2,1.5,2.5,0.001",
            ]
        ),
        encoding="utf-8",
    )

    test_csv = tmp_path / "test_metrics.csv"
    test_csv.write_text(
        "\n".join(
            [
                "model,case,test_loss",
                "StandardGCN,case33,0.123",
            ]
        ),
        encoding="utf-8",
    )

    out1 = tmp_path / "loss.png"
    out2 = tmp_path / "lr.png"
    out3 = tmp_path / "timing.png"
    out4 = tmp_path / "test.png"

    assert plot_loss_curves(str(metrics_csv), str(out1)) is not None
    assert plot_lr_curve(str(metrics_csv), str(out2)) is not None
    assert plot_timing(str(metrics_csv), str(out3)) is not None
    assert plot_test_metrics(str(test_csv), str(out4)) is not None

    for p in [out1, out2, out3, out4]:
        assert os.path.exists(p)


def test_case_summary_builders(tmp_path):
    m1 = tmp_path / "m1_metrics.csv"
    m1.write_text(
        "\n".join(
            [
                "epoch,epoch_time_s,val_loss",
                "0,1.0,0.5",
                "1,1.1,0.4",
            ]
        ),
        encoding="utf-8",
    )
    m2 = tmp_path / "m2_metrics.csv"
    m2.write_text(
        "\n".join(
            [
                "epoch,epoch_time_s,val_loss",
                "0,2.0,0.7",
                "1,2.1,0.6",
            ]
        ),
        encoding="utf-8",
    )

    t1 = tmp_path / "m1_test.csv"
    t1.write_text(
        "\n".join(
            [
                "model,case,test_loss,test_power_balance,test_voltage_limit",
                "StandardGCN,case33,0.2,,",
            ]
        ),
        encoding="utf-8",
    )
    t2 = tmp_path / "m2_test.csv"
    t2.write_text(
        "\n".join(
            [
                "model,case,test_loss,test_data_loss,test_power_balance,test_voltage_limit,test_branch_capacity",
                "PIGCN,case33,0.5,0.4,0.8,0.0,0.0",
            ]
        ),
        encoding="utf-8",
    )

    per_model_metrics = {"StandardGCN": str(m1), "PIGCN": str(m2)}
    per_model_tests = {"StandardGCN": str(t1), "PIGCN": str(t2)}

    master_csv = tmp_path / "case33_training_metrics.csv"
    assert build_case_training_metrics_csv(per_model_tests, str(master_csv)) is not None
    assert master_csv.exists()

    loss_overlay = tmp_path / "loss_overlay.png"
    assert plot_case_loss_overlay(per_model_metrics, str(loss_overlay)) is not None
    assert loss_overlay.exists()

    cmp_plot = tmp_path / "test_loss_comparison.png"
    assert plot_case_final_metrics(per_model_tests, "test_loss", str(cmp_plot)) is not None
    assert cmp_plot.exists()

    pub_plot = tmp_path / "publication_summary.png"
    assert plot_publication_summary(per_model_tests, per_model_metrics, str(pub_plot)) is not None
    assert pub_plot.exists()

    idx = tmp_path / "index.json"
    write_summary_index(str(idx), {"ok": True})
    assert idx.exists()

