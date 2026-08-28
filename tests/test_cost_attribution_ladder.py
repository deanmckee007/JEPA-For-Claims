import argparse

import pytest

from scripts.run_cost_attribution_ladder import (
    aggregate_runs,
    parse_checkpoint_spec,
    render_markdown,
)


def _metrics(value):
    return {
        "target_probe_mae_dollars": value,
        "target_probe_rmse_dollars": value + 1,
        "target_probe_rmse_log1p": value / 100,
        "target_probe_wape_percent": value / 10,
    }


def test_parse_checkpoint_spec_extracts_group_and_seed():
    result = parse_checkpoint_spec("sparse_seed43=experiments/model.ckpt")
    assert result == {
        "name": "sparse_seed43",
        "group": "sparse",
        "seed": 43,
        "path": "experiments/model.ckpt",
    }


@pytest.mark.parametrize("value", ["model.ckpt", "dense=model.ckpt", "=model.ckpt"])
def test_parse_checkpoint_spec_requires_named_seed(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_checkpoint_spec(value)


def test_aggregate_and_markdown_report_means_and_run_counts():
    runs = [
        {
            "condition": "dense_pretrained",
            "label_fraction": 1.0,
            "metrics": _metrics(10.0),
        },
        {
            "condition": "dense_pretrained",
            "label_fraction": 1.0,
            "metrics": _metrics(14.0),
        },
    ]
    aggregate = aggregate_runs(runs)
    row = aggregate["dense_pretrained|1"]
    assert row["num_runs"] == 2
    assert row["metrics"]["target_probe_mae_dollars"]["mean"] == 12.0
    markdown = render_markdown(aggregate)
    assert "dense_pretrained" in markdown
    assert "100%" in markdown
    assert "12.00" in markdown
