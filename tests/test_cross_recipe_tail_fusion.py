from scripts.run_cross_recipe_tail_fusion import append_budget_scores


def test_append_budget_scores_emits_one_run_per_budget():
    runs = []
    append_budget_scores(
        runs, "candidate", 42, [0.1, 0.2, 0.3, 0.4],
        [1.0, 2.0, 3.0, 4.0], [0.25, 0.5],
        {0.25: 3.0, 0.5: 2.0},
    )
    assert [run["evaluation_fraction"] for run in runs] == [0.25, 0.5]
    assert all(run["condition"] == "candidate" for run in runs)
