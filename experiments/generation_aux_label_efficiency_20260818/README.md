# Cost-label efficiency curve

The selected epoch-20 encoder was frozen and evaluated with downstream seeds
42--44 at six cost-label budgets. Token and label-free auxiliary objectives used
the full training cohort. Evaluation used frozen validation; test was untouched.

| Cost labels | Cost only MAE | TTNC-only MAE | All-token MAE | Consistency + SigReg MAE |
|---:|---:|---:|---:|---:|
| 1% (150) | 2,770.34 | 2,450.64 | 2,348.65 | **2,232.11** |
| 2% (300) | 2,534.73 | 2,125.17 | 2,118.77 | **1,974.94** |
| 5% (750) | 2,364.17 | 2,014.21 | 1,984.72 | **1,848.36** |
| 10% (1,500) | 1,945.55 | 1,723.61 | 1,721.37 | **1,665.67** |
| 25% (3,760) | 1,683.11 | 1,636.10 | **1,624.98** | 1,625.45 |
| 100% (15,048) | **1,575.44** | 1,620.19 | 1,654.12 | 1,624.16 |

Consistency plus SigReg won all three paired seeds from 1% through 25%. Its
paired improvement over cost-only was $538.24 at 1%, $559.80 at 2%, $515.81 at
5%, $279.88 at 10%, and $57.66 at 25%. At full cost supervision every auxiliary
objective hurt, showing that the benefit is specifically label efficiency rather
than a universally better cost architecture.

The complete metrics, paired deltas, protocol, and per-run records are in
`summary.json` and the adjacent JSON files.
