"""Paired statistical tests on the per-case scores recomputed on 2026-09-29.

The scores in scores/*_2026-09-29.csv were recomputed with meta24_compute_metrics.py
from the predictions in nnUNet_predictions/Dataset222_SBT/*_preds. All models are
evaluated on the same 10 SBT test cases, so the tests are paired (Wilcoxon signed-rank).

Two comparisons are reported:
  1. Fine-tuned vs scratch-trained, per model and averaged over the three models
     per case (one-sided: fine-tuned better).
  2. TverskyBCE vs Default and vs SegResNet, fine-tuned models only (two-sided).
p-values are Holm-corrected within each block of tests. Specificity is not tested
because it is ~1.000 for all models (background voxels dominate).

Usage: python utils/statistical_tests_2026-09-29.py [--scores-dir scores] [--suffix _2026-09-29]
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODELS = ['default', 'segresnet', 'tverskybce']
METRICS = [f'{m}_{r}' for m in ['DICE', 'Hausdorff', 'Sensitivity'] for r in ['et', 'tc', 'wt']]


def holm(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def better_direction(metric: str) -> str:
    # Lower HD95 is better; higher is better for the other metrics
    return 'less' if metric.startswith('Hausdorff') else 'greater'


def load_scores(scores_dir: Path, suffix: str) -> dict[str, pd.DataFrame]:
    scores = {}
    for model in MODELS:
        for setting in ['finetune', 'scratch']:
            df = pd.read_csv(scores_dir / f'scores_{model}_{setting}{suffix}.csv')
            scores[f'{model}_{setting}'] = df.set_index('sample').sort_index()
    return scores


def finetune_vs_scratch(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    groups = {model: [model] for model in MODELS}
    groups['mean_of_models'] = MODELS
    for group, models in groups.items():
        block = []
        for metric in METRICS:
            # Average over the models per case, so each case counts once
            ft = np.mean([scores[f'{m}_finetune'][metric].values for m in models], axis=0)
            st = np.mean([scores[f'{m}_scratch'][metric].values for m in models], axis=0)
            alternative = better_direction(metric)
            n_better = int((ft < st).sum() if alternative == 'less' else (ft > st).sum())
            block.append({
                'comparison': 'finetune_vs_scratch',
                'group': group,
                'metric': metric,
                'mean_a': ft.mean(),
                'mean_b': st.mean(),
                'a_better_cases': n_better,
                'n_cases': len(ft),
                'alternative': alternative,
                'p_value': wilcoxon(ft, st, alternative=alternative).pvalue,
            })
        for row, p_holm in zip(block, holm([r['p_value'] for r in block])):
            row['p_holm'] = p_holm
        rows.extend(block)
    return pd.DataFrame(rows)


def architectures(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    block = []
    for other in ['default', 'segresnet']:
        for metric in METRICS:
            a = scores['tverskybce_finetune'][metric].values
            b = scores[f'{other}_finetune'][metric].values
            n_better = int((a < b).sum() if better_direction(metric) == 'less' else (a > b).sum())
            block.append({
                'comparison': 'architecture_finetuned',
                'group': f'tverskybce_vs_{other}',
                'metric': metric,
                'mean_a': a.mean(),
                'mean_b': b.mean(),
                'a_better_cases': n_better,
                'n_cases': len(a),
                'alternative': 'two-sided',
                'p_value': wilcoxon(a, b).pvalue,
            })
    for row, p_holm in zip(block, holm([r['p_value'] for r in block])):
        row['p_holm'] = p_holm
    return pd.DataFrame(block)


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scores-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'scores')
    parser.add_argument('--suffix', default='_2026-09-29')
    args = parser.parse_args()

    scores = load_scores(args.scores_dir, args.suffix)
    results = pd.concat([finetune_vs_scratch(scores), architectures(scores)], ignore_index=True)
    output = args.scores_dir / f'statistical_tests{args.suffix}.csv'
    results.to_csv(output, index=False)

    with pd.option_context('display.width', 200, 'display.max_rows', None, 'display.float_format', '{:.3f}'.format):
        print(results.drop(columns=['comparison', 'alternative']).to_string(index=False))
    print(f'\nSaved to {output}')


if __name__ == '__main__':
    main()
