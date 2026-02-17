"""
Evaluation Script

Load results, compute aggregates, run statistical tests, print summary.
"""

import json
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.evaluation.statistical_tests import (
    cohens_d,
    wilcoxon_comparison,
)

console = Console()


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all results from directory."""
    results = []

    for filepath in results_dir.glob("*.json"):
        with open(filepath) as f:
            data = json.load(f)
            results.append(data)

    return pd.DataFrame(results)


def compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics per model."""
    aggregates = {}

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        aggregates[model] = {
            'n_experiments': len(model_df),
            'mae_mean': model_df['metrics'].apply(lambda x: x['mae']).mean(),
            'mae_std': model_df['metrics'].apply(lambda x: x['mae']).std(),
            'rmse_mean': model_df['metrics'].apply(lambda x: x['rmse']).mean(),
            'rmse_std': model_df['metrics'].apply(lambda x: x['rmse']).std(),
            'mape_mean': model_df['metrics'].apply(lambda x: x['mape']).mean(),
            'r2_mean': model_df['metrics'].apply(lambda x: x['r2']).mean(),
        }

    return aggregates


def run_statistical_tests(df: pd.DataFrame, metric: str = 'rmse') -> dict:
    """Run pairwise statistical tests."""
    models = df['model'].unique()
    tests = {}

    # Get scores per model
    scores = {}
    for model in models:
        model_df = df[df['model'] == model]
        scores[model] = model_df['metrics'].apply(lambda x: x[metric]).values

    # Pairwise comparisons
    for i, model_a in enumerate(models):
        for model_b in models[i+1:]:
            if len(scores[model_a]) > 0 and len(scores[model_b]) > 0:
                test_result = wilcoxon_comparison(scores[model_a], scores[model_b])
                tests[f"{model_a} vs {model_b}"] = test_result

    return tests


def print_summary_table(aggregates: dict):
    """Print summary table with rich."""
    table = Table(title="Model Performance Summary")

    table.add_column("Model", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAPE", justify="right")
    table.add_column("R²", justify="right")

    for model, metrics in sorted(aggregates.items(), key=lambda x: x[1]['rmse_mean']):
        table.add_row(
            model,
            str(metrics['n_experiments']),
            f"{metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}",
            f"{metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}",
            f"{metrics['mape_mean']:.2f}%",
            f"{metrics['r2_mean']:.4f}",
        )

    console.print(table)


@click.command()
@click.option('--results-dir', default='./outputs/results', help='Results directory')
@click.option('--metric', default='rmse', help='Primary metric for comparison')
@click.option('--run-tests', is_flag=True, help='Run statistical tests')
def main(results_dir: str, metric: str, run_tests: bool):
    """Main evaluation function."""
    results_path = Path(results_dir)

    if not results_path.exists():
        console.print(f"[red]Error: Results directory not found: {results_dir}[/red]")
        return

    console.print(f"\n[bold]Loading results from {results_dir}...[/bold]")
    df = load_results(results_path)

    if len(df) == 0:
        console.print("[red]No results found![/red]")
        return

    console.print(f"Loaded {len(df)} results\n")

    # Aggregate metrics
    aggregates = compute_aggregate_metrics(df)
    print_summary_table(aggregates)

    # Statistical tests
    if run_tests and len(df['model'].unique()) > 1:
        console.print("\n[bold]Statistical Tests (Wilcoxon):[/bold]")

        tests = run_statistical_tests(df, metric)

        for pair, result in tests.items():
            if 'error' not in result:
                sig = "✓" if result['significant'] else "✗"
                console.print(f"  {pair}: p={result['p_value']:.4f} {sig}")

                # Cohen's d
                model_a, model_b = pair.split(' vs ')
                scores_a = df[df['model'] == model_a]['metrics'].apply(lambda x: x[metric]).values
                scores_b = df[df['model'] == model_b]['metrics'].apply(lambda x: x[metric]).values
                d = cohens_d(scores_a, scores_b)

                console.print(f"    Effect size (d): {d:.3f}")

    # Best model
    best_model = min(aggregates.items(), key=lambda x: x[1]['rmse_mean'])
    console.print(f"\n[bold green]Best model: {best_model[0]} (RMSE: {best_model[1]['rmse_mean']:.4f})[/bold green]")


if __name__ == "__main__":
    main()
