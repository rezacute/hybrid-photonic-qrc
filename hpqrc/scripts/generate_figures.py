"""
Generate Figures

Load all results and generate comparison plots.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all results from directory."""
    results = []

    for filepath in results_dir.glob("**/*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)

                # Handle both single result and list of results
                if isinstance(data, list):
                    results.extend(data)
                elif isinstance(data, dict):
                    # Check if it's a result file
                    if 'metrics' in data or 'model' in data:
                        results.append(data)
        except Exception as e:
            warnings.warn(f"Could not load {filepath}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def plot_forecast_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate forecast comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Placeholder - actual implementation would use actual forecasts
    axes[0, 0].bar(['Model A', 'Model B'], [0.5, 0.6])
    axes[0, 0].set_title('RMSE Comparison')

    axes[0, 1].bar(['Model A', 'Model B'], [0.3, 0.35])
    axes[0, 1].set_title('MAE Comparison')

    axes[1, 0].scatter([1, 2, 3], [0.5, 0.6, 0.55])
    axes[1, 0].set_title('Parameter Count vs Error')

    axes[1, 1].boxplot([[0.5, 0.6, 0.55], [0.4, 0.45, 0.42]])
    axes[1, 1].set_xticklabels(['Model A', 'Model B'])
    axes[1, 1].set_title('Error Distribution')

    plt.tight_layout()
    filepath = output_dir / "forecast_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filepath}")


def plot_memory_capacity(results: dict, output_dir: Path):
    """Generate memory capacity comparison."""
    if not results:
        print("No memory results to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # STM capacity
    models = list(results.keys())
    stm = [results[m].get('stm_capacity', 0) for m in models]

    axes[0].bar(models, stm)
    axes[0].set_title('Short-Term Memory Capacity')
    axes[0].set_ylabel('MC')
    axes[0].tick_params(axis='x', rotation=45)

    # IPC
    ipc = [results[m].get('ipc_total', 0) for m in models]

    axes[1].bar(models, ipc)
    axes[1].set_title('Information Processing Capacity')
    axes[1].set_ylabel('IPC')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    filepath = output_dir / "memory_capacity.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filepath}")


def plot_ablation_heatmap(ablation_results: list[dict], output_dir: Path):
    """Generate ablation heatmap."""
    if not ablation_results:
        print("No ablation results to plot")
        return

    # Extract data
    variants = [r['variant'] for r in ablation_results]
    metrics = ['rmse', 'mae', 'mape']

    data = []
    for r in ablation_results:
        row = [r['metrics'].get(m, 0) for m in metrics]
        data.append(row)

    df = pd.DataFrame(data, index=variants, columns=metrics)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
    ax.set_title('Ablation Study: Model Performance')

    plt.tight_layout()
    filepath = output_dir / "ablation_heatmap.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filepath}")


def plot_parameter_efficiency(df: pd.DataFrame, output_dir: Path):
    """Generate parameter efficiency scatter plot."""
    if df.empty:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        # Get param counts (would need to load from actual results)
        params = [1000] * len(model_df)  # Placeholder
        errors = model_df['metrics'].apply(lambda x: x.get('rmse', 0))

        ax.scatter(params, errors, label=model, s=100, alpha=0.7)

    ax.set_xlabel('Parameters')
    ax.set_ylabel('RMSE')
    ax.set_title('Parameter Efficiency')
    ax.legend()
    ax.set_xscale('log')

    plt.tight_layout()
    filepath = output_dir / "parameter_efficiency.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filepath}")


def main():
    """Main figure generation function."""
    results_dir = Path("./outputs/results")
    memory_dir = Path("./outputs/memory")
    output_dir = Path("./outputs/figures")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    df = load_all_results(results_dir)

    if not df.empty:
        print(f"Loaded {len(df)} results")

        # Generate plots
        plot_forecast_comparison(df, output_dir)
        plot_parameter_efficiency(df, output_dir)
    else:
        print("No results found")

    # Memory benchmark results
    memory_file = memory_dir / "memory_benchmark.json"
    if memory_file.exists():
        with open(memory_file) as f:
            memory_results = json.load(f)
        plot_memory_capacity(memory_results, output_dir)

    # Ablation results
    ablation_files = list(results_dir.glob("ablation_*.json"))
    for ablation_file in ablation_files:
        with open(ablation_file) as f:
            ablation_results = json.load(f)
        plot_ablation_heatmap(ablation_results, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
