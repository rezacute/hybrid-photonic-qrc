"""
Export Results to LaTeX

Convert results to LaTeX table format.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import click


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all results from directory."""
    results = []
    
    for filepath in results_dir.glob("*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                elif isinstance(data, dict) and 'metrics' in data:
                    results.append(data)
        except Exception:
            pass
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def format_latex_table(df: pd.DataFrame, caption: str = "Results") -> str:
    """Format results as LaTeX table."""
    if df.empty:
        return "\\textbf{No results}"
    
    # Compute aggregates
    aggregates = {}
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        
        rmse_vals = model_df['metrics'].apply(lambda x: x.get('rmse', 0))
        mae_vals = model_df['metrics'].apply(lambda x: x.get('mae', 0))
        mape_vals = model_df['metrics'].apply(lambda x: x.get('mape', 0))
        
        aggregates[model] = {
            'rmse': f"{rmse_vals.mean():.4f} $\\pm$ {rmse_vals.std():.4f}",
            'mae': f"{mae_vals.mean():.4f} $\\pm$ {mae_vals.std():.4f}",
            'mape': f"{mape_vals.mean():.2f}",
            'n': len(model_df),
        }
    
    # Build table
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & RMSE & MAE & MAPE ($\\%$) & $N$ \\\\")
    lines.append("\\midrule")
    
    # Sort by RMSE
    sorted_models = sorted(
        aggregates.items(),
        key=lambda x: float(x[1]['rmse'].split()[0])
    )
    
    for model, metrics in sorted_models:
        lines.append(f"{model} & {metrics['rmse']} & {metrics['mae']} & {metrics['mape']} & {metrics['n']} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:results}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def format_ablation_table(ablation_results: List[Dict]) -> str:
    """Format ablation results as LaTeX table."""
    if not ablation_results:
        return "\\textbf{No ablation results}"
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Variant & RMSE & MAE & Params \\\\")
    lines.append("\\midrule")
    
    # Sort by RMSE
    sorted_results = sorted(
        ablation_results,
        key=lambda x: x.get('metrics', {}).get('rmse', float('inf'))
    )
    
    for result in sorted_results:
        variant = result.get('variant', 'unknown')
        rmse = result.get('metrics', {}).get('rmse', 0)
        mae = result.get('metrics', {}).get('mae', 0)
        params = result.get('params', {}).get('total', 0)
        
        lines.append(f"{variant} & {rmse:.4f} & {mae:.4f} & {params:,} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def format_memory_table(memory_results: Dict) -> str:
    """Format memory capacity results as LaTeX table."""
    if not memory_results:
        return "\\textbf{No memory results}"
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Memory Capacity Comparison}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Model & STM & IPC \\\\")
    lines.append("\\midrule")
    
    sorted_models = sorted(
        memory_results.items(),
        key=lambda x: x[1].get('stm_capacity', 0),
        reverse=True
    )
    
    for model, result in sorted_models:
        if 'error' not in result:
            stm = result.get('stm_capacity', 0)
            ipc = result.get('ipc_total', 0)
            lines.append(f"{model} & {stm:.3f} & {ipc:.3f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:memory}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


@click.command()
@click.option('--results-dir', default='./outputs/results', help='Results directory')
@click.option('--output', default='./outputs/tables.tex', help='Output file')
@click.option('--format', default='all', type=click.Choice(['main', 'ablation', 'memory', 'all']))
def main(results_dir: str, output: str, format: str):
    """Export results to LaTeX."""
    results_path = Path(results_dir)
    output_path = Path(output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    latex_lines = []
    latex_lines.append("% Auto-generated LaTeX tables")
    latex_lines.append("% Generated by export_results.py")
    latex_lines.append("")
    
    # Main results
    if format in ['main', 'all']:
        df = load_results(results_path)
        if not df.empty:
            latex_lines.append(format_latex_table(df, "Model Comparison"))
            latex_lines.append("\n\n")
    
    # Ablation results
    if format in ['ablation', 'all']:
        ablation_files = list(results_path.glob("ablation_*.json"))
        for ablation_file in ablation_files:
            with open(ablation_file) as f:
                ablation_results = json.load(f)
            latex_lines.append(format_ablation_table(ablation_results))
            latex_lines.append("\n\n")
    
    # Memory results
    if format in ['memory', 'all']:
        memory_file = Path("./outputs/memory/memory_benchmark.json")
        if memory_file.exists():
            with open(memory_file) as f:
                memory_results = json.load(f)
            latex_lines.append(format_memory_table(memory_results))
    
    # Write output
    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX tables saved to {output_path}")


if __name__ == "__main__":
    main()
