import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results(dataset_name):
    """Load all CSV results from the results folder structure for a specific dataset."""
    # The results folders (tahoe/essential) are located in the same directory as this script
    results_dir = Path(__file__).parent / dataset_name
    
    methods = ["contextmean", "knn", "mlpcontinuous", "mlponehot", "st"]
    
    # Define cell types based on dataset
    if dataset_name == "tahoe":
        cell_types = ["CVCL_1097", "CVCL_1285", "CVCL_1098", "CVCL_0334", "CVCL_0480"]
    elif dataset_name == "essential":
        cell_types = ["hepg2", "k562", "rpe1", "jurkat"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    all_data = []
    
    for method in methods:
        for cell_type in cell_types:
            # Determine potential directory paths
            # For Essential, models are trained/saved per cell line (e.g. mlpcontinuous_hepg2)
            # For Tahoe, models are shared (e.g. mlpcontinuous)
            potential_dirs = []

            if dataset_name == "essential":
                potential_dirs.append(results_dir / f"{method}_{cell_type}")
            
            # Always check the base method directory as fallback (or primary for Tahoe)
            potential_dirs.append(results_dir / method)

            csv_file = None
            for d in potential_dirs:
                f = d / f"{cell_type}_agg_results.csv"
                if f.exists():
                    csv_file = f
                    break

            if csv_file is None:
                print(f"Warning: Results not found for {method} on {cell_type}")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Check if required metrics exist and filter for "mean" statistic
                if 'discrimination_score_l1' in df.columns and 'mae' in df.columns and 'overlap_at_N' in df.columns and 'statistic' in df.columns:
                    # Filter for rows where statistic equals "mean"
                    mean_rows = df[df['statistic'] == 'mean']
                    if len(mean_rows) > 0:
                        data_row = {
                            'method': method,
                            'cell_type': cell_type,
                            'discrimination_score_l1': mean_rows['discrimination_score_l1'].iloc[0],
                            'mae': mean_rows['mae'].iloc[0],
                            'overlap_at_N': mean_rows['overlap_at_N'].iloc[0]
                        }
                        all_data.append(data_row)
                        print(f"Loaded data for {method}/{cell_type}: disc_l1={data_row['discrimination_score_l1']:.3f}, mae={data_row['mae']:.3f}, overlap_N={data_row['overlap_at_N']:.3f}")
                    else:
                        print(f"Warning: No 'mean' statistic found in {csv_file}")
                else:
                    print(f"Warning: Required metrics not found in {csv_file}")
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    return pd.DataFrame(all_data)

def create_comparison_plots(df, dataset_name):
    """Create comparison plots for each cell type."""
    cell_types = df['cell_type'].unique()
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for cell_type in cell_types:
        cell_data = df[df['cell_type'] == cell_type]
        
        if len(cell_data) == 0:
            print(f"No data available for {cell_type}")
            continue
        
        # Create subplot with 1 row, 3 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Method Comparison for {cell_type} ({dataset_name})', fontsize=16, fontweight='bold')
        
        # Plot 1: discrimination_score_l1
        methods = cell_data['method'].values
        disc_scores = cell_data['discrimination_score_l1'].values
        
        bars1 = ax1.bar(methods, disc_scores, alpha=0.7)
        ax1.set_title('Discrimination Score L1\n(Higher is Better)', fontweight='bold')
        ax1.set_ylabel('Discrimination Score L1')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars1, disc_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: mae
        mae_scores = cell_data['mae'].values
        
        bars2 = ax2.bar(methods, mae_scores, alpha=0.7, color='orange')
        ax2.set_title('Mean Absolute Error\n(Lower is Better)', fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars2, mae_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: overlap_at_N
        overlap_scores = cell_data['overlap_at_N'].values
        
        bars3 = ax3.bar(methods, overlap_scores, alpha=0.7, color='green')
        ax3.set_title('Overlap at N\n(Higher is Better)', fontweight='bold')
        ax3.set_ylabel('Overlap at N')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars3, overlap_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create plots directory if it doesn't exist
        plots_dir = Path("plots") / dataset_name
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the plot
        output_file = plots_dir / f"{cell_type}_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        
        plt.close()

def create_summary_plot(df, dataset_name):
    """Create a summary plot averaging results across all cell types."""
    if len(df) == 0:
        print("No data available for summary plot")
        return
    
    # Calculate average metrics across all cell types for each method
    avg_metrics = df.groupby('method').agg({
        'discrimination_score_l1': 'mean',
        'mae': 'mean',
        'overlap_at_N': 'mean'
    }).reset_index()
    
    # Map method names to display names (dataset-specific)
    if dataset_name == "essential":
        method_name_map = {
            'contextmean': 'Train Mean',
            'knn': 'kNN',
            'mlpcontinuous': 'MLP (WaveGC)',
            'mlponehot': 'MLP (One-Hot)',
            'st': 'STATE'
        }
    else:  # tahoe
        method_name_map = {
            'contextmean': 'Train Mean',
            'knn': 'kNN',
            'mlpcontinuous': 'MLP (ChatGPT)',
            'mlponehot': 'MLP (One-Hot)',
            'st': 'STATE'
        }
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplot with 1 row, 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Map method names for display
    display_methods = [method_name_map.get(method, method) for method in avg_metrics['method'].values]
    
    # Plot 1: discrimination_score_l1 (higher is better)
    disc_scores = avg_metrics['discrimination_score_l1'].values
    bars1 = ax1.bar(display_methods, disc_scores, alpha=0.7)
    
    # Add individual dots for each cell line
    for i, method in enumerate(avg_metrics['method'].values):
        method_data = df[df['method'] == method]
        cell_line_scores = method_data['discrimination_score_l1'].values
        # Add jitter if there are multiple points
        if len(cell_line_scores) > 1:
            x_positions = [i + np.random.uniform(-0.1, 0.1) for _ in range(len(cell_line_scores))]
        else:
            x_positions = [i] * len(cell_line_scores)
        ax1.scatter(x_positions, cell_line_scores, color='black', s=30, alpha=0.3, zorder=3)
    
    ax1.set_title('Discrimination Score L1 ↑')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, disc_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: overlap_at_N (higher is better)
    overlap_scores = avg_metrics['overlap_at_N'].values
    bars2 = ax2.bar(display_methods, overlap_scores, alpha=0.7, color='green')
    
    # Add individual dots for each cell line
    for i, method in enumerate(avg_metrics['method'].values):
        method_data = df[df['method'] == method]
        cell_line_scores = method_data['overlap_at_N'].values
        # Add jitter if there are multiple points
        if len(cell_line_scores) > 1:
            x_positions = [i + np.random.uniform(-0.1, 0.1) for _ in range(len(cell_line_scores))]
        else:
            x_positions = [i] * len(cell_line_scores)
        ax2.scatter(x_positions, cell_line_scores, color='black', s=30, alpha=0.3, zorder=3)
    
    ax2.set_title('Overlap at N ↑')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, score in zip(bars2, overlap_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: mae (lower is better)
    mae_scores = avg_metrics['mae'].values
    bars3 = ax3.bar(display_methods, mae_scores, alpha=0.7, color='orange')
    
    # Add individual dots for each cell line
    for i, method in enumerate(avg_metrics['method'].values):
        method_data = df[df['method'] == method]
        cell_line_scores = method_data['mae'].values
        # Add jitter if there are multiple points
        if len(cell_line_scores) > 1:
            x_positions = [i + np.random.uniform(-0.1, 0.1) for _ in range(len(cell_line_scores))]
        else:
            x_positions = [i] * len(cell_line_scores)
        ax3.scatter(x_positions, cell_line_scores, color='black', s=30, alpha=0.3, zorder=3)
    
    ax3.set_title('MAE ↓')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars3, mae_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("plots") / dataset_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    output_file = plots_dir / "average_comparison.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {output_file}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"AVERAGE PERFORMANCE ACROSS ALL CELL TYPES ({dataset_name})")
    print("="*80)
    for _, row in avg_metrics.iterrows():
        print(f"{row['method']:15} | Disc L1: {row['discrimination_score_l1']:.3f} | MAE: {row['mae']:.3f} | Overlap N: {row['overlap_at_N']:.3f}")

def create_summary_table(df, dataset_name):
    """Create a summary table of all results."""
    if len(df) == 0:
        print("No data available for summary table")
        return
    
    # Pivot the data for better visualization
    disc_pivot = df.pivot(index='cell_type', columns='method', values='discrimination_score_l1')
    mae_pivot = df.pivot(index='cell_type', columns='method', values='mae')
    overlap_pivot = df.pivot(index='cell_type', columns='method', values='overlap_at_N')
    
    print("\n" + "="*80)
    print(f"DISCRIMINATION SCORE L1 (Higher is Better) - {dataset_name}")
    print("="*80)
    print(disc_pivot.round(3))
    
    print("\n" + "="*80)
    print(f"MEAN ABSOLUTE ERROR (Lower is Better) - {dataset_name}")
    print("="*80)
    print(mae_pivot.round(3))
    
    print("\n" + "="*80)
    print(f"OVERLAP AT N (Higher is Better) - {dataset_name}")
    print("="*80)
    print(overlap_pivot.round(3))
    
    # Save summary tables to CSV
    disc_pivot.round(3).to_csv(f"{dataset_name}_discrimination_score_l1_summary.csv")
    mae_pivot.round(3).to_csv(f"{dataset_name}_mae_summary.csv")
    overlap_pivot.round(3).to_csv(f"{dataset_name}_overlap_at_N_summary.csv")
    print(f"\nSaved summary tables: {dataset_name}_discrimination_score_l1_summary.csv, {dataset_name}_mae_summary.csv, {dataset_name}_overlap_at_N_summary.csv")

def main():
    """Main function to run the comparison analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare results across methods for a dataset")
    parser.add_argument("--dataset", choices=["tahoe", "essential"], required=True,
                       help="Dataset to analyze")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    
    print(f"Loading results from CSV files for {dataset_name} dataset...")
    df = load_results(dataset_name)
    
    if len(df) == 0:
        print(f"No data found for {dataset_name}. Please check that the results folder structure is correct.")
        return
    
    print(f"Loaded {len(df)} result entries")
    print(f"Methods found: {sorted(df['method'].unique())}")
    print(f"Cell types found: {sorted(df['cell_type'].unique())}")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(df, dataset_name)
    
    # Create summary plot
    print("\nCreating summary plot...")
    create_summary_plot(df, dataset_name)
    
    # Create summary table
    create_summary_table(df, dataset_name)
    
    print(f"\nAnalysis complete for {dataset_name}!")

if __name__ == "__main__":
    main()
