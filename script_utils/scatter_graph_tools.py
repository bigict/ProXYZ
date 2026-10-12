import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, linregress
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd

from proxyz.utils import dict2object


@click.command(context_settings={"show_default": True})
@click.argument("y_files", type=click.Path(), nargs=-1)
@click.option("--x_file", type=click.Path(), required=True, help="X-axis data file.")
@click.option(
    "--column_name", type=str, default="P@L", help="Column name containing data."
)
@click.option(
    "--color_name", type=str, default="PPL", help="Headmap name containing data."
)
@click.option(
    "--color_sigma_n", type=float, default=1, help="Headmap (vmin, max) containing data."
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    # 1. Configuration
    args = dict2object(**args)

    # 2. Load baseline data
    df_base = pd.read_csv(args.x_file)
    num_targets = len(args.y_files)

    # 3. Dynamic layout math: set up a grid based on number of target files
    # Increased width slightly (8 instead of 7) to give clean breathing room for colorbars
    fig, axes = plt.subplots(
        nrows=1, ncols=num_targets, figsize=(8 * num_targets, 7), squeeze=False
    )

    # 4. Loop through each target file in the arbitrary list
    for idx, y_file in enumerate(args.y_files):
        df_target = pd.read_csv(y_file)

        # Merge dataframes on 'id' to sync matching index entries
        merged_df = pd.merge(
            df_base, df_target, on="id", suffixes=("_base", f"_target")
        )

        x_data = merged_df[f"{args.column_name}_base"]
        y_data = merged_df[f"{args.column_name}_target"]
        
        # Pull the absolute 'PPL' column from the target dataset for color variations
        # Uses .get() to provide a safe fallback array of ones if the column doesn't exist
        has_ppl = True
        if args.color_name in merged_df.columns:
            color_data = merged_df[args.color_name]
        elif f"{args.color_name}_target" in merged_df.columns:
            color_data = merged_df[f"{args.color_name}_target"]
        else:
            if args.verbose:
                print(f"Warning: '{args.color_name}' column missing in {y_file}. Defaulting to uniform color.")
            color_data = np.ones(len(y_data))
            has_ppl = False

        # --- DYNAMIC SIGMA BOUNDS CALCULATION ---
        if has_ppl:
            ppl_mean = color_data.mean()
            ppl_std = color_data.std()
            
            # Use max(0, ...) to ensure vmin doesn't drop below zero if your metric can't be negative
            calculated_vmin = max(0.0, ppl_mean - args.color_sigma_n * ppl_std)
            calculated_vmax = ppl_mean + args.color_sigma_n * ppl_std
        else:
            calculated_vmin, calculated_vmax = 0.0, 1.0

        # Statistical telemetry calculation blocks
        mean_x = x_data.mean()
        mean_y = y_data.mean()
        r_value, _ = pearsonr(x_data, y_data)
        r2 = r2_score(x_data, y_data)
        slope, intercept, _, _, _ = linregress(x_data, y_data)

        # Calculate unique bounds for axis symmetry on this plot
        min_val = min(x_data.min(), y_data.min()) * 0.95
        max_val = max(x_data.max(), y_data.max()) * 1.05

        # Identify active subplot cell framework
        main_ax = axes[0, idx]

        # Create marginal distribution axes directly inside the subplot spaces using inset tracking
        # Top histogram axis location locator logic mapping bounds inside cell boundaries
        ax_top = main_ax.inset_axes([0, 1.02, 1, 0.15], sharex=main_ax)
        ax_right = main_ax.inset_axes([1.02, 0, 0.15, 1], sharey=main_ax)

        # --- A. Plot Central Panel Metrics ---
        # Center scatter plots with color depth scaling mapped explicitly to PPL data
        scatter = main_ax.scatter(
            x_data,
            y_data,
            c=color_data,           # Maps color intensity dynamically to the PPL column values
            vmin=calculated_vmin,   # Lower bound: mean - 1 sigma
            vmax=calculated_vmax,   # Upper bound: mean + 1 sigma
            cmap="Blues",           # Sequential colormap transitioning light to deep blue
            alpha=0.85,             # High opacity keeps lighter depth values visible
            edgecolors="#333333",   # Dark gray edge crispness so lighter values don't dissolve
            linewidths=0.6,
            s=50,
            label="Data Points",
            zorder=5
        )

        # Center line tracking profiles (y=x line)
        main_ax.plot(
            [min_val, max_val], [min_val, max_val],
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label="y = x"
        )

        # Linear trendline mapping configurations
        x_trend = np.linspace(min_val, max_val, 100)
        y_trend = slope * x_trend + intercept
        main_ax.plot(
            x_trend,
            y_trend,
            color="#2ca02c",
            linestyle="-",
            linewidth=1.8,
            label=f"Fit: y={slope:.2f}x+{intercept:.2f}"
        )

        # Center crosshairs mean positioning arrays
        main_ax.axvline(mean_x, color="gray", linestyle=":", linewidth=1)
        main_ax.axhline(mean_y, color="gray", linestyle=":", linewidth=1)

        # --- B. Plot Marginal Density Bar Graphs (Histograms) ---
        # Top density strip
        sns.histplot(
            x=x_data,
            ax=ax_top,
            color="#7f7f7f",
            alpha=0.4,
            bins=10,
            kde=True,
            zorder=3
        )
        ax_top.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            bottom=False,
            left=False,
            labelleft=False
        )
        ax_top.set_ylabel("")
        ax_top.set_xlabel("")

        # Right density strip
        sns.histplot(
            y=y_data,
            ax=ax_right,
            color="#7f7f7f",
            alpha=0.4,
            bins=10,
            kde=True,
            zorder=3
        )
        ax_right.tick_params(
            axis="both",
            which="both",
            labelleft=False,
            left=False,
            bottom=False,
            labelbottom=False
        )
        ax_right.set_ylabel("")
        ax_right.set_xlabel("")

        # --- C. Labeling and Aesthetics ---
        main_ax.set_xlim(min_val, max_val)
        main_ax.set_ylim(min_val, max_val)
        main_ax.set_xlabel(f"Base: {args.x_file}", fontsize=11)
        main_ax.set_ylabel(f"Target: {y_file}", fontsize=11)
        main_ax.set_title(
            f"Comparison vs {y_file}", fontsize=12, fontweight="bold", pad=25
        )
        main_ax.grid(True, linestyle="-", alpha=0.15, zorder=0)
        main_ax.legend(loc="lower right", fontsize=9)

        # Inject unique text statistics data fields safely
        stats_text = (
            f"Mean (Base): {mean_x:.3f}\n"
            f"Mean (Target): {mean_y:.3f}\n"
            f"Pearson $r$: {r_value:.3f}\n"
            f"$R^2$ Score: {r2:.3f}"
        )
        main_ax.text(
            0.04,
            0.96,
            stats_text,
            transform=main_ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray"
            )
        )

        # --- D. Colorbar Append Mechanism ---
        # Appends colorbar dynamically outside both the main panel and right histogram panel
        if has_ppl:
            cbar = fig.colorbar(scatter, ax=[main_ax, ax_right], pad=0.05, shrink=0.15)
            cbar.set_label(f"{args.color_name} Value Depth", fontsize=5, fontweight="bold")
            cbar.ax.tick_params(labelsize=8)

    # 5. Spacing Engine Layout Corrections
    # tight_layout balances inner axis tags, then subplots_adjust moves everything down slightly 
    # to shield the individual plot areas from overlapping the main super-title.
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, wspace=0.35)

    # 6. Clean layout geometry calculations and file rendering loops
    plt.suptitle(
        f"Multi-File Scatter Comparisons with Distribution Densities: {args.column_name}",
        y=1.08,
        fontsize=15,
        fontweight="bold"
    )
    output_filename = "multi_scatter_density_grid.png"

    # Save the final structural visual file cleanly
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(
        f"Success! The comprehensive analysis grid has been exported as '{output_filename}'."
    )

    plt.show()


if __name__ == "__main__":
    main()
