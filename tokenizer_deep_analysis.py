"""
tokenizer_deep_analysis.py
Part 0: Compute Length-Stratified SSE Lift with Binomial/Fisher exact tests.
Part 1: All-Helix Fraction Curve (visualizing 3.6-residue periodicity)
Part 2: Boundary Alignment Lift Report (with exact analytical baseline)

Dependencies: pip install biotite transformers numpy matplotlib seaborn tqdm scipy glob
"""
from collections import defaultdict, Counter
import glob
import random

from biotite.sequence import ProteinSequence
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb
import click
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from proxyz.utils import dict2object
from proxyz.data.utils import lines, opener


# Map biotite SSE codes to 4 simplified classes.
# biotite annotate_sse mainly returns 'H'/'E'/'C', but we also handle
# the finer 8-class DSSP codes for robustness.
SSE_MAP = {
    "H": "Helix", "A": "Helix", "G": "Helix", "I": "Helix",
    "E": "Sheet", "B": "Sheet",
    "T": "Turn",  "S": "Turn",
    "C": "Coil",  "-": "Coil", " ": "Coil",
}

categories = ["Helix", "Sheet", "Turn", "Coil"]

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def load_structure(file_path):
    """Load a PDB or PDBx/mmCIF file and keep protein atoms only."""
    if file_path.endswith(".cif") or file_path.endswith(".pdbx"):
        pdbx_file = pdbx.PDBxFile.read(file_path)
        structure = pdbx.get_structure(pdbx_file, model=1)
    else:
        pdb_file = pdb.PDBFile.read(file_path)
        structure = pdb.get_structure(pdb_file, model=1)
    # Keep protein atoms only
    structure = structure[struc.filter_amino_acids(structure)]
    return structure


def extract_seq_and_sse(structure, chain_id):
    """Extract single-letter sequence and aligned SSE labels for one chain."""
    # Subset atoms belonging to the target chain
    mask = structure.chain_id == chain_id
    sub = structure[mask]

    # annotate_sse returns an array aligned to residues of the given chain
    sse_raw = struc.annotate_sse(sub)

    # Convert residue names to one-letter codes
    _, aa_names = struc.get_residues(sub)
    seq_chars = []
    for name in aa_names:
        try:
            seq_chars.append(ProteinSequence.convert_letter_3to1(name))
        except Exception:
            seq_chars.append("X")  # fallback for non-standard residues
    seq = "".join(seq_chars)
    sse_labels = [SSE_MAP.get(s.upper(), "Coil") for s in sse_raw]

    # Guard against length mismatch between sequence and SSE
    min_len = min(len(seq), len(sse_labels))
    return seq[:min_len], sse_labels[:min_len]


# ==========================================
# PART 0: Length-Stratified SSE Lift
# ==========================================
def compute_lift_with_significance(
    tokenizer,
    data_files,
    smoothing_alpha=1.0,
    min_samples=30,
    min_baseline_count=20,
    alpha_level=0.05,
):
    """
    Compute Lift matrix with Binomial and Fisher exact test p-values.
    
    Returns:
        lift_matrix, pval_binom_matrix, pval_fisher_matrix, 
        sig_annotations, mask, lengths, categories, bpe_totals
    """
    special_tokens = set(tokenizer.all_special_tokens)
    max_token_len = max(
        len(vocab) for vocab in tokenizer.get_vocab() if vocab not in special_tokens
    )

    lengths = list(range(1, max_token_len + 1))
    num_cats = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Accumulators
    bpe_counts = np.zeros((max_token_len, num_cats), dtype=np.int64)
    bpe_totals = np.zeros(max_token_len, dtype=np.int64)
    rand_counts = np.zeros((max_token_len, num_cats), dtype=np.int64)
    rand_totals = np.zeros(max_token_len, dtype=np.int64)

    for data_file in tqdm(data_files, desc="Accumulating counts"):
        structure = load_structure(data_file)
        for cid in np.unique(structure.chain_id):
            seq, sse_labels = extract_seq_and_sse(structure, cid)
            L = len(seq)
            if L < 15:  # peptide
                continue

            # Tokenize with character-level offsets for alignment
            enc = tokenizer(seq, return_offsets_mapping=True, add_special_tokens=False)

            # BPE statistics
            for start, end in enc.offset_mapping:
                tok_len = end - start
                if 1 <= tok_len <= max_token_len:
                    region_sse = sse_labels[start:end]
                    if region_sse:
                        major = Counter(region_sse).most_common(1)[0][0]
                        if major in cat_to_idx:
                            bpe_counts[tok_len - 1, cat_to_idx[major]] += 1
                            bpe_totals[tok_len - 1] += 1

            # Random baseline: sample continuous segments
            for l in lengths:
                if L - l < 1:
                    continue
                n_samples = min(100, L - l)
                starts = random.sample(range(L - l + 1), n_samples)
                for s in starts:
                    region_sse = sse_labels[s:s + l]
                    if region_sse:
                        major = Counter(region_sse).most_common(1)[0][0]
                        if major in cat_to_idx:
                            rand_counts[l - 1, cat_to_idx[major]] += 1
                            rand_totals[l - 1] += 1

    # === Compute Lift, P-values, and Significance Annotations ===
    lift_matrix = np.full((max_token_len, num_cats), np.nan)
    pval_binom = np.full((max_token_len, num_cats), np.nan)
    pval_fisher = np.full((max_token_len, num_cats), np.nan)
    sig_annotations = np.empty((max_token_len, num_cats), dtype='U3')
    sig_annotations[:] = ''

    # Reliability mask
    mask = bpe_totals < min_samples

    # Total number of tests for Bonferroni correction
    valid_tests = np.sum(~mask) * num_cats
    corrected_alpha = alpha_level / max(valid_tests, 1)

    for i in range(max_token_len):
        if mask[i]:
            continue
        n_bpe = int(bpe_totals[i])
        n_rand = int(rand_totals[i])

        for j in range(num_cats):
            k_bpe = int(bpe_counts[i, j])
            k_rand = int(rand_counts[i, j])

            # Probabilities with Laplace smoothing (for Lift only)
            p_bpe = (k_bpe + smoothing_alpha) / (n_bpe + smoothing_alpha * num_cats)
            p_rand = (k_rand + smoothing_alpha) / (n_rand + smoothing_alpha * num_cats)
            lift_matrix[i, j] = p_bpe / p_rand

            # --- Binomial Test ---
            # H0: BPE tokens of this length have SSE probability == p_rand (unsmoothed)
            p_rand_raw = k_rand / n_rand if n_rand > 0 else 0.0
            if n_bpe > 0 and 0 < p_rand_raw < 1:
                # Two-sided binomial test
                pval_binom[i, j] = stats.binomtest(k_bpe, n_bpe, p_rand_raw).pvalue
            else:
                pval_binom[i, j] = 1.0

            # --- Fisher's Exact Test ---
            # 2x2 contingency table:
            #              Target_SSE | Other_SSE
            # BPE tokens:   k_bpe     | n_bpe - k_bpe
            # Rand segs:    k_rand    | n_rand - k_rand
            table = np.array([
                [k_bpe, n_bpe - k_bpe],
                [k_rand, n_rand - k_rand]
            ])
            if np.all(table >= 0):
                _, pval_fisher[i, j] = stats.fisher_exact(table)
            else:
                pval_fisher[i, j] = 1.0

            # --- Significance Annotation ---
            # Use the MORE CONSERVATIVE (larger) p-value between the two tests
            conservative_p = max(pval_binom[i, j], pval_fisher[i, j])
            if conservative_p < corrected_alpha / 10:
                sig_annotations[i, j] = "***"
            elif conservative_p < corrected_alpha / 5:
                sig_annotations[i, j] = "**"
            elif conservative_p < corrected_alpha:
                sig_annotations[i, j] = "*"
            else:
                sig_annotations[i, j] = "ns"

    # NEW: mask cells whose RANDOM BASELINE count is too low to trust.
    # This kills the Turn-row artifact (k_rand ~ 0 -> Lift explosion).
    baseline_mask = rand_counts < min_baseline_count      # (max_token_len, num_cats)
    cell_mask = mask[:, np.newaxis] | baseline_mask       # row mask broadcast to 2D

    lift_matrix[cell_mask] = np.nan
    sig_annotations[cell_mask] = ""

    """Plot Lift heatmap overlaid with significance stars."""

    """mask_2d is now (max_token_len, num_cats); NaN cells render blank."""
    masked_lift = lift_matrix          # already NaN where unreliable
    masked_sig = sig_annotations

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.30, wspace=0.05
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main heatmap
    sns.heatmap(masked_lift.T, xticklabels=lengths, yticklabels=categories,
                cmap="RdBu_r", center=1.0, vmin=0.5, vmax=2.0,
                annot=masked_sig.T, fmt='', cbar_kws={'label': 'Lift'},
                ax=ax_main, linewidths=.5, linecolor='lightgray',
                annot_kws={"size": 14, "weight": "bold"})

    # Overlay numeric Lift values as small text below stars
    for i in range(len(lengths)):
        for j in range(len(categories)):
            val = masked_lift[i, j]
            if not np.isnan(val):
                ax_main.text(i + 0.5, j + 0.72, f'{val:.2f}',
                             ha='center', va='center', fontsize=7,
                             color='gray', alpha=0.8)

    ax_main.set_title("Length-Stratified SSE Lift with Significance", fontsize=13)
    # Move the legend line OUT of the axes to avoid overlapping the top marginal
    fig.text(0.5, 0.02,
             "* p<α, ** p<α/5, *** p<α/10, ns = not significant; "
             "Bonferroni-corrected; blank = insufficient BPE or baseline samples",
             ha='center', fontsize=9)

    # FIX: seaborn heatmap x-axis is categorical (0..N-1), so plot bars
    # at categorical positions, NOT at data positions 1..N.
    ax_top.bar(
        np.arange(len(lengths)), bpe_totals, color='steelblue', edgecolor='white'
    )
    ax_top.set_yscale('log')
    ax_top.set_ylabel("Token Count\n(Log)", fontsize=10)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Median is robust to any residual outlier cells
    mean_lifts = np.nanmedian(masked_lift.T, axis=1)
    ax_right.barh(categories, mean_lifts, color='coral', edgecolor='white')
    ax_right.set_xlabel("Median Lift", fontsize=10)
    ax_right.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_right.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("part0_lift_heatmap_with_significance.png", dpi=150, bbox_inches='tight')


# ==========================================
# PART 1: All-Helix Fraction Periodicity
# ==========================================
def analyze_all_helix_periodicity(tokenizer, data_files):
    """Compute pure-helix fractions for BPE vs Exact Analytical Baseline."""
    special_tokens = set(tokenizer.all_special_tokens)
    max_token_len = max(
        len(vocab) for vocab in tokenizer.get_vocab() if vocab not in special_tokens
    )
    lengths = np.arange(1, max_token_len + 1)
    
    bpe_pure_h = np.zeros(max_token_len, dtype=np.int64)
    bpe_total = np.zeros(max_token_len, dtype=np.int64)
    
    rand_pure_h = np.zeros(max_token_len, dtype=np.int64)
    rand_total = np.zeros(max_token_len, dtype=np.int64)
    
    for data_file in tqdm(data_files, desc="Part 1: Helix Periodicity"):
        structure = load_structure(data_file)
        for cid in np.unique(structure.chain_id):
            seq, sse_labels = extract_seq_and_sse(structure, cid)
            L = len(seq)
            if L < 15:  # peptide
                continue

            # Exact analytical baseline: sliding window of all segments
            is_helix = np.array([s == "Helix" for s in sse_labels], dtype=int)
            # Prefix sum for O(1) range sum queries
            cum_helix = np.concatenate(([0], np.cumsum(is_helix)))

            for l in lengths:
                total_segs = L - l + 1
                if total_segs <= 0:
                    continue

                # Count segments of length l that are 100% Helix
                # sum in window is cum_helix[l:] - cum_helix[:-l]
                window_sums = cum_helix[l:] - cum_helix[:-l]
                pure_h_count = np.sum(window_sums == l)

                rand_pure_h[l-1] += pure_h_count
                rand_total[l-1] += total_segs

            # BPE statistics
            enc = tokenizer(seq, return_offsets_mapping=True, add_special_tokens=False)
            for start, end in enc.offset_mapping:
                l = end - start
                if 1 <= l <= max_token_len:
                    bpe_total[l-1] += 1
                    # Check if this specific BPE token is 100% Helix
                    if (cum_helix[end] - cum_helix[start]) == l:
                        bpe_pure_h[l-1] += 1

    # Compute fractions
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_bpe = np.where(bpe_total > 0, bpe_pure_h / bpe_total, 0)
        frac_rand = np.where(rand_total > 0, rand_pure_h / rand_total, 0)
        lift = np.where(frac_rand > 0, frac_bpe / frac_rand, np.nan)

    MIN_TOKEN_COUNT = 30   # mask lengths with too few BPE tokens

    valid = bpe_total >= MIN_TOKEN_COUNT
    frac_bpe_plot = np.where(valid, frac_bpe, np.nan)   # kills the fake 0.0 at len 6
    lift_plot = np.where(valid, lift, np.nan)

    ci_low = np.full(max_token_len, np.nan)
    ci_high = np.full(max_token_len, np.nan)
    stars = [""] * max_token_len

    for i in range(max_token_len):
        if not valid[i]:
            continue
        k, n = int(bpe_pure_h[i]), int(bpe_total[i])
        ci_low[i], ci_high[i] = wilson_ci(k, n)
        p0 = frac_rand[i]
        if 0 < p0 < 1:
            # binomtest requires scipy >= 1.7; use stats.binom_test on older versions
            pval = min(stats.binomtest(k, n, p0).pvalue * max_token_len, 1.0)
            stars[i] = ('***' if pval < 0.001 else '**' if pval < 0.01
                        else '*' if pval < 0.05 else 'ns')

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Line plot for fractions
    ax1.plot(lengths, frac_rand, "o--", color="gray", label="Random Baseline (Exact)", linewidth=2)
    ax1.plot(lengths, frac_bpe, "s-", color="#e74c3c", label="ProXYZ BPE", linewidth=2, markersize=8)
    
    ax1.set_xlabel("Token / Segment Length (AA)", fontsize=13)
    ax1.set_ylabel("Fraction of 100% Pure Helix Tokens", fontsize=13)
    ax1.set_title("All-Helix Fraction vs Length (3.6-residue Periodicity Check)", fontsize=15)
    ax1.set_xticks(lengths)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # BPE line WITH error bars (replaces the plain ax1.plot for BPE)
    ax1.errorbar(lengths, frac_bpe_plot,
                 yerr=[frac_bpe_plot - ci_low, ci_high - frac_bpe_plot],
                 fmt='s-', color='#e74c3c', label='ProXYZ BPE',
                 linewidth=2, markersize=8, capsize=4)

    # Annotate significance + sample size above each valid point
    for i in range(max_token_len):
        if valid[i]:
            ax1.annotate(f'{stars[i]} (n={bpe_total[i]})',
                         (lengths[i], frac_bpe_plot[i]),
                         textcoords='offset points', xytext=(0, 14),
                         ha='center', fontsize=8, color='darkred')
    
    # Secondary y-axis for Lift
    ax2 = ax1.twinx()
    ax2.bar(lengths + 0.2, lift_plot, width=0.4, color="steelblue", alpha=0.4, label="Lift (BPE / Rand)")
    ax2.set_ylabel("Lift Factor", color="steelblue", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.axhline(y=1.0, color="blue", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig("part1_all_helix_periodicity.png", dpi=150)
    plt.show()

# ==========================================
# PART 2: Boundary Alignment Lift (Exact)
# ==========================================
def compute_boundary_alignment_lift(tokenizer, data_files, max_dist=4):
    """Compute distance CDF and Lift using exact geometric probabilities."""
    dist_bins = np.arange(0, max_dist + 1)
    bpe_hits = np.zeros(max_dist + 1, dtype=np.int64)
    bpe_total = 0
    
    # Accumulate exact random probabilities weighted by BPE boundary counts
    rand_expected_hits = np.zeros(max_dist + 1, dtype=float)
    
    for data_file in tqdm(data_files, desc="Part 2: Boundary Alignment"):
        structure = load_structure(data_file)
        for cid in np.unique(structure.chain_id):
            seq, sse_labels = extract_seq_and_sse(structure, cid)
            L = len(seq)
            if L < 15:
                continue
            
            # 1. Identify structure boundaries (SSE transitions)
            struct_bounds = set()
            for i in range(1, L):
                if sse_labels[i] != sse_labels[i-1]:
                    struct_bounds.add(i)
            
            if not struct_bounds:
                continue
                
            struct_bounds_arr = np.array(sorted(list(struct_bounds)))
            
            # 2. Calculate exact random probabilities for each distance d
            # P_rand(d) = (Number of valid positions within distance d of ANY struct bound) / L
            p_rand = np.zeros(max_dist + 1)
            for d in dist_bins:
                covered = set()
                for s in struct_bounds:
                    for offset in range(-d, d + 1):
                        pos = s + offset
                        if 0 <= pos < L:
                            covered.add(pos)
                p_rand[d] = len(covered) / L
            
            # 3. Tokenize and get BPE boundaries
            enc = tokenizer(seq, return_offsets_mapping=True, add_special_tokens=False)
            bpe_bounds = []
            for start, end in enc.offset_mapping:
                if end < L:
                    bpe_bounds.append(end)
                    
            n_bpe = len(bpe_bounds)
            if n_bpe == 0:
                continue
                
            bpe_total += n_bpe
            
            # Accumulate expected random hits (weighted by N_bpe in this chain)
            rand_expected_hits += n_bpe * p_rand
            
            # 4. Measure actual BPE distances
            for b in bpe_bounds:
                dist = np.min(np.abs(struct_bounds_arr - b))
                for d in dist_bins:
                    if dist <= d:
                        bpe_hits[d] += 1
        
    # Compute rates
    rate_bpe = bpe_hits / bpe_total if bpe_total > 0 else np.zeros(max_dist + 1)
    rate_rand = rand_expected_hits / bpe_total if bpe_total > 0 else np.zeros(max_dist + 1)
    lift = np.where(rate_rand > 0, rate_bpe / rate_rand, np.nan)
    
    # --- Print Report ---
    print("\n" + "="*60)
    print("BOUNDARY ALIGNMENT LIFT REPORT (Exact Analytical Baseline)")
    print("="*60)
    print(f"{'Max Dist (AA)':<15} | {'BPE Rate':<10} | {'Random Rate':<12} | {'Lift':<8}")
    print("-" * 60)
    for i, d in enumerate(dist_bins):
        print(f"<= {d:<12} | {rate_bpe[i]:<10.3f} | {rate_rand[i]:<12.3f} | {lift[i]:<8.2f}")
    print("="*60 + "\n")
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(dist_bins, rate_rand, 'o--', color='gray', label='Random Baseline (Exact)', linewidth=2)
    ax.plot(dist_bins, rate_bpe, 's-', color='#2ecc71', label='ProXYZ BPE', linewidth=2, markersize=8)
    
    # Annotate Lift on the plot
    for i, d in enumerate(dist_bins):
        if not np.isnan(lift[i]):
            ax.annotate(f'Lift={lift[i]:.2f}', 
                        (d, rate_bpe[i]), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=10, color='blue', weight='bold')
            
    ax.set_xlabel("Distance to Nearest Structure Boundary (AA)", fontsize=13)
    ax.set_ylabel("Cumulative Fraction of Boundaries", fontsize=13)
    ax.set_title("BPE Boundary Alignment vs Exact Random Baseline", fontsize=15)
    ax.set_xticks(dist_bins)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("part2_boundary_alignment_lift.png", dpi=150)
    plt.show()


@click.command(context_settings={"show_default": True})
@click.argument("data_files", type=click.Path(), nargs=-1)
@click.option(
    "--data_list_file",
    type=click.Path(),
    default=None,
    help="A text file that lists data_files line-by-line.",
)
@click.option(
    "--tokenizer_file",
    type=click.Path(),
    default="my_tokenizer.json",
    help="Path to the tokenizer json file.",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    # FIX: seed both RNGs at the top of compute_lift_with_significance
    # so that baseline sampling (and therefore the stars) is reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.data_list_file is not None:
        with opener(args.data_list_file, "r") as f:
            data_files = tuple(line for line in lines(f))
        args.data_files = args.data_files + data_files

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_file,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    print("Running Part 0: Length-Stratified SSE Lift...")
    compute_lift_with_significance(
        tokenizer, args.data_files, smoothing_alpha=1.0, min_samples=50, alpha_level=0.05
    )

    print("Running Part 1: All-Helix Periodicity...")
    analyze_all_helix_periodicity(tokenizer, args.data_files)
    
    print("Running Part 2: Boundary Alignment Lift...")
    compute_boundary_alignment_lift(tokenizer, args.data_files)

# === Entry Point ===
if __name__ == "__main__":
    main()
