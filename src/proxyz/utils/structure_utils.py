import numpy as np
import torch


def contact_precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: torch.Tensor | None = None,
    minsep: int = 6,
    maxsep: int | None = None,
    override_length: int | None = None,  # for casp
):
    """Computes contact precisions.

    For protein contact prediction, precision is measured for the top (L/K) highest confidence predictions,
    with L being the length of the protein sequence and K generally being equal to 1 or 5.

    K = 5 measures the predictions of the very highest confidence contacts, while K = 1 is a more general measure
    over all relatively high confidence predictions.

    Since there are roughly ~L true contacts in a protein, this is a reasonable cutoff.


    Args:
        predictions (torch.Tensor): Tensor of probabilities of size (B, L, L)
        targets (torch.Tensor): Tensor of true contacts of size (B, L, L)
        src_lengths (torch.Tensor, optional): Lengths of each sample in the batch, if using variable lengths.
            If not provided, inferred from the size of the predictions.
        minsep (int): Minimum separation distance to consider. We often want to measure contacts at a
            certain range. Typical ranges are short [6, 12), medium [12, 24), and long [24, inf).
        maxsep (int, optional): Used in conjunction with minsep to specify a contact range. If not provided uses
            assumes no maximum range
        override_length (int, optional): Used for casp evaluation where sometimes the "true" length is not
            the same as the length of the input. Kept for posterity, we probably don't need this argument.
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()

    # Step 1) Construct a mask of size [B, L, L] to mask invalid contacts
    seqlen_range = torch.arange(seqlen, device=device)
    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    # Mask contacts that are closer than minsep
    valid_mask = sep >= minsep
    # Mask contacts where target is negative (padding or unknown)
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    # Mask contacts that are farther than maxsep, if provided
    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        # If the lengths of the individual sequences are provided, mask positions
        # that are farther than the end of the sequence.
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    # Fill in the logit tensor with -inf for all invalid positions
    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    # Step 2) Select the top half of the prediction (should be symmetric)
    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    # Step 3) Select the topk values in each batch where k = L (length of sequence)
    topk = seqlen if override_length is None else max(seqlen, override_length)
    # Indices are the indices into the predictions corresponding to the most confident predictions
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    # topk_targets are the target values corresponding to the above indices
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        # If there aren't enough targets, pad to the output.
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])
    # FIX: ignore_index = -100
    topk_targets = topk_targets.where(topk_targets >= 0, 0)

    # Step 4) Sum the accuracy at of the top-i predictions for i in 1, L
    # topk_targets => 1/0 true vs. false contact, sorted by confidence of prediction
    # cmumulative sum => Number of correct answers for the top-i predictions.
    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    # Step 5) Find the gather indices. This should be P@(L / K) for varous values of K
    # The values will differ for each batch.
    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    # This gets you (0.1 * L, 0.2 * L, 0.3 * L, etc.)
    gather_indices = (
        (torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths).type(
            torch.long
        )
        - 1
    ).clamp_min(0)

    # Step 6) Gather the results and divide by the number of guesses to get the precision.
    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    # Select specific P@L/k. pl5 is index 1 b/c that corresponds to L * 0.2 in
    # gather_indices above
    pl5 = binned_precisions[:, 1]
    # pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    # AUC is the integral wrt K of P@L/K for K in range(1, L)
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L5": pl5}
