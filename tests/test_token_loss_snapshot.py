"""Smoke test: outputs["token_loss"] is a snapshot, not an alias of outputs["loss"].

Trainer.compute_loss scales `loss` in-place (loss *= loss_scale); without aux
heads the old `token_loss = loss` aliasing let that scaling leak into the
token_loss metric (2x inflation on 2 ranks). Run with:

    PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
        python tests/test_token_loss_snapshot.py
"""

import atexit
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

from proxyz import models

TINY = dict(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=64,
    attn_implementation="eager",
)
CHAR_TINY = dict(
    char_hidden_size=16,
    char_intermediate_size=32,
    char_num_hidden_layers=1,
    char_num_attention_heads=2,
    char_num_key_value_heads=2,
)


def make_inputs(vocab_size=32, batch=2, tok_len=8, char_len=12):
    input_ids = torch.randint(0, vocab_size, (batch, tok_len))
    labels = input_ids.clone()
    char_input_ids = torch.randint(0, vocab_size, (batch, char_len))
    repr_char_idx = torch.arange(tok_len).expand(batch, tok_len)
    cle_labels = torch.randint(0, 26, (batch, char_len))
    cle_labels[:, -1] = -100
    return input_ids, labels, char_input_ids, repr_char_idx, cle_labels


def check_no_aux(mod):
    model = mod.XYZForCausalLM(mod.XYZConfig(**TINY)).eval()
    input_ids, labels, *_ = make_inputs()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert outputs.loss is not None and outputs.token_loss is not None
    assert outputs.token_loss is not outputs.loss, "token_loss aliases loss"
    assert outputs.token_loss.data_ptr() != outputs.loss.data_ptr()
    assert torch.equal(outputs.token_loss, outputs.loss)
    snapshot = outputs.token_loss.clone()
    outputs.loss *= 2  # simulate Trainer.compute_loss's in-place scaling
    assert torch.equal(outputs.token_loss, snapshot), "in-place scale leaked"
    return outputs


def check_no_labels(mod):
    model = mod.XYZForCausalLM(mod.XYZConfig(**TINY)).eval()
    input_ids, *_ = make_inputs()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    assert outputs.loss is None and outputs.token_loss is None


def check_with_aux(mod):
    config = mod.XYZConfig(**TINY, **CHAR_TINY, has_cle_lm_head=True)
    model = mod.XYZForCausalLM(config).eval()
    input_ids, labels, char_input_ids, repr_char_idx, cle_labels = make_inputs()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            char_input_ids=char_input_ids,
            repr_char_idx=repr_char_idx,
            cle_labels=cle_labels,
            use_cache=False,
        )
    assert outputs.cle_loss is not None
    assert outputs.token_loss is not outputs.loss
    assert torch.allclose(outputs.token_loss + outputs.cle_loss, outputs.loss)
    token_snapshot = outputs.token_loss.clone()
    cle_snapshot = outputs.cle_loss.clone()
    outputs.loss *= 2
    assert torch.equal(outputs.token_loss, token_snapshot)
    assert torch.equal(outputs.cle_loss, cle_snapshot)


def load_old_module():
    old_src = subprocess.check_output(
        ["git", "show", "HEAD:src/proxyz/models/modeling_xyz.py"], text=True
    )
    # Standalone load: resolve relative imports against the current package
    # (configuration/processing are untouched by this change).
    old_src = old_src.replace(
        "from .configuration_xyz import", "from proxyz.models.configuration_xyz import"
    ).replace(
        "from .processing_xyz import", "from proxyz.models.processing_xyz import"
    )
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(old_src)
    spec = importlib.util.spec_from_file_location("old_modeling_xyz", path)
    mod = importlib.util.module_from_spec(spec)
    # Register so inspect.getsourcefile works (@auto_docstring needs it); keep
    # the file alive for the same reason and clean up at exit.
    sys.modules[spec.name] = mod
    atexit.register(os.unlink, path)
    spec.loader.exec_module(mod)
    return mod


def main():
    torch.manual_seed(0)

    check_no_aux(models)
    print("PASS: no-aux token_loss is an independent snapshot")
    check_no_labels(models)
    print("PASS: no-labels path yields loss=None, token_loss=None")
    check_with_aux(models)
    print("PASS: aux path keeps token_loss + cle_loss == loss, scale-safe")

    # Before/after: the old (HEAD) model aliases, the new one does not.
    old = load_old_module()
    model = old.XYZForCausalLM(old.XYZConfig(**TINY)).eval()
    input_ids, labels, *_ = make_inputs()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert outputs.token_loss is outputs.loss, "expected old aliasing"
    before = outputs.token_loss.item()
    outputs.loss *= 2
    assert outputs.token_loss.item() == before * 2, "expected old leak"
    print("PASS: confirmed old code aliases and leaks the in-place scale")

    print("All token_loss snapshot checks passed.")


if __name__ == "__main__":
    main()