"""UNGATED forward-with-ablation for the punctuation causal study.

Loads the model via the gated `capture_core.load_model` + `Adapter` (read-only use,
like `elicit.py` — this file NEVER modifies capture_core.py and lives entirely outside
the gate hash), then applies two intervention families through its OWN mutation hooks:

  - ablate_write(layers, positions, mode): zero the WRITE at (layer, position) so
    resid_post == resid_pre there. mode='both' (attn+mlp, the layer's total write),
    'attn' (attn_out only), or 'mlp' (mlp_out only). Tests LOCAL computation.
  - knockout_attn(layers, key_positions): mask attention INTO the given key positions
    (additive -inf on those key columns) so no query can read their K/V. Tests
    ROUTE-THROUGH / global-anchor.

Uncertified by design: mutating the forward pass breaks Claim B (incremental ≡ static),
so this is NOT gated and any claim from it is Rule-9'd from scratch. Deterministic
(greedy, seed-free single forward). Answer readout = logits at the generation position
over the {True, False, Unknown} label tokens.
"""

import hashlib
import json

import torch
from contextlib import contextmanager

from gentrace import capture_core   # read-only: load_model + Adapter

LABELS = ("True", "False", "Unknown")


# ---------- content-addressed patch spec (WO-22 Stage 2) --------------------
def canonical_patch_spec(baseline, donor, layer, positions, stream, scope):
    """Build a JSON-canonical, content-addressable residual-patch spec.

    `baseline` / `donor` are descriptor objects (e.g.
    {"tid": ..., "slot": ...} for a stored-trace source, or
    {"vectors_sha256": ...} for an explicit vectors file). The returned dict is
    deterministic (sorted positions); `patch_spec_id()` hashes it so a DIFFERENT
    spec yields a DIFFERENT output identity — the injection system never
    silently overwrites one patch's output with another's."""
    return {
        "baseline": baseline,
        "donor": donor,
        "layer": int(layer),
        "positions": sorted(int(p) for p in positions),
        "stream": stream,
        "scope": scope,
    }


def patch_spec_id(spec):
    """16-hex content id of a canonical_patch_spec() dict."""
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


class Intervenor:
    def __init__(self, model_id, dtype="float16", device="cuda"):
        self.model, self.tok, _ = capture_core.load_model(model_id, dtype, device)
        self.A = capture_core.Adapter(self.model)
        self.device = device

    # ---------- interventions ----------

    @contextmanager
    def ablate_write(self, layers, positions, mode="both"):
        pos = list(positions)
        handles = []
        for L in layers:
            if mode == "both":
                mods = [self.A.layers[L]]
                fn = self._layer_zero(pos)
            elif mode == "attn":
                mods = [self.A.attn[L]]
                fn = self._out_zero(pos)
            elif mode == "mlp":
                mods = [self.A.mlp[L]]
                fn = self._out_zero(pos)
            else:
                raise ValueError(mode)
            for m in mods:
                handles.append(m.register_forward_hook(fn, with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _layer_zero(self, pos):
        # decoder layer: out is a tuple, out[0] = resid_post; args[0] = resid_pre.
        # Setting resid_post[pos] = resid_pre[pos] zeros the layer's total write there.
        def hook(mod, args, kwargs, out):
            hs_in = args[0] if args else kwargs["hidden_states"]
            t = out[0] if isinstance(out, (tuple, list)) else out
            t[:, pos, :] = hs_in[:, pos, :]
            return out
        return hook

    def _out_zero(self, pos):
        # attn/mlp module: zero its output at pos → removes that component's write.
        def hook(mod, args, kwargs, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            t[:, pos, :] = 0
            return out
        return hook

    @contextmanager
    def knockout_attn(self, layers, key_positions):
        keys = list(key_positions)
        handles = []
        for L in layers:
            handles.append(self.A.attn[L].register_forward_pre_hook(
                self._mask_keys(keys), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _mask_keys(self, keys):
        # additive attention_mask is [B,1,q,kv]; -inf on key columns blocks reading them.
        # PREREG §4.4: a missing/non-float mask must RAISE — silently skipping would turn
        # every knockout into a baseline forward (the banned silent-failure mode). The
        # eager attention pinned by capture_core.load_model materializes the float mask.
        def pre(mod, args, kwargs):
            am = kwargs.get("attention_mask")
            if am is None or not torch.is_floating_point(am):
                raise RuntimeError(
                    "knockout_attn needs a materialized float attention mask "
                    f"(eager attention); got {type(am).__name__} — refusing to "
                    "run as a silent no-op")
            am = am.clone()
            am[..., keys] = torch.finfo(am.dtype).min
            kwargs["attention_mask"] = am
            return args, kwargs
        return pre

    @contextmanager
    def ablate_kv(self, layers, positions, which="value", source="zero", ids=None):
        """Key-only / value-only corruption at target positions, leaving the OTHER
        projection bitwise intact (GQA-mailbox prereg §A3 warm-up — discriminates
        WA-both's damage pathway). `which='value'` corrupts V (K untouched → the
        content/value pathway); `which='key'` corrupts K (V untouched → the
        addressing/routing pathway).

        `source='zero'`: set the projection-output slice at the position to 0 — a
        crude drain (least-anomalous value; cannot separate injection from no-op).
        `source='wa'`: set the projection INPUT at the position to the write-ablated
        (embedding-only) residual that layer's projection would see — the EXACT
        counterfactual matching WA-both's value/key. Needs `ids=` (one WA-both
        forward is run to cache the per-layer inputs). This is the cell that
        discriminates injection (V←wa catastrophic) from re-routing (K←wa
        catastrophic); see PREREG_gqa_mailbox §A3.

        Qwen/Llama (separate k_proj/v_proj) and GPT-2 (fused c_attn) both supported."""
        if which not in ("key", "value"):
            raise ValueError(which)
        if source not in ("zero", "wa"):
            raise ValueError(source)
        pos = list(positions)
        if source == "wa" and ids is None:
            raise ValueError("ablate_kv(source='wa') requires ids= for the WA capture")
        wa = self._wa_attn_inputs(ids, pos, layers) if source == "wa" else None
        handles = []
        for L in layers:
            attn = self.A.attn[L]
            if hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                proj = attn.k_proj if which == "key" else attn.v_proj
                if source == "zero":
                    handles.append(proj.register_forward_hook(
                        self._out_zero(pos), with_kwargs=True))
                else:
                    handles.append(proj.register_forward_pre_hook(
                        self._sub_input(pos, wa[L]), with_kwargs=True))
            elif hasattr(attn, "c_attn"):
                if source == "zero":
                    handles.append(attn.c_attn.register_forward_hook(
                        self._slice_zero(pos, which), with_kwargs=True))
                else:
                    handles.append(attn.c_attn.register_forward_hook(
                        self._slice_wa(pos, which, wa[L], attn.c_attn), with_kwargs=True))
            else:
                raise RuntimeError("ablate_kv: unrecognized attention module")
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _wa_attn_inputs(self, ids, pos, layers):
        """Cache each requested layer's attention-module input at `pos` under a
        FULL-DEPTH WA-both forward. The write-ablated residual reaching layer L must
        have writes zeroed at ALL layers 0..L-1 (so resid_L[P] == embedding[P]);
        ablating only the requested layers would leave each layer's INPUT clean (the
        write happens after attention reads) and the substitution a no-op. So the
        ablation spans range(n_layers) regardless of which layers we substitute at —
        this is the embedding-only counterfactual that matches full WA-both."""
        cache = {}
        def mk(L):
            def pre(mod, args, kwargs):
                hs = args[0] if args else kwargs.get("hidden_states")
                cache[L] = hs[:, pos, :].detach().clone()
            return pre
        handles = [self.A.attn[L].register_forward_pre_hook(mk(L), with_kwargs=True)
                   for L in layers]
        try:
            with self.ablate_write(range(self.A.n_layers), pos, "both"), torch.no_grad():
                self.model(ids)
        finally:
            for h in handles:
                h.remove()
        return cache

    def _sub_input(self, pos, wa_row):
        # replace a Linear's INPUT rows at pos (K/V projection sees the WA residual)
        def pre(mod, args, kwargs):
            hs = args[0].clone()
            hs[:, pos, :] = wa_row.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(args[1:]), kwargs
        return pre

    def _slice_zero(self, pos, which):
        # GPT-2 fused c_attn output is [B, T, 3*d] as q|k|v — zero one slice only.
        def hook(mod, args, kwargs, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            d = t.shape[-1] // 3
            sl = slice(d, 2 * d) if which == "key" else slice(2 * d, 3 * d)
            t[:, pos, sl] = 0
            return out
        return hook

    def _slice_wa(self, pos, which, wa_row, c_attn):
        # GPT-2: recompute one c_attn slice from the WA input (K/V from embedding resid)
        def hook(mod, args, kwargs, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            d = t.shape[-1] // 3
            sl = slice(d, 2 * d) if which == "key" else slice(2 * d, 3 * d)
            W = c_attn.weight[:, sl]
            b = c_attn.bias[sl]
            t[:, pos, sl] = (wa_row.to(W.dtype).to(W.device) @ W + b).to(t.dtype)
            return out
        return hook

    # ---------- per-(layer, head) interventions (WO-10; ungated) ----------
    #
    # Surgical single-head edits at the o_proj INPUT: head h's slice of the
    # concatenated per-head context vector (columns h*dh:(h+1)*dh). o_proj is
    # bias-free on Qwen3 (repo fact; additive identities exact), so zeroing
    # the slice removes EXACTLY head h's contribution to attn_out, and the
    # hook is a no-op for every other head BY CONSTRUCTION (their columns are
    # never touched; layers other than L have no hook installed).

    def _head_dim(self):
        cfg = self.model.config
        return getattr(cfg, "head_dim", None) or \
            cfg.hidden_size // cfg.num_attention_heads

    def _head_slice(self, head):
        dh = self._head_dim()
        return slice(head * dh, (head + 1) * dh)

    @contextmanager
    def ko_head(self, layer, head, positions=None):
        """KO(L,h): zero head h's o_proj input slice at `layer`. positions=None
        = ALL positions (v1 scope, PREREG_l32h8_causal §3)."""
        sl = self._head_slice(head)
        pos = slice(None) if positions is None else list(positions)

        def pre(mod, args, kwargs):
            hs = (args[0] if args else kwargs["input"]).clone()
            hs[:, pos, sl] = 0
            return (hs,) + tuple(args[1:]), kwargs

        h = self.A.oproj[layer].register_forward_pre_hook(
            pre, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    @contextmanager
    def wa_head(self, layer, head, ids, positions=None):
        """WA(L,h): replace head h's o_proj input slice at `layer` with the
        slice the SAME head computes under the WA-both donor convention
        (PREREG_punct_mechanism §5.2 / ablate_kv(source='wa') pattern): a
        full-depth write-ablated donor forward (resid == embedding + RoPE at
        the target positions), during which head h's o_proj input slice at
        `layer` is cached. Least-anomalous non-informative replacement — the
        head still writes 'something a parked/uninformed head would', rather
        than a hard zero. positions=None = ALL positions (v1 scope); the
        intervention forward must use the SAME ids as the donor forward."""
        sl = self._head_slice(head)
        pos = slice(None) if positions is None else list(positions)
        donor = {}

        def cache_pre(mod, args, kwargs):
            hs = args[0] if args else kwargs["input"]
            donor["row"] = hs[:, pos, sl].detach().clone()
            return None

        cache_h = self.A.oproj[layer].register_forward_pre_hook(
            cache_pre, with_kwargs=True)
        try:
            all_pos = list(range(ids.shape[1])) if positions is None \
                else list(positions)
            with self.ablate_write(range(self.A.n_layers), all_pos, "both"), \
                    torch.no_grad():
                self.model(ids)
        finally:
            cache_h.remove()

        def sub_pre(mod, args, kwargs):
            hs = (args[0] if args else kwargs["input"]).clone()
            hs[:, pos, sl] = donor["row"].to(hs.dtype).to(hs.device)
            return (hs,) + tuple(args[1:]), kwargs

        h = self.A.oproj[layer].register_forward_pre_hook(
            sub_pre, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    @contextmanager
    def sham_head(self, layer, head):
        """Sham arm: the hook is installed on the same module and exercises
        the same code path, but returns the inputs unchanged (clone-and-
        return, no mutation). Baseline for the paired KO/WA contrasts —
        must be bitwise identical to a hook-free forward (verified by
        verify_intervene_perhead.py)."""
        _ = self._head_slice(head)          # same validation path as KO/WA

        def pre(mod, args, kwargs):
            hs = (args[0] if args else kwargs["input"]).clone()
            return (hs,) + tuple(args[1:]), kwargs

        h = self.A.oproj[layer].register_forward_pre_hook(
            pre, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    # ---------- interchange patches (WO-13; ungated) ----------
    #
    # Donor vectors come from ANOTHER certified trace's STORED arrays
    # (family-paired t1<->t2 interchange; donor at its own slot position).
    # The caller loads the donor row from disk; these hooks only substitute.

    @contextmanager
    def patch_head(self, layer, head, pos, donor_row):
        """Substitute head h's o_proj input slice at (layer, pos) with
        `donor_row` (tensor [dh]) — the paired trace's stored
        oproj_in[layer, donor_slot, h*dh:(h+1)*dh]."""
        sl = self._head_slice(head)
        v = torch.as_tensor(donor_row).reshape(-1)

        def pre(mod, args, kwargs):
            hs = (args[0] if args else kwargs["input"]).clone()
            hs[:, pos, sl] = v.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(args[1:]), kwargs

        h = self.A.oproj[layer].register_forward_pre_hook(
            pre, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    @contextmanager
    def patch_attn_slot(self, layer, pos, donor_row):
        """Substitute the WHOLE attn_out at (layer, pos) with `donor_row`
        (tensor [d]) — the paired trace's stored attn_out[layer, donor_slot]."""
        v = torch.as_tensor(donor_row).reshape(-1)

        def hook(mod, args, kwargs, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            t[:, pos, :] = v.to(t.dtype).to(t.device)
            return out

        h = self.A.attn[layer].register_forward_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    @contextmanager
    def patch_resid_slot(self, layer, pos, donor_row):
        """Substitute the residual state ENTERING `layer` at `pos` with
        `donor_row` (tensor [d]) — the paired trace's stored
        resid_pre[layer, donor_slot]. Implemented as a pre-hook on the
        decoder layer (its input IS resid_pre at that layer)."""
        v = torch.as_tensor(donor_row).reshape(-1)

        def pre(mod, args, kwargs):
            if args:
                hs = args[0].clone()
                hs[:, pos, :] = v.to(hs.dtype).to(hs.device)
                return (hs,) + tuple(args[1:]), kwargs
            hs = kwargs["hidden_states"].clone()
            hs[:, pos, :] = v.to(hs.dtype).to(hs.device)
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hs
            return args, kwargs

        h = self.A.layers[layer].register_forward_pre_hook(
            pre, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    # ---------- resid-band donor patching (WO-22 Stage 2; ungated) ----------
    #
    # Generalizes patch_resid_slot to an arbitrary POSITION SET at a layer, on
    # either the resid_pre[l] or resid_post[l] stream, prefill- or through-decode
    # scoped. Donor vectors are supplied by the caller (from a donor trace's
    # stored arrays, or an explicit vectors file). The o_proj/attn/Adapter are
    # consumed read-only; nothing here touches gated code. Mechanical exactness
    # (null-patch identity, per-cell = donor, causal-mask honesty, stored-logits
    # anchor) is certified by studies/verify_patch_resid_band.py on cpu/fp32 AND
    # the A100 cuda/fp32 profile.

    @contextmanager
    def patch_resid_band(self, layer, positions, donor, stream="resid_pre",
                         scope="prefill"):
        """Overwrite the residual stream at (layer, positions) with `donor`
        before the next consumer reads it.

          stream='resid_pre'  : the state ENTERING `layer` (pre-hook on decoder
                                layer[layer]); == resid_pre[layer].
          stream='resid_post' : the state LEAVING `layer` (forward-hook on
                                layer[layer], out[0]); == resid_post[layer] ==
                                resid_pre[layer+1].

        `positions` = token indices P (set/list). `donor` is a tensor/array
        [len(P), d] (row i -> the sorted-unique positions[i]) or [d] (broadcast
        to every p in P). `scope='prefill'` writes only on a multi-token forward
        (seq_len>1); the patched state then rides the KV cache through decode.
        `scope='through_decode'` (alias 'all') writes on every forward for any p
        in range. Only rows at P are written, so the causal mask is respected by
        construction (upstream positions stay bit-exact — verified)."""
        if stream not in ("resid_pre", "resid_post"):
            raise ValueError(stream)
        if scope not in ("prefill", "through_decode", "all"):
            raise ValueError(scope)
        pos = sorted({int(p) for p in positions})
        D = torch.as_tensor(donor)
        if D.ndim == 1:
            D = D.unsqueeze(0).expand(len(pos), -1)
        if D.shape[0] != len(pos):
            raise ValueError(
                f"donor has {int(D.shape[0])} rows for {len(pos)} positions")

        def _apply(t):
            seq = t.shape[1]
            if scope == "prefill" and seq <= 1:
                return
            for i, p in enumerate(pos):
                if 0 <= p < seq:
                    t[:, p, :] = D[i].to(t.dtype).to(t.device)

        if stream == "resid_pre":
            def pre(mod, args, kwargs):
                if args:
                    hs = args[0].clone()
                    _apply(hs)
                    return (hs,) + tuple(args[1:]), kwargs
                hs = kwargs["hidden_states"].clone()
                _apply(hs)
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hs
                return args, kwargs
            h = self.A.layers[layer].register_forward_pre_hook(
                pre, with_kwargs=True)
        else:
            def hook(mod, args, kwargs, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                _apply(t)
                return out
            h = self.A.layers[layer].register_forward_hook(
                hook, with_kwargs=True)
        try:
            yield
        finally:
            h.remove()

    @contextmanager
    def steer(self, layers, positions, direction, alpha=1.0):
        """Per-layer directional steering: add alpha*direction to resid_post at
        (layer, positions). `direction` is a [hidden] tensor. Positive causal test
        (inject a direction, watch behaviour move); also an injection route-through
        probe (add at a punctuation position, see if downstream reads it)."""
        pos = list(positions)
        v = (alpha * direction.to(self.device)).to(next(self.model.parameters()).dtype)
        handles = []
        for L in layers:
            handles.append(self.A.layers[L].register_forward_hook(
                self._add_dir(pos, v), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _add_dir(self, pos, v):
        def hook(mod, args, kwargs, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            t[:, pos, :] = t[:, pos, :] + v
            return out
        return hook

    @contextmanager
    def project_out(self, layers, positions, direction):
        """Directional (surgical) ablation: remove the component along `direction`
        from resid_post at (layer, positions), leaving the rest of the write intact.
        Isolates a specific direction (e.g. a content axis, or the massive-activation
        outlier) instead of zeroing everything — answers the 'zeroing conflates
        everything' objection."""
        pos = list(positions)
        d = direction.to(self.device).float()
        dhat = d / (d.norm() + 1e-8)
        handles = []
        for L in layers:
            handles.append(self.A.layers[L].register_forward_hook(
                self._proj_out(pos, dhat), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _proj_out(self, pos, dhat):
        def hook(mod, args, kwargs, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            sub = t[:, pos, :].float()
            coef = sub @ dhat                       # [b, npos]
            sub = sub - coef.unsqueeze(-1) * dhat
            t[:, pos, :] = sub.to(t.dtype)
            return out
        return hook

    # ---------- encoding / readout ----------

    def encode(self, prompt):
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    def tokens(self, enc):
        return self.tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    # NOTE (PREREG §4.5): the old answer_logits/_label_ids helpers were removed —
    # they tokenized labels WITHOUT the leading space and were dead code in the
    # pipeline (each script builds its own " "+label ids); keeping them invited a
    # silent wrong-token readout.
