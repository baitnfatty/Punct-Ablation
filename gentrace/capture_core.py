"""GenTrace gated capture core.

THIS FILE IS COVERED BY THE GATE. The capture machinery validated by
selfcheck.py is the machinery that produces real traces — capture.py imports
from here and adds only CLI plumbing and persistence. The gate's code hash is
sha256 over (this file + selfcheck.py); editing either invalidates every gate
and forces a re-run of the battery. See DERIVATION.md §7-8, PIPELINE_DESIGN.md §2.
"""

import hashlib
import json
import os

import torch

EPS = 1e-30
EPS_BY_DTYPE = {"float32": 2.0 ** -23, "float16": 2.0 ** -10}
TOL_BY_DTYPE = {
    # gates apply to max_rel (see metric()); dtype-scoped, never mixed
    # fp32 G3/G5/G7 are STACK-SCOPED (SPEC_perstack_rails.md, 2026-07-09): they
    # are measured rounding floors, not portable constants (REDTEAM_calibration.md
    # — the old globals G3 2e-5 / G5 1e-5 / G7 1e-4 were ROCm/local-era values,
    # violated on other stacks by clean models). They are sourced at gate time
    # from the APPROVED per-stack floors record (check_stackfloors); None here so
    # any consumer that skips that sourcing fails loudly. G1/G4 stay global: they
    # are exact-association identities (measure 0.0 bitwise on every stack).
    "float32": {"G1_additivity": 1e-6, "G3_per_head": None, "G4_telescope": 1e-6,
                "G5_logits_ident": None, "G7_incr_vs_static": None},
    # float16 G4/G7 calibrated against the measured hardware rounding floor
    # (Qwen3-1.7B, gfx1030, 2026-07: G4 1.24e-2, G7 6.86e-2; a-priori guesses
    # 5e-3 / 5e-2 sat below the floor). 2x measured headroom, still 1-2 orders
    # below real-bug signatures (O(1), layer-0 onset). HARDWARE-SCOPED: valid
    # only for the GPU they were measured on — supplement gates record
    # gpu_name and check_supplement() refuses a different GPU. See
    # DERIVATION.md §8.
    "float16": {"G1_additivity": 5e-3, "G3_per_head": 5e-3, "G4_telescope": 2.5e-2,
                "G5_logits_ident": 5e-3, "G7_incr_vs_static": 1e-1},
}
STREAMS = ("resid_pre", "attn_out", "mlp_out", "resid_post")

# Fixed diverse calibration suite for --calibrate. Chosen so the measured drift
# ceiling reflects prompt-content variance the drift study revealed (prose, QA,
# code/whitespace — the worst driver of the massive-activation channel — and
# formal text). Changing this set changes the gated code hash. Threads=1,
# greedy, fixed length; the measured max over the suite is the calibration
# floor and proposed thresholds scale from it. See studies/ENVELOPE_PROPOSAL.md.
CALIBRATION_PROMPTS = [
    "The history of measurement begins with the human body: the cubit, the "
    "span, and the foot were among the first rulers, and every later standard "
    "had to negotiate with them.",
    "Q: Why is the sky blue during the day but red at sunset? A:",
    "def quicksort(arr):",
    "ITEM 1A. RISK FACTORS. Investing in our common stock involves a high "
    "degree of risk.",
]
CALIBRATION_MAX_NEW_TOKENS = 64

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATED_FILES = (os.path.join(REPO_ROOT, "gentrace", "capture_core.py"),
               os.path.join(REPO_ROOT, "selfcheck.py"))


class RefusalError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# hashing / determinism / loading

def sha256_bytes(*chunks):
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def gated_code_sha256():
    """The gate-covered code hash: name-tagged concatenation of the gated files."""
    h = hashlib.sha256()
    for p in GATED_FILES:
        h.update(os.path.basename(p).encode() + b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return h.hexdigest()


def t_bytes(t):
    return t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()


def setup_determinism(seed, threads):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def resolve_eos_ids(model, tok):
    """EOS token ids for optional early-stop: union of model.config.eos_token_id
    (int or list) and the tokenizer's eos_token_id."""
    ids = set()
    cfg_eos = getattr(model.config, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        ids.add(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple)):
        ids.update(int(x) for x in cfg_eos)
    if getattr(tok, "eos_token_id", None) is not None:
        ids.add(int(tok.eos_token_id))
    return sorted(ids)


def load_model(model_id, dtype_str, device):
    """Shared loader: fp32/fp16, eager attention, eval mode. Returns
    (model, tokenizer, revision)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype_str]
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, attn_implementation="eager")
    model.to(device).eval()
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    return model, tok, revision


# ----------------------------------------------------------------------------
# metrics

def metric(x, ref):
    """Three views of |x-ref|:
    max_abs — raw; norm — max_abs / RMS(ref); max_rel — elementwise relative
    with an RMS floor: max_i |x_i-ref_i| / max(|ref_i|, RMS(ref)). Residual
    streams carry large-magnitude outlier coordinates, where 1-ULP rounding
    is ~RMS/1e4 in absolute terms; max_rel is the scale-honest metric, the
    others stay reported. element_audit() bounds what the RMS floor could
    hide. ref_abs_at_worst pins the magnitude at the worst-abs element."""
    if not x.numel():
        return {"max_abs": 0.0, "ref_rms": 0.0, "norm": 0.0,
                "max_rel": 0.0, "ref_abs_at_worst": 0.0}
    diff = (x - ref).abs()
    d = diff.max().item()
    ref_at = ref.flatten()[torch.argmax(diff)].abs().item()
    rms = ref.float().pow(2).mean().sqrt().item()
    rel = (diff / ref.abs().clamp_min(rms + EPS)).max().item()
    return {"max_abs": d, "ref_rms": rms, "norm": d / (rms + EPS),
            "max_rel": rel, "ref_abs_at_worst": ref_at}


def element_audit(x, ref, dims, eps):
    """Worst element by ABSOLUTE diff, worst by UNFLOORED RELATIVE diff
    (|Δ|/|ref|, no floor), and the floor-masking bound: the largest |Δ| among
    elements where the RMS floor is active (|ref| < RMS), in units of
    eps·RMS. Roundoff verdict requires agreement of both views: worst-by-abs
    is ULP-scale at its own magnitude AND floored elements carry only
    eps·RMS-scale absolute error. `eps` is the machine epsilon of the dtype
    the computation ran in (captures are upcast, arithmetic was not)."""
    diff = (x - ref).abs()
    rms = ref.float().pow(2).mean().sqrt().item()
    fd, fr, fx = diff.flatten(), ref.flatten(), x.flatten()

    def elem(i):
        idx, rem = [], i
        for s in reversed(x.shape):
            idx.append(int(rem % s))
            rem //= s
        d, r = fd[i].item(), fr[i].item()
        return {"loc": dict(zip(dims, reversed(idx))),
                "abs_diff": d, "ref": r, "test": fx[i].item(),
                "rel_unfloored": d / (abs(r) + EPS),
                "diff_in_ulp_at_ref": d / (eps * max(abs(r), EPS))}

    ia = int(torch.argmax(fd).item())
    ir = int(torch.argmax(fd / (fr.abs() + EPS)).item())
    floored = torch.where(fr.abs() < rms, fd, torch.zeros_like(fd))
    im = int(torch.argmax(floored).item())
    return {
        "ref_rms": rms, "ulp_unit_eps": eps,
        "worst_by_abs": elem(ia),
        "worst_by_rel_unfloored": elem(ir),
        "floor_masking_bound": {
            "max_abs_diff_among_floored_elements": floored[im].item(),
            "in_units_of_eps_x_rms": floored[im].item() / (eps * rms + EPS),
            "element": elem(im),
        },
    }


def grab(t):
    return t.detach().to("cpu", torch.float32).clone()


# ----------------------------------------------------------------------------
# architecture adapter

class Adapter:
    """Locates capture points. Families: llama-like (Llama/Qwen/Mistral...),
    gpt_neox (Pythia), gpt2. Anything else is refused (gate refusal)."""

    def __init__(self, model):
        cfg = model.config
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            core = model.model
            self.family = "llama-like"
            self.layers = list(core.layers)
            self.attn = [l.self_attn for l in self.layers]
            self.mlp = [l.mlp for l in self.layers]
            self.oproj = [l.self_attn.o_proj for l in self.layers]
            self.final_norm = core.norm
            self.lm_head = model.lm_head
            self.parallel_residual = False
        elif hasattr(model, "gpt_neox"):
            core = model.gpt_neox
            self.family = "gpt_neox"
            self.layers = list(core.layers)
            self.attn = [l.attention for l in self.layers]
            self.mlp = [l.mlp for l in self.layers]
            self.oproj = [l.attention.dense for l in self.layers]
            self.final_norm = core.final_layer_norm
            self.lm_head = model.embed_out
            self.parallel_residual = bool(getattr(cfg, "use_parallel_residual", True))
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            core = model.transformer
            self.family = "gpt2"
            self.layers = list(core.h)
            self.attn = [b.attn for b in self.layers]
            self.mlp = [b.mlp for b in self.layers]
            self.oproj = [b.attn.c_proj for b in self.layers]
            self.final_norm = core.ln_f
            self.lm_head = model.lm_head
            self.parallel_residual = False
        else:
            raise RefusalError(f"unsupported architecture: {type(model).__name__}")

        self.n_layers = len(self.layers)
        self.n_heads = cfg.num_attention_heads

        # Claim B scope: full causal attention only (DERIVATION.md §4).
        lt = getattr(cfg, "layer_types", None) or []
        if getattr(cfg, "use_sliding_window", False) or any("sliding" in str(x) for x in lt):
            raise RefusalError("sliding-window attention: Claim B scope violated")

    def layer_sum(self, pre, attn, mlp):
        """Replicates the model's own residual addition order."""
        if self.parallel_residual:  # HF GPT-NeoX: mlp + attn + hidden
            return mlp + attn + pre
        return (pre + attn) + mlp

    def head_recompose(self, z, layer_idx):
        """A3: sum of per-head W_O slices applied to captured o_proj input z
        [T, H*dh]. Check-side reductions run in fp32 regardless of model dtype."""
        mod = self.oproj[layer_idx]
        w = mod.weight.detach().to("cpu", torch.float32)
        b = mod.bias.detach().to("cpu", torch.float32) if mod.bias is not None else None
        conv1d = type(mod).__name__ == "Conv1D"  # gpt2: weight [in, out]
        return head_contrib(z, w, b, self.n_heads, conv1d)


def head_contrib(z, w, b, n_heads, is_conv1d, heads=None, add_bias=None):
    """Pure, weight-parameterized per-head attention contribution — the single
    implementation certified by G3 (via Adapter.head_recompose) and available
    for the viewer's per-head analysis to reuse with only o_proj weights, no
    live model (Phase 3 Step 4). z: [..., H*dh] captured o_proj input; w/b:
    fp32 o_proj weight/bias; is_conv1d True for gpt2-style [in,out] weight.
    `heads` (default all) selects which heads to sum, so callers can request a
    single head's write. bias is added by default only when summing ALL heads
    (the full attn_out reconstruction); pass add_bias to override. fp32."""
    dh = z.shape[-1] // n_heads
    head_list = list(range(n_heads)) if heads is None else list(heads)
    if add_bias is None:
        add_bias = len(head_list) == n_heads
    acc = None
    for h in head_list:
        zh = z[..., h * dh:(h + 1) * dh]
        ch = zh @ w[h * dh:(h + 1) * dh, :] if is_conv1d else zh @ w[:, h * dh:(h + 1) * dh].T
        acc = ch if acc is None else acc + ch
    if b is not None and add_bias:
        acc = acc + b
    return acc


# ----------------------------------------------------------------------------
# capture

class Capture:
    def __init__(self, adapter):
        self.A = adapter
        self.active = False
        self.handles = []
        L = adapter.n_layers
        for i in range(L):
            self.handles.append(adapter.layers[i].register_forward_pre_hook(
                self._mk_pre(i), with_kwargs=True))
            self.handles.append(adapter.layers[i].register_forward_hook(
                self._mk_post(i), with_kwargs=True))
            self.handles.append(adapter.attn[i].register_forward_hook(
                self._mk_stream("attn_out", i), with_kwargs=True))
            self.handles.append(adapter.mlp[i].register_forward_hook(
                self._mk_stream("mlp_out", i), with_kwargs=True))
            self.handles.append(adapter.oproj[i].register_forward_pre_hook(
                self._mk_oproj(i), with_kwargs=True))

    @staticmethod
    def _hidden(args, kwargs):
        if args:
            return args[0]
        return kwargs["hidden_states"]

    @staticmethod
    def _untuple(out):
        return out[0] if isinstance(out, (tuple, list)) else out

    def _mk_pre(self, i):
        def fn(mod, args, kwargs):
            if self.active:
                self.buf["resid_pre"][i] = grab(self._hidden(args, kwargs)[0])
        return fn

    def _mk_post(self, i):
        def fn(mod, args, kwargs, out):
            if self.active:
                self.buf["resid_post"][i] = grab(self._untuple(out)[0])
        return fn

    def _mk_stream(self, name, i):
        def fn(mod, args, kwargs, out):
            if self.active:
                self.buf[name][i] = grab(self._untuple(out)[0])
        return fn

    def _mk_oproj(self, i):
        def fn(mod, args, kwargs):
            if self.active:
                self.buf["oproj_in"][i] = grab(self._hidden(args, kwargs)[0])
        return fn

    def start(self):
        self.buf = {k: [None] * self.A.n_layers
                    for k in STREAMS + ("oproj_in",)}
        self.active = True

    def collect(self):
        self.active = False
        rec = {}
        for k, v in self.buf.items():
            missing = [i for i, t in enumerate(v) if t is None]
            if missing:
                raise RuntimeError(f"capture hole: stream {k} layers {missing}")
            rec[k] = torch.stack(v)  # [L, T, width]
        return rec


# ----------------------------------------------------------------------------
# cache introspection (G6)

def cache_kv(past):
    if hasattr(past, "layers"):
        pairs = []
        for lyr in past.layers:
            k = getattr(lyr, "keys", None)
            v = getattr(lyr, "values", None)
            if k is None or v is None:
                raise RuntimeError(f"unknown cache layer layout: {type(lyr).__name__}")
            pairs.append((k, v))
        return pairs
    if hasattr(past, "key_cache"):
        return list(zip(past.key_cache, past.value_cache))
    if isinstance(past, (tuple, list)):
        return [(kv[0], kv[1]) for kv in past]
    raise RuntimeError(f"unsupported cache type: {type(past).__name__}")


def cache_snapshot(past):
    return [(k.detach().clone(), v.detach().clone()) for k, v in cache_kv(past)]


def cache_prefix_check(snap, past):
    """Entries for previously-written positions must be bitwise unchanged."""
    worst = 0.0
    bitwise = True
    for (k0, v0), (k1, v1) in zip(snap, cache_kv(past)):
        n = k0.shape[-2]
        for old, new in ((k0, k1[..., :n, :]), (v0, v1[..., :n, :])):
            if not torch.equal(old, new):
                bitwise = False
                worst = max(worst, (old - new).abs().max().item())
    return {"bitwise": bitwise, "max_abs": worst}


# ----------------------------------------------------------------------------
# generation + capture driver (THE production path — validated by the gate)

def select_next(logits_row, sp, gen):
    if not sp["do_sample"]:
        return int(torch.argmax(logits_row).item())
    l = logits_row.float().cpu() / max(sp["temperature"], 1e-8)
    if sp["top_k"] and sp["top_k"] > 0:
        kth = torch.topk(l, min(sp["top_k"], l.numel())).values[-1]
        l = torch.where(l < kth, torch.full_like(l, float("-inf")), l)
    if sp["top_p"] < 1.0:
        srt, idx = torch.sort(l, descending=True)
        cum = torch.softmax(srt, -1).cumsum(-1)
        cut = cum - torch.softmax(srt, -1) > sp["top_p"]
        srt[cut] = float("-inf")
        l = torch.full_like(l, float("-inf")).scatter(0, idx, srt)
    return int(torch.multinomial(torch.softmax(l, -1), 1, generator=gen).item())


def run_capture(model, cap, prompt_ids, n_new, sp, seed, eos_ids=None):
    """Prefill + up to n_new decode steps (last materialized step exists to
    capture the final token's column), then one static pass over the full
    sequence (the Claim B cross-check / per-trace verification reference).
    Each call is a fully independent generation run (fresh cache, fresh RNG).

    If eos_ids is given and a sampled token is in it, generation stops early
    (the EOS token IS generated and captured); stop_reason records eos vs
    max_tokens. EOS stopping changes only WHEN we stop, not any computed value
    — determinism is unaffected."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    eos = set(eos_ids or [])
    records, logits_parts, cache_checks = [], [], []
    stop_reason = "max_tokens"

    with torch.no_grad():
        cap.start()
        out = model(input_ids=prompt_ids, use_cache=True)
        records.append(cap.collect())
        logits_parts.append(out.logits[0].to("cpu", torch.float32))
        past = out.past_key_values
        snap = cache_snapshot(past)

        cur = select_next(out.logits[0, -1], sp, gen)
        gen_ids = [cur]
        for t in range(1, n_new + 1):
            cap.start()
            out = model(input_ids=torch.tensor([[cur]], device=prompt_ids.device),
                        past_key_values=past, use_cache=True)
            records.append(cap.collect())
            logits_parts.append(out.logits[0].to("cpu", torch.float32))
            past = out.past_key_values
            cache_checks.append(cache_prefix_check(snap, past))
            snap = cache_snapshot(past)
            if cur in eos:               # the just-captured token was EOS: stop
                stop_reason = "eos"
                break
            if t < n_new:
                cur = select_next(out.logits[0, -1], sp, gen)
                gen_ids.append(cur)

        final_kv = [(grab(k), grab(v)) for k, v in cache_kv(past)]

        full_ids = torch.cat(
            [prompt_ids, torch.tensor([gen_ids], device=prompt_ids.device)], dim=1)

        cap.start()
        s_out = model(input_ids=full_ids, use_cache=False)
        static_rec = cap.collect()
        static_logits = s_out.logits[0].to("cpu", torch.float32)

    incr = {k: torch.cat([r[k] for r in records], dim=1)
            for k in records[0]}  # [L, T, width]
    incr_logits = torch.cat(logits_parts, dim=0)  # [T, V]
    return {
        "full_ids": full_ids, "gen_ids": gen_ids,
        "incr": incr, "incr_logits": incr_logits,
        "static": static_rec, "static_logits": static_logits,
        "cache_checks": cache_checks, "final_kv": final_kv,
        "stop_reason": stop_reason,
    }


def run_hash(r):
    chunks = [r["full_ids"].to("cpu", torch.int64).numpy().tobytes()]
    for src in ("incr", "static"):
        for k in sorted(r[src]):
            chunks.append(t_bytes(r[src][k]))
    chunks += [t_bytes(r["incr_logits"]), t_bytes(r["static_logits"])]
    for k, v in r["final_kv"]:
        chunks += [t_bytes(k), t_bytes(v)]
    return sha256_bytes(*chunks)


# ----------------------------------------------------------------------------
# checks (used by selfcheck battery AND per-trace verification)

def check_G1(A, rec):
    per_layer = []
    worst = {"max_rel": 0.0, "norm": 0.0}
    for l in range(A.n_layers):
        m = metric(A.layer_sum(rec["resid_pre"][l], rec["attn_out"][l],
                               rec["mlp_out"][l]), rec["resid_post"][l])
        per_layer.append(m)
        if m["max_rel"] >= worst["max_rel"]:
            worst = {**m, "layer": l}
    return {"per_layer_max_abs": [m["max_abs"] for m in per_layer],
            "max_rel": worst["max_rel"],
            "max_norm": max(m["norm"] for m in per_layer), "worst": worst}


def check_G2(A, rec):
    per_boundary = []
    bitwise = True
    for l in range(A.n_layers - 1):
        eq = torch.equal(rec["resid_pre"][l + 1], rec["resid_post"][l])
        d = 0.0 if eq else (rec["resid_pre"][l + 1] - rec["resid_post"][l]).abs().max().item()
        bitwise &= eq
        per_boundary.append({"boundary": l, "bitwise": eq, "max_abs": d})
    return {"bitwise": bitwise, "per_boundary": per_boundary}


def check_G3(A, rec, eps):
    per_layer = []
    worst = {"max_rel": 0.0}
    recomposed = []
    for l in range(A.n_layers):
        z = rec["oproj_in"][l]
        rc = A.head_recompose(z, l)
        recomposed.append(rc)
        m = metric(rc, rec["attn_out"][l])
        per_layer.append(m["max_rel"])
        if m["max_rel"] >= worst["max_rel"]:
            worst = {**m, "layer": l}
    audit = element_audit(torch.stack(recomposed), rec["attn_out"],
                          ("layer", "pos", "dim"), eps)
    return {"per_layer_rel": per_layer, "max_rel": worst["max_rel"],
            "max_norm": worst["norm"], "worst": worst, "element_audit": audit}


def check_G4(A, rec):
    acc = rec["resid_pre"][0].clone()
    for l in range(A.n_layers):
        acc = A.layer_sum(acc, rec["attn_out"][l], rec["mlp_out"][l])
    m = metric(acc, rec["resid_post"][A.n_layers - 1])
    return {"max_abs": m["max_abs"], "max_rel": m["max_rel"],
            "max_norm": m["norm"], "ref_rms": m["ref_rms"]}


def check_G5(A, rec, logits):
    """Recompute Unembed(LN_f(resid_final)) with the model's own modules in
    the model's own dtype/device (round-trip through the fp32 capture is
    lossless), so the comparison isolates the identity, not a dtype change."""
    w = A.lm_head.weight
    x = rec["resid_post"][A.n_layers - 1].to(w.device, w.dtype)
    with torch.no_grad():
        recomputed = A.lm_head(A.final_norm(x)).to("cpu", torch.float32)
    m = metric(recomputed, logits)
    return {"max_abs": m["max_abs"], "max_rel": m["max_rel"],
            "max_norm": m["norm"], "ref_rms": m["ref_rms"]}


def check_G7(A, r, tokens, eps):
    out = {"streams": {}, "overall_max_rel": 0.0, "overall_max_norm": 0.0}
    T = r["static_logits"].shape[0]
    # Dtype-independent exact sub-gate: layer-0 resid_pre is the embedding
    # lookup of identical token ids — must match bitwise in ANY dtype. Gross
    # positional/cache bugs surface here even where fp16 tolerances are loose.
    out["layer0_resid_pre_bitwise"] = torch.equal(
        r["incr"]["resid_pre"][0], r["static"]["resid_pre"][0])
    for k in STREAMS:
        a, b = r["incr"][k], r["static"][k]  # [L, T, d]
        per_pos = []
        for p in range(T):
            m = metric(a[:, p, :], b[:, p, :])
            per_pos.append({"pos": p, "token": tokens[p], **m})
        per_layer = [metric(a[l], b[l])["max_rel"] for l in range(A.n_layers)]
        s_rel = max(pp["max_rel"] for pp in per_pos)
        s_norm = max(pp["norm"] for pp in per_pos)
        out["streams"][k] = {
            "max_rel": s_rel,
            "max_norm": s_norm,
            "per_layer_rel": per_layer,
            "per_position": per_pos,
            "worst_positions": sorted(per_pos, key=lambda x: -x["max_rel"])[:5],
            "element_audit": element_audit(a, b, ("layer", "pos", "dim"), eps),
        }
        out["overall_max_rel"] = max(out["overall_max_rel"], s_rel)
        out["overall_max_norm"] = max(out["overall_max_norm"], s_norm)

    lp = []
    for p in range(T):
        m = metric(r["incr_logits"][p], r["static_logits"][p])
        srt = torch.sort(r["static_logits"][p], descending=True).values
        lp.append({"pos": p, "token": tokens[p], **m,
                   "argmax_match": int(torch.argmax(r["incr_logits"][p]))
                                   == int(torch.argmax(r["static_logits"][p])),
                   "static_top1_top2_margin": (srt[0] - srt[1]).item()})
    out["logits"] = {"max_rel": max(x["max_rel"] for x in lp),
                     "max_norm": max(x["norm"] for x in lp), "per_position": lp,
                     "argmax_all_match": all(x["argmax_match"] for x in lp),
                     "element_audit": element_audit(
                         r["incr_logits"], r["static_logits"], ("pos", "vocab"), eps)}
    out["overall_max_rel"] = max(out["overall_max_rel"], out["logits"]["max_rel"])
    out["overall_max_norm"] = max(out["overall_max_norm"], out["logits"]["max_norm"])
    return out


def check_G8(r1, r2, tok):
    """Independent regeneration: two fresh generation runs with identical
    recorded params must produce the identical token sequence. (The naive
    form — captured logits argmax to the run's own next token — is circular:
    the run selected those tokens FROM those logits. Cross-kernel argmax
    agreement incr-vs-static is recorded per position in G7.)"""
    ids1 = r1["full_ids"][0].tolist()
    ids2 = r2["full_ids"][0].tolist()
    match = ids1 == ids2
    out = {"independent_runs_token_match": match,
           "run1_len": len(ids1), "run2_len": len(ids2)}
    if not match:
        k = next((i for i, (x, y) in enumerate(zip(ids1, ids2)) if x != y),
                 min(len(ids1), len(ids2)))
        out["first_divergence"] = {
            "pos": k,
            "run1_token_id": ids1[k] if k < len(ids1) else None,
            "run2_token_id": ids2[k] if k < len(ids2) else None,
            "run1_token": tok.decode([ids1[k]]) if k < len(ids1) else None,
            "run2_token": tok.decode([ids2[k]]) if k < len(ids2) else None,
        }
    return out


# ----------------------------------------------------------------------------
# trust chain (refusal semantics, no override path)

def _load_gate(path):
    if not os.path.exists(path):
        raise RefusalError(f"missing gate record: {path}; run the selfcheck first")
    with open(path) as f:
        return json.load(f)


def _staleness(g):
    import transformers
    stale = []
    if g.get("code_sha256") != gated_code_sha256():
        stale.append("gated code hash")
    if g.get("torch_version") != torch.__version__:
        stale.append("torch version")
    if g.get("transformers_version") != transformers.__version__:
        stale.append("transformers version")
    return stale


def check_gate(model_id, revision=None, gates_dir="gates"):
    """Valid PRIMARY certificate (cpu/fp32) or refusal. No override path."""
    p = os.path.join(gates_dir, model_id.replace("/", "--") + ".json")
    g = _load_gate(p)
    if gate_selfhash(g) != g.get("record_sha256"):
        raise RefusalError(f"gate {p} self-hash missing/mismatched (hand-edited "
                           f"or pre-hash era); rerun selfcheck")
    if g.get("role") != "primary_certificate" or not g.get("passed"):
        raise RefusalError(f"gate for {model_id} is not a passing primary certificate")
    if revision is not None and g.get("revision") != revision:
        raise RefusalError(f"gate for {model_id} certifies revision "
                           f"{g.get('revision')}, loaded {revision}; rerun selfcheck")
    stale = _staleness(g)
    if stale:
        raise RefusalError(f"gate for {model_id} stale ({', '.join(stale)}); rerun selfcheck")
    return g, p


def check_supplement(model_id, revision, device, dtype, gpu_name=None, gates_dir="gates"):
    """Valid hardware/dtype supplement for exactly (device, dtype, GPU) or
    refusal. cuda-profile thresholds (fp16 AND fp32) are hardware-scoped:
    a supplement measured on one GPU is refused on another (DERIVATION.md §8;
    SPEC_perstack_rails.md §3d — records are natively per-GPU-variant)."""
    p = _resolve_gpu_scoped(
        os.path.join(gates_dir, model_id.replace("/", "--")
                     + f".supplement-{device.replace(':', '')}-{dtype}"),
        device, gpu_name)
    g = _load_gate(p)
    if gate_selfhash(g) != g.get("record_sha256"):
        raise RefusalError(f"supplement {p} self-hash missing/mismatched "
                           f"(hand-edited or pre-hash era); rerun selfcheck")
    if g.get("role") != "hardware_dtype_supplement" or not g.get("passed"):
        raise RefusalError(f"no passing {device}/{dtype} supplement for {model_id}")
    if revision is not None and g.get("revision") != revision:
        raise RefusalError(f"supplement for {model_id} certifies revision "
                           f"{g.get('revision')}, loaded {revision}; rerun selfcheck")
    if gpu_name is not None and gpu_class(g.get("gpu_name")) != gpu_class(gpu_name):
        raise RefusalError(
            f"supplement for {model_id} was measured on {g.get('gpu_name')!r} "
            f"(class {gpu_class(g.get('gpu_name'))!r}), this device is "
            f"{gpu_name!r} (class {gpu_class(gpu_name)!r}); cuda-profile floors "
            f"are hardware-class-scoped — re-measure on this GPU")
    stale = _staleness(g)
    if stale:
        raise RefusalError(f"supplement for {model_id} stale ({', '.join(stale)}); "
                           f"rerun selfcheck")
    return g, p


# ----------------------------------------------------------------------------
# per-(model, profile) trace-verification calibration records
#
# The SELFCHECK BATTERY (G1-G9) certifies the instrument on a fixed standardized
# probe and keeps its global TOL_BY_DTYPE thresholds. Open-ended TRACES vary in
# drift with prompt content (measured: code/whitespace prompts drive the
# massive-activation channel and ~2x the cross-kernel drift ceiling), so trace
# validity is gated by a per-(model, profile) COMPOSITE record instead of one
# global scalar. Records are produced by `selfcheck.py --calibrate` (measure
# only; never writes a gate), start `approved: false`, and are enforced only
# after a human sets `approved: true`. Gates/traces bind the record's sha256.
# Scoped to threads=1 and (fp16) the recorded GPU. See DERIVATION.md §9.

def profile_slug(device, dtype):
    return f"{device.replace(':', '')}-{dtype}"


def gpu_class(gpu_name):
    """Hardware EQUIVALENCE CLASS for cuda-record scoping (SPEC_perstack_rails.md
    Amendment A2, Matt-approved 2026-07-09). The NVIDIA A100-80GB PCIe and SXM4
    variants are ONE class: measured equivalent on the full calibration suite —
    6 roster models compared across variants, bitwise-identical for 2, max
    1.21e-7 relative deviation (last-bit fp32 rounding) on the rest; same GA100
    die, deterministic kernels
    (evidence: runs/stackfloors/a100-variant-equivalence.json). Records still
    store the exact gpu_name they measured on (provenance); only SCOPING uses the
    class. Any other GPU is its own class (e.g. the local RX 6900 XT)."""
    if gpu_name and "MIG" in gpu_name:
        return gpu_name   # MIG slices: own class — no equivalence evidence exists
    if gpu_name and "A100" in gpu_name and "80GB" in gpu_name:
        return "NVIDIA A100-80GB"
    return gpu_name


def gpu_slug(gpu_name):
    """Filesystem-safe slug of the GPU CLASS: both A100-80GB variants ->
    'NVIDIA-A100-80GB' (one cert per class); other GPUs slug their own name."""
    return "".join(c if (c.isalnum() or c in "-_.") else "-"
                   for c in gpu_class(gpu_name))


def _resolve_gpu_scoped(base_no_ext, device, gpu_name):
    """Path for a cuda-profile record. With gpu_name: the exact per-variant file.
    Without (callers that don't know the GPU, e.g. profile listings): resolve by
    glob — exactly one variant file matches, or refuse as ambiguous; zero matches
    falls back to the legacy unsuffixed path (pre-scoping records; they die of
    code-hash staleness on their own)."""
    if not device.startswith("cuda"):
        return base_no_ext + ".json"
    if gpu_name is not None:
        return f"{base_no_ext}.{gpu_slug(gpu_name)}.json"
    import glob as _glob
    hits = sorted(_glob.glob(base_no_ext + ".*.json"))
    if len(hits) > 1:
        raise RefusalError(
            f"multiple GPU-variant records match {base_no_ext}.*.json "
            f"({', '.join(os.path.basename(h) for h in hits)}); pass gpu_name")
    return hits[0] if hits else base_no_ext + ".json"


def calibration_path(model_id, device, dtype, gates_dir="gates", gpu_name=None):
    base = os.path.join(gates_dir, "calibration-" + model_id.replace("/", "--")
                        + f"-{profile_slug(device, dtype)}")
    return _resolve_gpu_scoped(base, device, gpu_name)


def trace_composite(v_g7, g1_max_rel):
    """Extract the composite trace-validity scalars from a check_G7 result +
    the additivity max_rel. Pure reduction over already-measured numbers."""
    streams = v_g7["streams"]
    depth = {}
    for s, st in streams.items():
        pl = st["per_layer_rel"]
        mx = max(pl) or 1.0
        # fraction of worst drift already present in layers 0-1: a positional/
        # cache bug is O(1) with layer-0 onset; rounding grows smoothly.
        depth[s] = max(pl[0], pl[1] if len(pl) > 1 else 0.0) / mx
    lp = v_g7["logits"]["per_position"]
    argmax_rate = sum(p["argmax_match"] for p in lp) / max(1, len(lp))
    return {
        "g1_max_rel": g1_max_rel,
        "stream_max_rel": max(st["max_rel"] for st in streams.values()),
        "logits_max_rel": v_g7["logits"]["max_rel"],
        "worst_abs_ulp": max(st["element_audit"]["worst_by_abs"]["diff_in_ulp_at_ref"]
                             for st in streams.values()),
        "floor_bound_eps_rms": max(
            st["element_audit"]["floor_masking_bound"]["in_units_of_eps_x_rms"]
            for st in streams.values()),
        "layer0_bitwise": bool(v_g7["layer0_resid_pre_bitwise"]),
        "argmax_all_match": bool(v_g7["logits"]["argmax_all_match"]),
        "argmax_rate": argmax_rate,
        "early_layer_fraction": max(depth.values()),
    }


def evaluate_trace_validity(record, comp):
    """(valid, reasons) for a trace's composite metrics against an APPROVED
    calibration record.

    Cross-kernel magnitude drift (stream/logits max_rel) depends
    non-monotonically on sequence length via the static reference pass's kernel
    shape, so a single-length calibration cannot tightly bound it; the magnitude
    gates therefore use generous headroom that clears the measured rounding
    regime and stays orders below the O(1)-relative bug regime (DERIVATION.md
    §9). The load-bearing discrimination is the EXACT invariants (layer-0
    resid_pre bitwise — any dtype), token integrity (argmax agreement rate), and
    the depth-onset shape (a positional/cache bug is O(1) at layer 0/1)."""
    t = record["thresholds"]
    reasons = {
        "g1_additivity": comp["g1_max_rel"] <= t["g1_max_rel"],
        "stream_max_rel": comp["stream_max_rel"] <= t["stream_max_rel"],
        "logits_max_rel": comp["logits_max_rel"] <= t["logits_max_rel"],
        "worst_abs_ulp": comp["worst_abs_ulp"] <= t["worst_abs_ulp"],
        "layer0_resid_pre_bitwise": comp["layer0_bitwise"],
        "argmax_agreement": comp["argmax_rate"] >= t.get("argmax_min_rate", 0.0),
        "smooth_depth_onset": comp["early_layer_fraction"] <= t["early_layer_fraction"],
    }
    if t.get("floor_bound_eps_rms") is not None:
        reasons["floor_bound"] = comp["floor_bound_eps_rms"] <= t["floor_bound_eps_rms"]
    return all(reasons.values()), reasons


def check_calibration(model_id, revision, device, dtype, gpu_name=None,
                      gates_dir="gates"):
    """Approved, current calibration record for exactly this (model, profile,
    GPU-for-fp16) or refusal. No override path. Records carry `approved`, a
    self-sha256 excluding the sha field, and code/lib staleness like gates."""
    p = calibration_path(model_id, device, dtype, gates_dir, gpu_name=gpu_name)
    if not os.path.exists(p):
        raise RefusalError(
            f"no calibration record for {model_id} [{profile_slug(device, dtype)}"
            f"{', ' + gpu_name if gpu_name else ''}]; "
            f"run: selfcheck.py --calibrate --model {model_id} "
            f"--device {device} --dtype {dtype}, then a human approves it")
    with open(p) as f:
        rec = json.load(f)
    if not rec.get("approved"):
        raise RefusalError(
            f"calibration record {p} is not approved (set approved:true after "
            f"human review of its measured floors and bug-signature checks)")
    if calibration_selfhash(rec) != rec.get("record_sha256"):
        raise RefusalError(f"calibration record {p} was edited after approval "
                           f"(self-hash mismatch); re-calibrate and re-approve")
    if rec.get("approval_sha256") != approval_hash(rec):
        raise RefusalError(f"calibration record {p} approval hash missing/"
                           f"mismatched (approved bit set outside the approve "
                           f"tool); re-approve via approve_calibration.py")
    if not rec.get("signatures_ok") and "--force" not in str(rec.get("approved_note")):
        raise RefusalError(f"calibration record {p} has failed bug signatures; "
                           f"consumption requires an approval note that "
                           f"explicitly acknowledges --force")
    if "threads" not in rec:
        raise RefusalError(f"calibration record {p} lacks the threads field "
                           f"(thread-scoping is mandatory); re-calibrate")
    if rec.get("model") != model_id or rec.get("dtype") != dtype \
            or rec.get("device") != device:
        raise RefusalError(f"calibration record {p} profile mismatch")
    if revision is not None and rec.get("revision") != revision:
        raise RefusalError(f"calibration record {p} certifies revision "
                           f"{rec.get('revision')}, loaded {revision}; re-calibrate")
    if device.startswith("cuda") and gpu_name is not None \
            and gpu_class(rec.get("gpu_name")) != gpu_class(gpu_name):
        raise RefusalError(
            f"calibration for {model_id} measured on {rec.get('gpu_name')!r} "
            f"(class {gpu_class(rec.get('gpu_name'))!r}), this GPU is "
            f"{gpu_name!r} (class {gpu_class(gpu_name)!r}); cuda-profile floors "
            f"(fp16 AND fp32) are hardware-class-scoped — re-calibrate on this GPU")
    stale = _staleness(rec)
    if stale:
        raise RefusalError(f"calibration record {p} stale ({', '.join(stale)}); "
                           f"re-calibrate")
    return rec, p


def calibration_selfhash(rec):
    """sha256 over the record with volatile/self fields removed — binds the
    approved content so post-approval edits are detectable."""
    clean = {k: v for k, v in rec.items()
             if k not in ("record_sha256", "approved", "approved_note",
                          "approval_sha256")}
    return sha256_bytes(json.dumps(clean, sort_keys=True).encode())


def approval_hash(rec):
    """Approval-time hash covering the approval BIT and note (adversarial-review
    decision 6): flipping approved:true by hand without re-running the approve
    tool is now detectable. Keyless by design (advisory tamper evidence, not
    cryptographic authority — the human running the approve tool IS the gate)."""
    return sha256_bytes((str(rec.get("record_sha256", "")) + "|approved|"
                         + str(rec.get("approved_note", ""))).encode())


def gate_selfhash(rec):
    """sha256 over a gate/supplement record minus its own hash — gate records
    were previously hash-free (a hand-written passed:true record was admissible;
    adversarial-review decision 6)."""
    clean = {k: v for k, v in rec.items() if k != "record_sha256"}
    return sha256_bytes(json.dumps(clean, sort_keys=True).encode())


# ----------------------------------------------------------------------------
# per-stack fp32 floors (SPEC_perstack_rails.md, approved+locked 2026-07-09)
#
# The fp32 INEXACT battery gates (G3/G5/G7) and the depth-onset applicability
# floor are measured rounding floors of a software/hardware stack, not portable
# constants (REDTEAM_calibration.md; demonstrated three times: base-4B G7 on
# cu128 CPU-fleet, hybrid-8B G3 on cu128 CPU-fleet — both vanish on A100-host
# CPUs — and Instruct's cuda/fp32 supplement G7). Each stack (keyed by torch
# build string) carries a human-approved floors record derived by the locked
# formulas (2x worst clean battery measurement across the roster; depth floor
# min(4x worst clean fp32 stream drift, 1e-2)). A primary fp32 gate REFUSES to
# run without an approved floors record for its stack — inheritance across
# stacks is what failed and has no path here. Floors records key on
# (torch, transformers) versions, NOT the gated code hash: they are measured
# model/stack properties; the code hash of the runs that measured them is
# recorded as provenance (`measured_under_code_sha256`).

def stack_slug(torch_version=None):
    tv = torch_version if torch_version is not None else torch.__version__
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in tv)


def stackfloors_path(gates_dir="gates", torch_version=None):
    return os.path.join(gates_dir, f"stackfloors-{stack_slug(torch_version)}.json")


def stackfloors_selfhash(rec):
    clean = {k: v for k, v in rec.items()
             if k not in ("record_sha256", "approved", "approved_note")}
    return sha256_bytes(json.dumps(clean, sort_keys=True).encode())


def check_stackfloors(gates_dir="gates", device=None):
    """Approved per-stack fp32 floors record for the RUNNING stack, or refusal.
    No override path; no cross-stack inheritance. When `device` is given, the
    record's derived_on_device class must match (adversarial-review decision 1:
    A100 floors come from A100 measurements, cpu floors from cpu measurements —
    a torch build can host both device classes)."""
    import transformers
    p = stackfloors_path(gates_dir)
    if not os.path.exists(p):
        raise RefusalError(
            f"no per-stack fp32 floors record for torch {torch.__version__} "
            f"({p}); measure the battery across the roster on THIS stack "
            f"(selfcheck.py --measure-only), derive floors per "
            f"SPEC_perstack_rails.md, and have a human approve the record")
    with open(p) as f:
        rec = json.load(f)
    if not rec.get("approved"):
        raise RefusalError(f"stack floors record {p} is not approved")
    if stackfloors_selfhash(rec) != rec.get("record_sha256"):
        raise RefusalError(f"stack floors record {p} was edited after approval "
                           f"(self-hash mismatch); re-derive and re-approve")
    if rec.get("torch_version") != torch.__version__:
        raise RefusalError(f"stack floors record {p} is for torch "
                           f"{rec.get('torch_version')}, running "
                           f"{torch.__version__}")
    if rec.get("transformers_version") != transformers.__version__:
        raise RefusalError(f"stack floors record {p} is for transformers "
                           f"{rec.get('transformers_version')}, running "
                           f"{transformers.__version__}")
    need = {"G3_per_head", "G5_logits_ident", "G7_incr_vs_static",
            "depth_signature_floor"}
    missing = need - set(rec.get("floors", {}))
    if missing:
        raise RefusalError(f"stack floors record {p} missing floors: "
                           f"{sorted(missing)}")
    if device is not None:
        want = "cuda" if device.startswith("cuda") else "cpu"
        if rec.get("derived_on_device") != want:
            raise RefusalError(
                f"stack floors record {p} derived on device class "
                f"{rec.get('derived_on_device')!r}, this battery runs on "
                f"{want!r}; floors must be measured on the device class they "
                f"gate — derive a {want} floors record for this stack")
    return rec, p
