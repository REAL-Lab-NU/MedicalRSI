"""The two training steps, run with torchrun.

  joint      one joint update of both roles on a round's batch: Improver samples {"prompt", "completion", "adv"} and
             Solver attempts {"messages", "adv"} (environment messages masked), L = (L_clip(D_imp) + L_clip(D_sol)) / 2
             with each role's loss averaged over its own tokens; the AdamW state is kept across rounds.
  warmstart  supervised fine-tuning of a new LoRA adapter on {"prompt", "completion"} rows.

Log-probabilities are computed only at the labelled positions."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

HEAD = "<|im_start|>assistant\n<think>\n\n</think>\n\n"     # the generation prompt with thinking off
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|gate_proj|up_proj|down_proj)"


# ---------------------------------------------------------------- encoding

def render(tok, msgs, gen):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)


def end_of_turn(tok):
    m = [{"role": "user", "content": "U"}, {"role": "assistant", "content": "XYZZY"}]
    return render(tok, m, False).split("XYZZY", 1)[1].rstrip("\n")


def encode_template(tok, msgs, max_len, max_turn):
    """(ids, labels) of a conversation in the model's chat template, thinking off; assistant turns are labelled."""
    text = render(tok, msgs, False)
    eot = end_of_turn(tok)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        gen, full = render(tok, msgs[:i], True), render(tok, msgs[:i + 1], False)
        if not (full.startswith(gen) and text.startswith(full)):
            raise ValueError(f"chat template is not prefix-consistent at message {i}")
        body = full[len(gen):]
        k = body.rfind(eot)
        spans.append((len(gen), len(gen) + k, len(gen) + k + len(eot)))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    lab = [-100] * len(ids)
    for a, b, c in spans:
        content = [j for j, (s, e) in enumerate(offs) if s >= a and e <= b and e > s]
        keep_eot = len(content) < max_turn
        for j, (s, e) in enumerate(offs):
            if e > s and s >= a and (e <= b or (keep_eot and e <= c)):
                lab[j] = ids[j]
    return ids[:max_len], lab[:max_len]


def encode_episode(tok, msgs, max_len, max_turn):
    """(ids, labels) of a Solver attempt in the Qwen chat format; assistant turns are labelled."""
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    ids, lab = [], []
    for m in msgs:
        if m["role"] == "assistant":
            h, c, e, nl = enc(HEAD), enc(m["content"]), enc("<|im_end|>"), enc("\n")
            ids += h + c + e + nl
            lab += [-100] * len(h) + c + (e if len(c) < max_turn else [-100] * len(e)) + [-100] * len(nl)
        else:
            x = enc(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            ids += x
            lab += [-100] * len(x)
    return ids[:max_len], lab[:max_len]


def encode_harness(tok, prompt, completion, encoding, max_len):
    """(ids, labels) of an Improver sample: the prompt masked, the harness text labelled as written."""
    if encoding != "qwen":
        return encode_template(tok, [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
                               max_len, 10 ** 9)
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]
    ci = tok(completion + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return (pi + ci)[:max_len], ([-100] * len(pi) + ci)[:max_len]


# ---------------------------------------------------------------- model

def load_model(path, dtype=torch.bfloat16):
    import transformers
    try:
        return transformers.AutoModelForImageTextToText.from_pretrained(path, dtype=dtype)
    except ValueError:
        return transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype)


def patch_logprob_forward(cls):
    """With `dpo_labels`, the forward returns the log-probabilities of the labelled tokens ([batch, seq - 1])."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    orig = cls.forward

    def forward(self, input_ids=None, dpo_labels=None, **kw):
        if dpo_labels is None:
            return orig(self, input_ids=input_ids, **kw)
        kw.pop("labels", None)
        kw.pop("logits_to_keep", None)
        kw["use_cache"] = False
        h = self.model(input_ids=input_ids, **kw).last_hidden_state[:, :-1]
        tgt = dpo_labels[:, 1:]
        m = tgt != -100
        logits = self.lm_head(h[m]).float()
        scale = getattr(self.config, "logits_scaling", None)
        if scale:
            logits = logits / scale
        cap = getattr(self.config.get_text_config(), "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt[m].unsqueeze(-1)).squeeze(-1)
        ptl = torch.zeros(tgt.shape, dtype=lp.dtype, device=lp.device)
        ptl[m] = lp
        return CausalLMOutputWithPast(logits=ptl)
    cls.forward = forward


def setup():
    dist.init_process_group("nccl")
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    return dist.get_rank(), dist.get_world_size(), local, torch.device("cuda", local)


def wrap(model, dev, local):
    model.to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False,
                                                     broadcast_buffers=False)


# ---------------------------------------------------------------- joint update

def joint_update(a):
    rank, world, local, dev = setup()
    from peft import PeftModel
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model)
    patch_logprob_forward(type(model))
    model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    ddp = wrap(model, dev, local)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    if a.optim and (Path(a.optim) / "optim.pt").exists():
        opt.load_state_dict(torch.load(Path(a.optim) / "optim.pt", map_location=dev))
        for g in opt.param_groups:
            g["lr"] = a.lr

    rows = [json.loads(line) for line in open(a.batch)]
    roles = sorted({r.get("role", "all") for r in rows})
    weight = {ro: 1.0 / len(roles) for ro in roles}
    per = math.ceil(len(rows) / world)
    rows = rows + [dict(rows[i % len(rows)], adv=0.0) for i in range(per * world - len(rows))]
    enc = []
    for r in rows[rank::world]:
        if "messages" in r:
            ids_, lab_ = (encode_episode if a.encoding == "qwen" else encode_template)(tok, r["messages"], a.max_len,
                                                                                       a.turn_tokens)
        else:
            ids_, lab_ = encode_harness(tok, r["prompt"], r["completion"], a.encoding, 10 ** 9)
        enc.append((ids_, lab_, float(r["adv"]), r.get("role", "all")))
    n_tok = torch.tensor([sum(sum(x != -100 for x in lab_) for _, lab_, adv, ro in enc if adv != 0.0 and ro == role)
                          for role in roles], device=dev, dtype=torch.float32)
    dist.all_reduce(n_tok)
    denom = {role: max(n_tok[k].item(), 1.0) for k, role in enumerate(roles)}

    ddp.train()
    t0, stats, old, per_step = time.time(), {"loss": 0.0, "logp_tok": 0.0, "n_tok": 0}, [None] * len(enc), []
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        clipped = n_tok_step = kl = loss_step = 0.0
        for i, (ids_, lab_, adv, role) in enumerate(enc):
            ids = torch.tensor([ids_], device=dev)
            labels = torch.tensor([lab_], device=dev)
            lab_pos = labels[0, 1:] != -100
            ctx = ddp.no_sync() if i != len(enc) - 1 else torch.enable_grad()
            with ctx:
                ptl = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), dpo_labels=labels, return_dict=True,
                          use_cache=False).logits.reshape(-1)[lab_pos]
                if step == 0:
                    old[i] = ptl.detach()
                ratio = torch.exp(ptl - old[i])
                surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - a.clip_low, 1 + a.clip_high) * adv)
                loss = -surr.sum() * weight[role] / denom[role]
                loss.backward()
            with torch.no_grad():
                if adv != 0.0:
                    clipped += ((ratio < 1 - a.clip_low) | (ratio > 1 + a.clip_high)).sum().item()
                    n_tok_step += ratio.numel()
                    kl += (old[i] - ptl.detach()).sum().item()
            loss_step += loss.item()
            if step == 0:
                stats["loss"] += loss.item()
                stats["logp_tok"] += ptl.detach().sum().item()
                stats["n_tok"] += int(lab_pos.sum())
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        opt.step()
        st_ = torch.tensor([loss_step, clipped, n_tok_step, kl, float(gn)], device=dev, dtype=torch.float64)
        dist.all_reduce(st_)
        per_step.append({"loss": st_[0].item(), "clip_frac": st_[1].item() / max(st_[2].item(), 1),
                         "approx_kl_tok": st_[3].item() / max(st_[2].item(), 1), "grad_norm": st_[4].item() / world})

    tot = torch.tensor([stats["loss"], stats["logp_tok"], stats["n_tok"]], device=dev, dtype=torch.float64)
    dist.all_reduce(tot)
    if rank == 0:
        ddp.module.save_pretrained(a.out)
        if a.optim:
            Path(a.optim).mkdir(parents=True, exist_ok=True)
            torch.save(opt.state_dict(), Path(a.optim) / "optim.pt")
        info = {"samples": len(rows), "loss": tot[0].item(), "mean_logp_per_token": tot[1].item() / max(tot[2].item(), 1),
                "grad_norm": per_step[0]["grad_norm"], "seconds": round(time.time() - t0, 1), "lr": a.lr, "steps": a.steps,
                "per_step": per_step, "role_tokens": {ro: denom[ro] for ro in roles}}
        (Path(a.out) / "train_info.json").write_text(json.dumps(info))
        print(json.dumps(info))
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------- warm start

def warm_start(a):
    rank, world, local, dev = setup()
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model)
    patch_logprob_forward(type(model))
    model = get_peft_model(model, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             task_type="CAUSAL_LM", target_modules=a.lora_targets))
    ddp = wrap(model, dev, local)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)

    data = []
    for r in map(json.loads, open(a.data)):
        prompt = r["prompt"].replace(f"({a.label_in_data})", f"({a.label})") if a.label_in_data else r["prompt"]
        data.append(encode_harness(tok, prompt, r["completion"], a.encoding, a.max_len))
    per_epoch = math.ceil(len(data) / a.batch)
    total = per_epoch * a.epochs
    warm = max(1, round(a.warmup * total))

    def lr_at(step):
        if step < warm:
            return a.lr * (step + 1) / warm
        return a.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    ddp.train()
    t0, log, step = time.time(), [], 0
    for ep in range(a.epochs):
        order = list(range(len(data)))
        random.Random(a.seed + ep).shuffle(order)
        for s0 in range(0, len(order), a.batch):
            batch = order[s0:s0 + a.batch]
            per = math.ceil(len(batch) / world)
            items = [(i, 1.0) for i in batch] + [(batch[0], 0.0)] * (per * world - len(batch))
            mine = items[rank::world]
            ntok = torch.tensor([sum(sum(x != -100 for x in data[i][1]) * w for i, w in mine)], device=dev,
                                dtype=torch.float32)
            dist.all_reduce(ntok)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for j, (i, w) in enumerate(mine):
                ids = torch.tensor([data[i][0]], device=dev)
                lab = torch.tensor([data[i][1]], device=dev)
                ctx = ddp.no_sync() if j < len(mine) - 1 else contextlib.nullcontext()
                with ctx:
                    ptl = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), dpo_labels=lab, return_dict=True,
                              use_cache=False).logits
                    loss = -ptl.sum() * w / ntok.item()
                    loss.backward()
                loss_sum += loss.item()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            st_ = torch.tensor([loss_sum], device=dev, dtype=torch.float64)
            dist.all_reduce(st_)
            if rank == 0:
                log.append({"step": step, "epoch": ep, "loss": round(st_.item(), 4), "lr": lr_at(step),
                            "grad_norm": round(float(gn), 4), "tokens": int(ntok.item()), "seconds": round(time.time() - t0)})
                print(json.dumps(log[-1]), flush=True)
            step += 1
    if rank == 0:
        ddp.module.save_pretrained(a.out)
        (Path(a.out) / "train_info.json").write_text(json.dumps(
            {"data": a.data, "examples": len(data), "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "steps": total,
             "warmup_steps": warm, "seconds": round(time.time() - t0), "log": log}, indent=1))
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("joint")
    g.add_argument("--model", required=True)
    g.add_argument("--batch", required=True)
    g.add_argument("--adapter", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--optim", default=None)
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--steps", type=int, default=2)
    g.add_argument("--clip_low", type=float, default=0.2)
    g.add_argument("--clip_high", type=float, default=0.28)
    g.add_argument("--max_len", type=int, default=32768)
    g.add_argument("--turn_tokens", type=int, default=512)
    g.add_argument("--encoding", choices=["qwen", "template"], default="qwen")
    s = sub.add_parser("warmstart")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--batch", type=int, default=16, help="examples per optimizer step over all ranks")
    s.add_argument("--warmup", type=float, default=0.05)
    s.add_argument("--max_len", type=int, default=36864)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--lora_rank", type=int, default=64)
    s.add_argument("--lora_alpha", type=int, default=128)
    s.add_argument("--lora_dropout", type=float, default=0.05)
    s.add_argument("--lora_targets", default=LORA_TARGETS)
    s.add_argument("--encoding", choices=["qwen", "template"], default="qwen")
    s.add_argument("--label", default="Qwen3.5-9B", help="the model's name in the prompts")
    s.add_argument("--label_in_data", default="Qwen3.5-9B", help="the Solver name used in the data")
    a = ap.parse_args()
    {"joint": joint_update, "warmstart": warm_start}[a.cmd](a)


if __name__ == "__main__":
    main()
