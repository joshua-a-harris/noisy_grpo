import logging
import os
import sys
from typing import Union, Dict, List, Optional

import numpy as np
import wandb
import torch


def setup_logging(run_name):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(f'{run_name}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# --------------------------- RL logger ----------------------------- #
class RLLogger:
    """
    Centralizes all training-time metrics and Weights & Biases logging.
    Holds any logging-only state and step counters.
    """

    def __init__(self, *, wandb_enabled: bool):
        self.wandb_enabled = bool(wandb_enabled)
        self.opt_step = 0          # increments per optimizer update
        self.train_step_axis = 0   # increments per outer train step
        self._last_loss_stats: Dict[str, float] = {}
        self._last_prefix_stats: Dict[str, float] = {}

    def store_prefix_stats(self, completions: List[str]) -> None:
        props, lens = get_prefix_proportions(completions)
        props = np.array(props, dtype=float)
        lens = np.array(lens, dtype=float)
        self._last_prefix_stats = {
            "prefix/mean_prop_len_prefix": float(props.mean()) if props.size else 0.0,
            "prefix/min_len_prefix": float(lens.min()) if lens.size else 0.0,
            "prefix/prop_seq_identical": float((props > 0.99).sum() / props.shape[0]) if props.size else 0.0,
        }

    def update_loss_stats(
            self,
            logits_n: torch.Tensor,
            logits_c: torch.Tensor,
            mask: torch.Tensor,
            completion_tok_log_probs_n: torch.Tensor,
            entropy_n: torch.Tensor | float,
            abs_noise_mean: torch.Tensor | float,
    ) -> None:
        """
        Computes per-backward metrics and stores them.
        Caller will aggregate across grad-accum steps.
        """
        with torch.no_grad():
            mask_bool = mask.bool()
            mask_f = mask_bool.float()
            den = mask_f.sum().clamp_min(1)

            # Logit gap
            top2_vals = logits_n.topk(2, dim=-1).values  # (B, Tpred, 2)
            gaps = top2_vals[..., 0] - top2_vals[..., 1]  # (B, Tpred)
            gap_mean = ((gaps * mask_f).sum() / den).detach().item()

            # Flips
            pred_n = logits_n.argmax(-1)
            pred_c = logits_c.argmax(-1)
            flip_mask = (pred_n != pred_c) & mask_bool
            flip_rate = (flip_mask.float().sum() / den).detach().item()

            # KL
            p_n = logits_n.softmax(-1)
            logp_c = logits_c.log_softmax(-1)
            kl_nc = (p_n * (p_n.log() - logp_c)).sum(-1)
            kl_mean = ((kl_nc * mask_f).sum() / den).detach().item()

            # ONLY FOR FLIPS: clean probability of the noisy-greedy token
            # "How likely was the noisy greedy token under the clean distribution?"
            p_c = logp_c.exp()  # logits_c.softmax(-1), but reuse logp_c for stability
            noisy_under_clean = p_c.gather(-1, pred_n.unsqueeze(-1)).squeeze(-1)  # (B, Tpred)
            max_under_clean = p_c.gather(-1, pred_c.unsqueeze(-1)).squeeze(-1)  # (B, Tpred)
            prob_gap_flip_c = max_under_clean - noisy_under_clean
            flip_mask_f = flip_mask.float()
            flip_den = flip_mask_f.sum().clamp_min(1)
            flip_clean_prob_mean = ((noisy_under_clean * flip_mask_f).sum() / flip_den).detach().item()
            flip_clean_gap_prob_mean = ((prob_gap_flip_c * flip_mask_f).sum() / flip_den).detach().item()

            # Margin degradation
            gap_c = logits_c.topk(2, dim=-1).values
            gap_n = logits_n.topk(2, dim=-1).values
            margin_delta = (
                    (((gap_n[..., 0] - gap_n[..., 1]) - (gap_c[..., 0] - gap_c[..., 1])) * mask_f).sum() / den
            ).detach().item()

            # Probs / numerics
            lp = completion_tok_log_probs_n
            if lp.numel():
                p = torch.exp(lp)
                median_prob = p.median().item()
                nan_frac = torch.isnan(lp).float().mean().item()
                inf_frac = torch.isinf(lp).float().mean().item()
                abs_lp_mean = lp.abs().mean().item()
            else:
                median_prob = 0.0
                nan_frac = 0.0
                inf_frac = 0.0
                abs_lp_mean = 0.0

            entropy_scalar = float(entropy_n.detach().item() if torch.is_tensor(entropy_n) else entropy_n)

        self._last_loss_stats = {
            "dist/entropy_mean": float(entropy_scalar),
            "dist/median_max_prob": float(median_prob),
            "dist/kl_noisy_vs_clean_mean": float(kl_mean),
            "dist/top2_gap_mean": float(gap_mean),
            "dist/flip_rate": float(flip_rate),
            "dist/flip_clean_prob_mean": float(flip_clean_prob_mean),
            "dist/flip_clean_gap_prob_mean": float(flip_clean_gap_prob_mean),
            "dist/margin_delta": float(margin_delta),
            "numerics/nan_frac": float(nan_frac),
            "numerics/inf_frac": float(inf_frac),
            "observed/abs_logprob_mean": float(abs_lp_mean),
            "observed/abs_noise_mean": float(abs_noise_mean.item()),
        }

    # ---------- inner-loop accumulation helpers ---------- #
    def init_inner_acc(self) -> Dict[str, list]:
        # include container for all loss_stats keys so we can aggregate every metric
        return {
            "total_loss": [],
            "tok_loss": [],
            "base_lr": [],
            "loss_stats": {},  # dict[str, List[float]]
        }

    def accumulate_inner(
        self,
        acc: Dict[str, list],
        *,
        total_loss: torch.Tensor,
        tok_loss: torch.Tensor,
        base_lr: float,
    ) -> None:
        acc["total_loss"].append(float(total_loss.detach().cpu().item()))
        acc["tok_loss"].append(float(tok_loss.detach().cpu().item()))
        acc["base_lr"].append(float(base_lr))
        # pull all last loss stats computed earlier in the step
        for k, v in self._last_loss_stats.items():
            acc["loss_stats"].setdefault(k, []).append(float(v))

    def aggregate_inner(self, acc: Dict[str, list]) -> Dict[str, float]:
        mean = lambda x: float(np.mean(x)) if x else 0.0
        out: Dict[str, float] = {
            "total_loss": mean(acc["total_loss"]),
            "tok_loss": mean(acc["tok_loss"]),
            "base_lr": mean(acc["base_lr"]),
        }
        # flatten aggregated loss_stats back to metric keys used in update_loss_stats
        for k, arr in acc["loss_stats"].items():
            out[k] = mean(arr)
        return out

    # ---------- W&B emitters ---------- #
    def log_inner_update(
        self,
        *,
        train_step: int,
        update_iter: int,
        aggregated_inner: Dict[str, float],
    ) -> int:
        """
        Logs optimizer-step metrics. Returns the step used.
        """
        step_used = self.opt_step
        if self.wandb_enabled:
            payload = {
                "train_step": int(train_step),
                "update_iter": int(update_iter),
                "total_loss": float(aggregated_inner.get("total_loss", 0.0)),
                "tok_loss": float(aggregated_inner.get("tok_loss", 0.0)),
                "base_lr": float(aggregated_inner.get("base_lr", 0.0)),
            }
            # add all aggregated metrics produced by update_loss_stats
            for k, v in aggregated_inner.items():
                if k in ("total_loss", "tok_loss", "base_lr"):
                    continue
                payload[k] = float(v)
            wandb.log(payload, step=step_used)
        self.opt_step += 1
        return step_used

    def log_outer_rewards(
        self,
        *,
        train_step: int,
        rewards_per_func_mean: torch.Tensor,
        func_names: List[str],
        mean_completion_len: float,
    ) -> int:
        """
        Logs reward metrics once per outer train step. Returns the step used.
        """
        step_used = self.opt_step
        if self.wandb_enabled:
            reward_metrics = {
                f"rewards/{func_names[i]}": float(rewards_per_func_mean[i].item())
                for i in range(len(func_names))
            }
            median_seq_reward = (
                float(rewards_per_func_mean.median().item())
                if len(rewards_per_func_mean) > 0
                else 0.0
            )
            payload = {
                "train_step": int(train_step),
                "median_seq_reward": float(median_seq_reward),
                "mean_completion_len": float(mean_completion_len),
            }
            payload.update({k: float(v) for k, v in self._last_prefix_stats.items()})
            payload.update(reward_metrics)
            wandb.log(payload, step=step_used)
        self.train_step_axis += 1
        return step_used

    def log_eval_results(
        self,
        *,
        rewards_per_func_mean: torch.Tensor,
        func_names: List[str],
    ):
        """
        Logs reward metrics once per outer train step. Returns the step used.
        """
        step_used = self.opt_step
        if self.wandb_enabled:
            reward_metrics = {
                f"eval_rewards/{func_names[i]}": float(rewards_per_func_mean[i].item())
                for i in range(len(func_names))
            }
            payload = reward_metrics
            wandb.log(payload, step=step_used)

    def log_grad_metrics(
        self,
        layer: torch.nn.Module,
        *,
        log_name: str = "",
        learning_rate: float = 1.0,
        step: Optional[int] = None,
        total_norm_pre_clip: float = 0,
    ) -> None:
        """
        Compact gradient diagnostics for a module.
        """
        grad_tensors = []
        param_tensors = []
        for _, p in layer.named_parameters():
            if p.grad is not None:
                grad_tensors.append(p.grad.view(-1))
            if p.data is not None:
                param_tensors.append(p.data.view(-1))

        if grad_tensors:
            all_grads = torch.cat(grad_tensors)
            grad_norm = torch.norm(all_grads).item()
            grad_max = all_grads.abs().max().item()
            grad_abs_mean = all_grads.abs().mean().item()
            grad_sparsity = (all_grads == 0).float().mean().item()
        else:
            grad_norm = grad_max = grad_abs_mean = grad_sparsity = 0.0

        if param_tensors:
            all_params = torch.cat(param_tensors)
            param_norm = torch.norm(all_params).item()
        else:
            param_norm = 0.0

        relative_update = (learning_rate * grad_norm / param_norm) if param_norm > 0 else 0.0

        if self.wandb_enabled:
            payload = {
                f"grads/{log_name}_norm": grad_norm,
                f"grads/{log_name}_max": grad_max,
                f"grads/{log_name}_abs_mean": grad_abs_mean,
                f"grads/{log_name}_sparsity": grad_sparsity,
                f"grads/{log_name}_relative_update": relative_update,
                f"grads/total_model_norm_pre_clip": total_norm_pre_clip,
            }
            if step is None:
                wandb.log(payload)
            else:
                wandb.log(payload, step=int(step))


def get_prefix_proportions(sequences: list[str]) -> Union[tuple[list[float], list[Union[int, float]]], list[float]]:
    """
    AI Gen
    For each sequence, efficiently finds the longest shared prefix with another
    sequence in the list, returned as a proportion of its own length.
    """
    n = len(sequences)
    if n < 2:
        return [0.0] * n

    # Sort sequences to place strings with common prefixes next to each other.
    s_seqs = sorted(sequences)

    # Compute the Longest Common Prefix (LCP) length for all adjacent pairs.
    adj_lcps = [len(os.path.commonprefix([s1, s2])) for s1, s2 in zip(s_seqs, s_seqs[1:])]

    # For each string, its max LCP is the larger of the LCPs with its left
    # and right neighbors in the sorted list. Padding with 0 handles endpoints.
    max_lcps = [max(p, n) for p, n in zip([0] + adj_lcps, adj_lcps + [0])]

    # Map each unique string to its calculated maximum LCP. This handles duplicates.
    lcp_map = dict(zip(s_seqs, max_lcps))

    # Using the map, calculate the proportion for each string in the original order.
    # The check `if s else 0.0` prevents division by zero for empty strings.
    return [lcp_map[s] / len(s) if s else 0.0 for s in sequences], [lcp_map[s] if s else 0.0 for s in sequences]
