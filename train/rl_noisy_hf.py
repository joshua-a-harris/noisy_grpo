import pandas as pd
import wandb
import train.rl_logging as rl_logging
from models.smollm_2.LlamaNoise import LlamaNoise
from train.rl_logging import RLLogger
import argparse
import os
import pathlib
from functools import partial
import torch
from torch.utils.data import DistributedSampler, DataLoader
from transformers import AutoTokenizer, get_scheduler
from data.data_utils import load_template, get_gsm8k_questions, get_reasoning_gym_questions
from data.dataset import RLDataset, padded_collate_rl
from models.qwen_2.QwenNoise import QwenNoise
from train.reward_funcs import REWARD_FUNCTIONS
from train.utils import load_config, set_seed, \
    expand_inputs_for_generation, get_decay_parameter_names
import torch.nn.functional as F



class NoisyGRPO:
    """ Class for implementing GRPO + Noise rewards and loss"""
    def __init__(self,
                 policy,
                 tokenizer,
                 opt,
                 device,
                 reward_funcs=None,
                 mem_efficient_logits=False,
                 epsilon=0.2,
                 logger: RLLogger | None = None,
                 ):
        """
        Initialize algorithm
        :param policy: Transformers model
        :param tokenizer: Transformer tokenizer
        :param opt: Torch optimizer
        :param device: Torch training device
        :param reward_funcs: Reward functions
        :param mem_efficient_logits: Whether to use memory efficient logits
        :param epsilon: Ratio clip epsilon
        :param logger: RL logger
        """
        self.epsilon = epsilon
        self.policy = policy
        self.p_tokenizer = tokenizer
        self.reward_funcs = reward_funcs if reward_funcs else []
        self.device = device
        self.opt = opt
        self.mem_efficient_logits = mem_efficient_logits
        self.logger = logger

    def get_reward_from_funcs(self, prompts, completions, batch):
        """
        Returns rewards in shape [B, r_funcs]
        """
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=self.device)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_kwargs = {key.replace('oi_', ''): value.tolist() if isinstance(value, torch.Tensor) else value for key, value in batch.items() if
                             key.split('_')[0] == 'oi'}
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.device)
        return rewards_per_func

    def get_rewards(self, batch, full_seq,
                    full_labels, num_repeats_per_seq):
        """GRPO advantage calculation - Modified from TRL"""
        # Returns rewards per func (B, n_funcs)
        rewards_per_verifier, completions = self.collect_rewards(batch, full_labels, full_seq)
        # update logger prefix stats
        if self.logger is not None:
            self.logger.store_prefix_stats(completions)

        # Aggregates across funcs (B, 1)
        rewards = rewards_per_verifier.sum(dim=1)

        # Compute grouped-wise rewards (B, 1) -> (B/r, r) -> (B/r, 1)
        mean_grouped_rewards = rewards.view(-1, num_repeats_per_seq).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, num_repeats_per_seq).std(dim=1)

        # Normalize to compute advantages (B)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_repeats_per_seq, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_repeats_per_seq, dim=0)
        rewards = (rewards - mean_grouped_rewards) / std_grouped_rewards.clamp_min(0.05)
        rewards = rewards.unsqueeze(1)
        # Repeat scaler rewards for full sequence (B, 1) -> (B, seq)
        full_seq_rewards = rewards.repeat(1, full_seq.shape[-1])
        # Mask rewards where attention masked
        tok_mask = (full_labels != -100)
        full_seq_rewards = full_seq_rewards * tok_mask
        mean_completion_len = tok_mask.sum(dim=1).cpu().type(torch.float32).mean().item()
        return rewards_per_verifier, full_seq_rewards, completions, mean_completion_len

    def get_loss(self, prompt_len, full_seq_rewards, full_seq,
                 full_labels, full_attn_mask, full_position_ids, old_log_probs, full_noise, temperature):
        """ GRPO loss calculation with trajectory random noise """
        def grpo_loss(log_probs, old_log_probs, mask, rets):
            if old_log_probs is None:
                old_log_probs = log_probs.detach()
            per_token_ratio = torch.exp(log_probs - old_log_probs)
            per_token_loss = -torch.min(per_token_ratio * rets, torch.clamp(per_token_ratio,
                                                                            1-self.epsilon, 1+self.epsilon) * rets)
            masked_loss = per_token_loss * mask
            loss_per_sequence = (
                    masked_loss.sum(dim=1) / (mask.sum(dim=1).clamp(min=1).float())
            )
            loss = loss_per_sequence.mean()
            return loss
        # Set noise used in rollout and forward
        self.policy.set_current_noise(full_noise)
        tok_log_probs, llm_outputs = self.get_full_log_probs(full_attn_mask, full_labels, full_position_ids, full_seq,
                                                             prompt_len, temperature)
        # Align rewards and mask
        tok_rets = full_seq_rewards[:, 1:]
        tok_mask = (full_labels[:, 1:] != -100)
        tok_loss = grpo_loss(tok_log_probs, old_log_probs, tok_mask, tok_rets)
        # Entropy calc
        logits_for_entropy = llm_outputs.logits[:, :-1, :] / temperature  # [B, T-1, V]
        ent_per_tok = torch.distributions.Categorical(logits=logits_for_entropy).entropy()  # [B, T-1]
        if self.mem_efficient_logits:
            # when you used num_logits_to_keep, the first usable position is prompt_len-1
            offset = prompt_len - 1
            logit_mask = tok_mask[:, offset:]
        else:
            logit_mask = tok_mask
        ent_term = (ent_per_tok * logit_mask).sum() / logit_mask.sum().clamp_min(1)
        # Final loss
        total_loss = tok_loss
        # Log metrics for noise vs clean policy
        if self.logger is not None:
            with torch.no_grad():
                # Clean logits
                self.policy.set_current_noise(torch.zeros_like(full_noise))
                tok_log_probs_no_noise, llm_outputs_no_noise  = self.get_full_log_probs(full_attn_mask, full_labels, full_position_ids, full_seq,
                                                        prompt_len,temperature)
                logits_n = logits_for_entropy
                logits_c = llm_outputs_no_noise.logits[:, :-1, :] / temperature
                # update logging state only
                self.logger.update_loss_stats(logits_n,
                                              logits_c,
                                              logit_mask,
                                              tok_log_probs[tok_mask.bool()],
                                              ent_term,
                                              torch.abs(full_noise[:, prompt_len:, :]).mean())
                del llm_outputs_no_noise
        return total_loss, tok_loss, tok_log_probs.detach(), ent_term.detach()

    def get_full_log_probs(self, full_attn_mask, full_labels, full_position_ids, full_seq,
                           prompt_len, temperature):
        if self.mem_efficient_logits:
            # TODO - **CURRENTLY ONLY WORKS FOR BATCH SIZE 1 (REPEATED N TIMES)**
            max_completion_len = full_labels[:, prompt_len:].shape[-1] + 1
            llm_outputs = self.policy(input_ids=full_seq,
                                      attention_mask=full_attn_mask,
                                      position_ids=full_position_ids,
                                      use_cache=False)
            llm_outputs.logits = llm_outputs.logits[:, -max_completion_len:, :]
            start_idx = prompt_len - 1
            tok_log_probs = self.get_log_probs(full_seq[:, start_idx:], llm_outputs, temperature)
            tok_log_probs = F.pad(tok_log_probs, pad=(start_idx, 0), mode='constant', value=0)
        else:
            llm_outputs = self.policy(input_ids=full_seq,
                                      attention_mask=full_attn_mask,
                                      position_ids=full_position_ids,
                                      use_cache=False)
            tok_log_probs = self.get_log_probs(full_seq, llm_outputs, temperature)
        return tok_log_probs, llm_outputs

    def get_log_probs(self, full_seq, llm_outputs, temperature):
        """Calculate sampled log probs from logits"""
        logits = llm_outputs.logits[:, :-1, :].float() / temperature  # force fp32
        labels = full_seq[:, 1:]
        tok_log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return tok_log_probs

    def collect_rewards(self, batch, full_labels, full_seq):
        """Calculate all reward funcs for sequences"""
        rewards_per_func = None
        completions = self.p_tokenizer.batch_decode(
            [full_seq[idx][(x != -100)] for idx, x in enumerate(full_labels)], skip_special_tokens=True)
        if self.reward_funcs:
            prompts = self.p_tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            rewards_per_func = self.get_reward_from_funcs(prompts, completions, batch)
        return rewards_per_func, completions


class NoisyLLMEnv:
    """RL Env for rollouts and updates with noise injection"""
    def __init__(self,
                 agent,
                 dl,
                 device,
                 eval_dl=None,
                 wnb=False,
                 run_name='test',
                 scheduler=None,
                 num_repeats_per_seq=1,
                 num_grad_accum_steps=1,
                 checkpoint_freq=None,
                 num_update_iters=1,
                 use_const_noise=False,
                 logger: RLLogger | None = None):
        self.num_update_iters = num_update_iters
        self.use_const_noise = use_const_noise
        self.agent = agent
        self.dl = dl
        self.iter_dl = iter(dl)
        self.eval_dl = eval_dl
        self.run_name = run_name
        self.wnb = wnb
        self.device = device
        output_path = os.getenv('OUTPUT_PATH', 'outputs')
        self.output_dir = pathlib.Path(f'{output_path}/{self.run_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.output_dir.resolve()
        self.scheduler = scheduler
        self.num_grad_accum_steps = num_grad_accum_steps
        self.checkpoint_freq = checkpoint_freq
        self.num_repeats_per_seq = num_repeats_per_seq
        self.data_buffer = None
        self.buffer_idx = None
        self.logger = logger

    # -------------------------------------------------------------

    def batch_data_init(self, batch, max_gen_toks, num_repeats_per_seq):
        if num_repeats_per_seq > 1:
            input_ids, batch = expand_inputs_for_generation(
                input_ids=batch['input_ids'],
                expand_size=num_repeats_per_seq,
                **{key: value for key, value in batch.items() if key != 'input_ids'},
            )
            batch = {key: [x for x in value for _ in range(num_repeats_per_seq)] if isinstance(value, list) else value for key, value in batch.items()}
            batch['input_ids'] = input_ids
        prompt_len = batch['input_ids'].shape[1]
        batch_shape = (batch['input_ids'].shape[0], max_gen_toks + prompt_len)
        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        position_ids = (batch['attention_mask'].cumsum(1) - batch['attention_mask'].long()).to(self.device)
        # Initialise batch data buffer
        self.data_buffer = dict(
            full_attn_mask = torch.zeros(batch_shape).type(batch['attention_mask'].dtype).to(self.device),
            full_seq = torch.zeros(batch_shape).type(batch['input_ids'].dtype).to(self.device),
            full_labels = torch.full(batch_shape, fill_value=-100).type(batch['labels'].dtype).to(self.device),
            full_position_ids = torch.zeros(batch_shape).type(position_ids.dtype).to(self.device),
            full_noise = torch.zeros((*batch_shape, self.agent.policy.config.hidden_size)).to(self.device),
        )
        # Set initial values from batch inputs
        self.data_buffer['full_attn_mask'][:, :prompt_len].copy_(batch['attention_mask'])
        self.data_buffer['full_seq'][:, :prompt_len].copy_(batch['input_ids'])
        self.data_buffer['full_labels'][:, :prompt_len].copy_(batch['labels'].clone())
        self.data_buffer['full_position_ids'][:, :prompt_len].copy_(position_ids)
        batch['position_ids'] = position_ids
        self.buffer_idx = int(prompt_len)
        return batch

    def get_buffer_data(self, key):
        """Retrieve data buffer key values up to longest sequence len"""
        return self.data_buffer[key][:, :self.buffer_idx]

    def train(self, training_steps, max_gen_toks, tok_sampling=True, generation_config=None):
        self.agent.policy.train()
        if generation_config is None:
            generation_config = {}
        if "temperature" not in generation_config.keys():
            generation_config = {**generation_config, "temperature": 1.0}
        for train_step in range(training_steps):
            print(f'Running generation for train step {train_step}....')
            print(f'Current generation config: {generation_config}')
            with torch.no_grad():
                # Get batch of prompts - Env reset
                try:
                    batch = next(self.iter_dl)
                except StopIteration:
                    self.iter_dl = iter(self.dl)
                    batch = next(self.iter_dl)
                # Set constant noise for each trajectory if using const nose
                if self.use_const_noise:
                    const_noise = torch.randn(size=(batch['attention_mask'].shape[0] * self.num_repeats_per_seq, 1,
                                                    self.agent.policy.config.hidden_size)).to(self.device)
                else:
                    const_noise = None
                # Initialise batch and data buffers
                batch = self.batch_data_init(batch, max_gen_toks, self.num_repeats_per_seq)
                # Rollout trajectories
                self.rl_generation(batch, max_gen_toks,
                                                  const_noise=const_noise,
                                   tok_sampling=tok_sampling, **generation_config)
                # Collect rewards
                rewards_per_verifier, full_seq_rewards, completions, mean_completion_len = \
                    self.agent.get_rewards(
                        batch,
                        self.get_buffer_data('full_seq'),
                        self.get_buffer_data('full_labels'),
                        self.num_repeats_per_seq
                    )
            total_entropy = 0
            batch_size = batch['input_ids'].shape[0]
            num_per_backward_batch =  batch_size // self.num_grad_accum_steps
            old_log_p = {x: None for x in range(self.num_grad_accum_steps)}
            for update_iter in range(self.num_update_iters):
                inner_acc = self.logger.init_inner_acc() if self.logger is not None else None
                self.agent.opt.zero_grad()
                start_idx = 0
                for backward_idx in range(self.num_grad_accum_steps):
                    end_idx = min(start_idx + num_per_backward_batch, batch_size)
                    sub_seq = self.get_buffer_data('full_seq')[start_idx:end_idx, ...]
                    sub_labels = self.get_buffer_data('full_labels')[start_idx:end_idx, ...]
                    sub_attn_mask = self.get_buffer_data('full_attn_mask')[start_idx:end_idx, ...]
                    sub_position_ids = self.get_buffer_data('full_position_ids')[start_idx:end_idx, ...]
                    sub_full_noise = self.get_buffer_data('full_noise')[start_idx:end_idx, ...]
                    sub_full_seq_rewards = full_seq_rewards[start_idx:end_idx, ...]
                    sub_prompt_len = batch['input_ids'].shape[1]
                    print(f'Running backward for iter step: back step {update_iter}:{backward_idx}....')
                    total_loss, tok_loss, tok_log_probs, entropy = \
                            (self.agent.
                            get_loss(
                                sub_prompt_len,
                                sub_full_seq_rewards,
                                sub_seq,
                                sub_labels,
                                sub_attn_mask,
                                sub_position_ids,
                                old_log_p[backward_idx],
                                sub_full_noise,
                                generation_config["temperature"],
                            ))
                    total_entropy += entropy
                    if old_log_p[backward_idx] is None:
                        old_log_p[backward_idx] = tok_log_probs

                    # grad accumulation
                    total_loss = total_loss / self.num_grad_accum_steps
                    total_loss.backward()
                    start_idx = end_idx

                    # accumulate inner-loop metrics via logger
                    if self.logger is not None and inner_acc is not None:
                        self.logger.accumulate_inner(
                            inner_acc,
                            total_loss=total_loss,
                            tok_loss=tok_loss,
                            base_lr=self.agent.opt.param_groups[0]['lr'],
                        )

                # step optimizer once per update_iter
                total_norm = torch.nn.utils.clip_grad_norm_(self.agent.policy.model.parameters(), 0.5)
                print(f'Param update train step: iter - {train_step}: {update_iter}')
                self.agent.opt.step()
                # aggregate and log inner metrics once per update_iter
                if self.logger is not None and inner_acc is not None:
                    aggregated_inner = self.logger.aggregate_inner(inner_acc)
                    step_used = self.logger.log_inner_update(
                        train_step=train_step,
                        update_iter=update_iter,
                        aggregated_inner=aggregated_inner,
                    )
                    # also log gradient diagnostics at the same step
                    self.logger.log_grad_metrics(
                        self.agent.policy.model.layers[-1],
                        log_name='final_hid',
                        learning_rate=self.agent.opt.param_groups[0]['lr'],
                        step=step_used,
                        total_norm_pre_clip=total_norm,
                    )
                # Step learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
            avg_entropy = total_entropy / (self.num_update_iters * self.num_grad_accum_steps)
            print(f'Avg entropy: {avg_entropy}')
            # ---------- log outer-loop reward metrics ONCE per train step ----------
            if self.logger is not None and self.wnb:
                rewards_mean_over_batch = rewards_per_verifier.detach().cpu().mean(0)
                func_names = [func.__name__ for func in self.agent.reward_funcs]
                self.logger.log_outer_rewards(
                    train_step=train_step,
                    rewards_per_func_mean=rewards_mean_over_batch,
                    func_names=func_names,
                    mean_completion_len=float(mean_completion_len),
                )
            print('Train Step: ', train_step,
                  '| Mean Seq Reward: ', rewards_per_verifier.detach().cpu().mean(0).sum(),
                  )
            # Save checkpoint and run evaluation
            if self.checkpoint_freq and train_step % self.checkpoint_freq == 0:
                print('Saving checkpoint...')
                self.save_checkpoint(train_step)
                if self.eval_dl is not None:
                    print('Running evaluation...')
                    eval_rewards, completions = self.eval(max_gen_toks)
                    self.logger.log_eval_results(rewards_per_func_mean=eval_rewards.detach().cpu().mean(0),
                        func_names=[func.__name__ for func in self.agent.reward_funcs])
            # Log full output sequences generated to disk.
            log_seqs = self.agent.p_tokenizer.batch_decode(self.get_buffer_data('full_seq'), skip_special_tokens=False)
            with open(os.path.join(self.output_dir, f'{self.run_name}_inf.txt'),
                      'a', encoding='utf-8') as file:
                file.write("\n***********************\n" + str(train_step) + "\n***********************\n")
                for line in log_seqs:
                    file.write(line + "\n-------------------------\n")
                for line in completions:
                    file.write(line + "\n-------------------------\n")
        # Checkpoint final model
        self.save_checkpoint('final_step')

    def save_checkpoint(self, train_step):
        """Helper for saving model and optimizer state checkpoints"""
        check_dir = os.path.join(self.output_dir, f'checkpoint_{train_step}')
        self.agent.policy.save_pretrained(check_dir)
        torch.save(self.agent.opt.state_dict(), f"{check_dir}/optimizer.pt")

    def eval(self, max_gen_toks, tok_sampling=False, generation_config=None):
        self.agent.policy.eval()
        if generation_config is None:
            generation_config = {}
        all_completions = []
        all_rewards = []
        eval_iter_dl = iter(self.eval_dl)
        print('Eval generation config:', generation_config)
        print('Eval tok sampling:', tok_sampling)
        for batch in eval_iter_dl:
            batch = self.batch_data_init(batch, max_gen_toks, 1)
            const_noise = torch.zeros(size=(batch['attention_mask'].shape[0], 1,
                                                self.agent.policy.config.hidden_size)).to(self.device)

            with torch.no_grad():
                self.rl_generation(batch, max_gen_toks, const_noise=const_noise, tok_sampling=tok_sampling, **generation_config)
            rewards_per_verifier, completions = self.agent.collect_rewards(batch,
                                                                           self.get_buffer_data('full_labels'),
                                                                           self.get_buffer_data('full_seq'))
            all_completions.extend(completions)
            all_rewards.append(rewards_per_verifier)
        all_rewards = torch.cat(all_rewards)
        print(f'Average Reward: {all_rewards.mean(0)}')
        df = pd.DataFrame(all_rewards.cpu().numpy())
        df.to_csv(os.path.join(self.output_dir, f'rewards.csv'), index=False, header=False)
        log_seqs = self.agent.p_tokenizer.batch_decode(self.get_buffer_data('full_seq'), skip_special_tokens=False)
        with open(os.path.join(self.output_dir, f'{self.run_name}_eval.txt'),
                  'w', encoding='utf-8') as file:
            for line in log_seqs:
                file.write(line + "\n-------------------------\n")
            for line in all_completions:
                file.write(line + "\n-------------------------\n")
        self.agent.policy.train()
        return all_rewards, all_completions

    def sample_next_inputs(self, llm_outputs, const_noise=None, tok_sampling=True, top_p=None, top_k=None, temperature=1.0, noise_scaler=1.0):
        if tok_sampling:
            token_logits = llm_outputs.logits[:, -1, :] / temperature
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for i in range(token_logits.size(0)):
                    token_logits[i, sorted_indices[i, sorted_indices_to_remove[i]]] = float('-inf')
                next_tok_ids = torch.distributions.Categorical(logits=token_logits).sample()
            elif top_k is not None and top_k > 0:
                topk_token_logits, topk_token_indices = torch.topk(token_logits, top_k, dim=-1)
                token_dist = torch.distributions.Categorical(logits=topk_token_logits)
                token_sample = token_dist.sample().unsqueeze(-1)
                next_tok_ids = topk_token_indices.gather(dim=-1, index=token_sample).squeeze(-1)
            else:
                next_tok_ids = torch.distributions.Categorical(logits=token_logits).sample()
        else:
            next_tok_ids = torch.nn.functional.softmax(llm_outputs.logits[:, -1, :], dim=-1).argmax(dim=-1)
        next_labels = next_tok_ids.clone()
        next_tok_ids = next_tok_ids.unsqueeze(-1)
        next_labels = next_labels.unsqueeze(-1)
        if const_noise is not None:
            next_noise = const_noise * noise_scaler
        else:
            next_noise = torch.randn(size=(next_tok_ids.shape[0], next_tok_ids.shape[1],
                                           self.agent.policy.config.hidden_size)).to(
                self.device) * noise_scaler
        next_attn = torch.ones_like(next_tok_ids)
        return next_tok_ids, next_attn, next_labels, next_noise

    def rl_generation(self, batch, max_gen_tokens, const_noise=None, tok_sampling=True, **kwargs):
        batch_shape = batch['input_ids'].shape
        # Tracks sequences that have ended
        continue_gen = torch.ones((batch_shape[0], 1)).type(torch.bool)
        for step in range(max_gen_tokens):
            if step == 0:
                # Set noise for first generation step (default all zeros)
                self.agent.policy.set_current_noise(self.get_buffer_data('full_noise'))
                # Forward batch
                llm_outputs = self.agent.policy(**{x:y for x, y in batch.items() if x != 'labels' and x.split('_')[0] not in ['completion', 'oi']}, use_cache=True)
                # Sample next inputs
                next_tok_ids, next_attn, next_labels, next_noise = self.sample_next_inputs(llm_outputs,
                                                                                           const_noise=const_noise, tok_sampling=tok_sampling, **kwargs)
                # Update for next step
                store_attn = next_attn
                position_ids = self.data_buffer['full_position_ids'][:, self.buffer_idx-1: self.buffer_idx] + 1
                continue_gen[next_tok_ids[:, -1:] == self.agent.p_tokenizer.eos_token_id] = False
                # Break if all sequences have ended
                if (~continue_gen).all():
                    break
            else:
                # Set sampled noise for generation step
                self.agent.policy.set_current_noise(next_noise)
                # Forward model
                llm_outputs = self.agent.policy(input_ids=next_tok_ids,
                                                attention_mask=torch.concat([self.get_buffer_data('full_attn_mask'), next_attn], dim=-1),
                                                past_key_values=llm_outputs.past_key_values,
                                                position_ids=position_ids,
                                                use_cache=True
                                                )
                # Retrieve final noise applied (post scaling by layer magnitude)
                next_noise = self.agent.policy.get_current_noise()
                # Update buffer with previous inputs
                self.update_buffer(store_attn, next_labels, next_tok_ids, position_ids, next_noise)
                # Sample next inputs
                next_tok_ids, next_attn, next_labels, next_noise = self.sample_next_inputs(
                    llm_outputs,
                    const_noise=const_noise, tok_sampling=tok_sampling, **kwargs)
                # Update for next step
                position_ids = self.data_buffer['full_position_ids'][:, self.buffer_idx-1: self.buffer_idx] + 1
                next_tok_ids[~continue_gen] = self.agent.p_tokenizer.eos_token_id
                next_labels[~continue_gen] = -100
                store_attn = next_attn.clone()
                store_attn[~continue_gen] = 0
                continue_gen[next_tok_ids[:, -1:] == self.agent.p_tokenizer.eos_token_id] = False
                # Break if all sequences have ended
                if (~continue_gen).all():
                    break
        # Update buffer with final outputs
        self.update_buffer(store_attn, next_labels, next_tok_ids, position_ids, torch.zeros_like(next_noise))

    def update_buffer(self, next_attn, next_labels, next_tok_ids, position_ids, next_noise):
        # Update buffer store
        self.data_buffer['full_seq'][:, self.buffer_idx] = next_tok_ids.squeeze(-1)
        self.data_buffer['full_labels'][:, self.buffer_idx] = next_labels.squeeze(-1)
        self.data_buffer['full_position_ids'][:, self.buffer_idx] = position_ids.squeeze(-1)
        self.data_buffer['full_attn_mask'][:, self.buffer_idx] = next_attn.squeeze(-1)
        self.data_buffer['full_noise'][:, self.buffer_idx, :] = next_noise.squeeze(1)
        self.buffer_idx += 1


def setup_scheduler(config, opt):
    if config["train"]["lr_scheduler"] is not None:
        total_steps = config["train"]["num_train_steps"]
        warmup_steps = int(total_steps * 0.005)
        scheduler = get_scheduler(
            name=config["train"]["lr_scheduler"],
            optimizer=opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return scheduler
    else:
        return None


def setup_data(config, p_tokenizer, seed):
    if config["system_prompt_path"]:
        sys_prompt = load_template(config["system_prompt_path"])
    else:
        sys_prompt = None
    if config["dataset"] == 'gsm8k':
        max_samples = config.get('num_samples', None)
        data = get_gsm8k_questions(sys_prompt=sys_prompt, seed=seed, split=config['split'],
                                   max_samples=max_samples)
    elif config["dataset"] == 'reasoning-gym':
        max_samples = config.get('num_samples', 100)
        dataset_config = config['dataset_config']
        gym_task = dataset_config.pop('task_name')
        data = get_reasoning_gym_questions(gym_task=gym_task, gym_config=dataset_config, sys_prompt=sys_prompt, seed=seed, split=config['split'],
                                   max_samples=max_samples)
    else:
        raise ValueError('Invalid Dataset')
    scale_batch = config.get('num_grad_accum_steps', 1)
    inf_batch = config["batch_size"] * scale_batch
    ds = RLDataset(p_tokenizer, data=data, chat_template=True)
    sampler = DistributedSampler(
        ds,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=seed,
    )
    dl = DataLoader(
        dataset=ds,
        batch_size=inf_batch,
        sampler=sampler,
        collate_fn=partial(
            padded_collate_rl,
            padding_idx=p_tokenizer.pad_token_id,
            ignore_idx=-100,
        )
    )
    return dl


def setup_optimizer(config, model):
    base_lr = float(config["train"]["base_lr"])
    param_sections = [model.model]
    section_lrs = [base_lr]
    weight_decays = [config["train"]["weight_decay"]]
    decay_param_names = [get_decay_parameter_names(x) for x in param_sections]
    opt_groups = [opt_group for idx, param_section in enumerate(param_sections) for opt_group in [{
        "params": [
            p for n, p in param_section.named_parameters() if (n in decay_param_names[idx] and p.requires_grad)
        ],
        "weight_decay": weight_decays[idx],
        "lr": section_lrs[idx]
    },
        {
            "params": [
                p for n, p in param_section.named_parameters() if (n not in decay_param_names[idx] and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": section_lrs[idx]
        }]]
    opt = torch.optim.AdamW(opt_groups, betas=(0.9, 0.99))
    if config["model"]["opt_checkpoint_path"]:
        optimizer_path = os.path.join(config["model"]["opt_checkpoint_path"], "optimizer.pt")
        optimizer_state = torch.load(optimizer_path)
        opt.load_state_dict(optimizer_state)
        print('COMPLETE: loaded opt state from checkpoint')
    return opt


def load_model(config, device, hf_token):
    from_checkpoint = config["model"]["checkpoint_path"] is not None
    model_path = config["model"]["checkpoint_path"] if from_checkpoint else config["model"]["model_name"]
    tokenizer_path = config["model"]["tokenizer_path"] if "tokenizer_path" in config["model"].keys() else model_path
    if config["model"]["model_name"] in ['Qwen/Qwen2.5-0.5B',
                                         'meta-llama/Llama-3.2-1B',
                                         'Qwen/Qwen2.5-0.5B-Instruct',
                                         'HuggingFaceTB/SmolLM2-360M',
                                         'meta-llama/Llama-3.2-1B-Instruct']:
        p_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token)
        if config["model"]["model_name"] in ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct']:
            model = QwenNoise.from_pretrained(model_path,
                                              token=hf_token,
                                              torch_dtype=torch.bfloat16
                                              ).to(device)
            p_tokenizer.pad_token_id = p_tokenizer.eos_token_id
        elif config["model"]["model_name"] in ['HuggingFaceTB/SmolLM2-360M', 'meta-llama/Llama-3.2-1B', 'meta-llama/Llama-3.2-1B-Instruct']:
            model = LlamaNoise.from_pretrained(model_path,
                                              token=hf_token,
                                              torch_dtype=torch.bfloat16).to(device)
            p_tokenizer.pad_token_id = p_tokenizer.eos_token_id
        else:
            raise ValueError('Unsupported Noise Model')
        model.set_layer_noise(config['model']['noise_layer'], config['model']['noise_scaling_factor'])
    else:
        raise ValueError('Unsupported Model')
    if config["model"]["chat_template_path"]:
        chat_template = load_template(config["model"]["chat_template_path"])
        p_tokenizer.chat_template = chat_template
    return model, p_tokenizer


def train(config, run_name):
    device = torch.device(config["device"])
    if config["seed"]:
        set_seed(config["seed"])

    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set in the environment!")
    if config["wnb"]["log"] and not wandb_api_key:
        raise ValueError("WANDB_API_KEY is not set in the environment!")
    model, p_tokenizer = load_model(config, device, hf_token)
    dl = setup_data(config["train"], p_tokenizer, seed=config["seed"])
    eval_dl = setup_data(config["eval"], p_tokenizer, seed=config["seed"]+1)

    opt = setup_optimizer(config, model)
    scheduler = setup_scheduler(config, opt)

    if config["wnb"]["log"]:
        wandb.login(key=wandb_api_key)
        wandb.init(project=config["wnb"]["project_name"],
                   config=config,
                   name=run_name)

    # centralized logger
    logger = RLLogger(wandb_enabled=config["wnb"]["log"])

    agent = NoisyGRPO(policy=model,
                      tokenizer=p_tokenizer,
                      opt=opt,
                      device=device,
                      reward_funcs=[REWARD_FUNCTIONS[x] for x in config["reward_funcs"]],
                      mem_efficient_logits=config["train"]["mem_efficient_logits"],
                      epsilon=config["train"]["epsilon"],
                      logger=logger,
                      )
    env = NoisyLLMEnv(agent=agent,
                      dl=dl,
                      eval_dl=eval_dl,
                      device=device,
                      wnb=config["wnb"]["log"],
                      run_name=run_name,
                      num_repeats_per_seq=config["train"]["num_repeats_per_seq"],
                      num_grad_accum_steps=config["train"]["num_grad_accum_steps"],
                      scheduler=scheduler,
                      checkpoint_freq=config["checkpoint_freq"],
                      use_const_noise=config["train"]["use_const_noise"],
                      num_update_iters=config["train"].get("num_update_iters", 1),
                      logger=logger,
                      )
    generation_config = dict(top_k=config["train"]["top_k"],
                      top_p=config["train"]["top_p"],
                      temperature=config["train"]["temperature"],
                             noise_scaler=1.0)
    env.train(training_steps=config["train"]["num_train_steps"],
              max_gen_toks=config["train"]["max_gen_toks"],
              tok_sampling=config["train"]["tok_sampling"],
              generation_config=generation_config)


def evaluate(config, run_name):
    device = torch.device(config["device"])
    if config["seed"]:
        set_seed(config["seed"])

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set in the environment!")

    model, p_tokenizer = load_model(config, device, hf_token)
    eval_dl = setup_data(config["eval"], p_tokenizer, seed=config["seed"])

    # disabled logger for eval path
    logger = RLLogger(wandb_enabled=False)

    agent = NoisyGRPO(policy=model,
                      tokenizer=p_tokenizer,
                      opt=None,
                      device=device,
                      reward_funcs=[REWARD_FUNCTIONS[x] for x in config["reward_funcs"]],
                      mem_efficient_logits=False,
                      logger=logger,
                      )
    env = NoisyLLMEnv(agent=agent,
                      dl=eval_dl,
                      eval_dl=eval_dl,
                      device=device,
                      wnb=False,
                      run_name=run_name,
                      num_repeats_per_seq=config["eval"]["num_repeats_per_seq"],
                      num_grad_accum_steps=1,
                      scheduler=None,
                      checkpoint_freq=None,
                      logger=logger,
                      )
    generation_config = dict(top_k=config["eval"]["top_k"],
                      top_p=config["eval"]["top_p"],
                      temperature=config["eval"]["temperature"],
                             noise_scaler=1.0)
    env.eval(max_gen_toks=config["eval"]["max_gen_toks"],
             tok_sampling=config["eval"]["tok_sampling"],
             generation_config=generation_config)


