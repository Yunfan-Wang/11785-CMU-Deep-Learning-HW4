import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement greedy search
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):

            if finished.all():
                break

            logits = self.score_fn(x)
            logits = logits / temperature
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            log_probs = torch.log_softmax(logits, dim=-1)
            next_tokens = torch.argmax(log_probs, dim=-1)  # (B,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            scores = torch.where(finished, scores, scores + token_scores)

            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos
        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # TODO: Implement beam search
        batch_size = x.size(0)
        device = x.device
        eos_id = self.tokenizer.eos_id

        prompt_len = x.size(1)
        debug = False
        debug_batch = 1
        def rank_score(score: torch.Tensor, seq: torch.Tensor) -> float:
            gen_len = max(1, seq.size(0) - prompt_len)
            return (score / gen_len).item()

        # Initial expansion
        logits = self.score_fn(x) / temperature                      # (B, V)
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        log_probs = torch.log_softmax(logits, dim=-1)               # (B, V)

        topk_scores, topk_tokens = torch.topk(log_probs, beam_width, dim=-1)  # (B, K)
        if debug:
            print("\ninitial expansion")
            for b in range(batch_size):
                toks = [self.tokenizer.decode([t.item()]) for t in topk_tokens[b]]
                print(f"batch {b}: topk tokens = {toks}")
                print(f"batch {b}: topk scores = {[round(s.item(), 4) for s in topk_scores[b]]}")
        active_sequences = []
        active_scores = []

        for b in range(batch_size):
            seqs_b = x[b].unsqueeze(0).repeat(beam_width, 1)  # (K, T)
            seqs_b = torch.cat([seqs_b, topk_tokens[b].unsqueeze(-1)], dim=-1)

            active_sequences.append(seqs_b)
            active_scores.append(topk_scores[b])                                          # (B, K)

        completed = [[] for _ in range(batch_size)]

        new_active_sequences = []
        new_active_scores = []
        for b in range(batch_size):
            keep_seq = []
            keep_score = []
            for k in range(beam_width):
                seq = active_sequences[b][k]
                score = active_scores[b][k]
                if seq[-1].item() == eos_id:
                    completed[b].append((score, seq.clone()))
                else:
                    keep_seq.append(seq)
                    keep_score.append(score)

            if len(keep_seq) == 0:
                keep_seq.append(active_sequences[b][0].clone())
                keep_score.append(active_scores[b][0].clone())

            new_active_sequences.append(torch.stack(keep_seq, dim=0))
            new_active_scores.append(torch.stack(keep_score, dim=0))

        active_sequences = new_active_sequences
        active_scores = new_active_scores

        # Iterative decoding
        current_len = x.size(1) + 1
        while current_len < self.max_length:
            all_done = True

            next_active_sequences = []
            next_active_scores = []

            beam_log_probs = []
            for k in range(beam_width):
                beam_x = torch.stack([active_sequences[b][k] for b in range(batch_size)], dim=0)   # (B, T)
                logits_k = self.score_fn(beam_x) / temperature                                      # (B, V)
                logits_k = self._apply_repeat_penalty(logits_k, beam_x, repeat_penalty)
                log_probs_k = torch.log_softmax(logits_k, dim=-1)                                   # (B, V)

                finished_mask_k = torch.tensor(
                    [
                        1 if active_sequences[b][k][-1].item() == eos_id else 0
                        for b in range(batch_size)
                    ],
                    dtype=torch.bool,
                    device=device
                )

                if finished_mask_k.any():
                    log_probs_k = log_probs_k.masked_fill(finished_mask_k.unsqueeze(-1), float("-inf"))
                    log_probs_k[:, eos_id] = torch.where(
                        finished_mask_k,
                        torch.zeros(batch_size, device=device, dtype=log_probs_k.dtype),
                        log_probs_k[:, eos_id]
                    )

                beam_log_probs.append(log_probs_k)

            for b in range(batch_size):
                seqs_b = active_sequences[b]
                scores_b = active_scores[b]

                if debug and b == debug_batch:
                    print(f"\nstep len {current_len}, batch {b} active beams")
                    for kk in range(seqs_b.size(0)):
                        print(
                            f"beam {kk}: seq='{self.tokenizer.decode(seqs_b[kk].tolist(), skip_special_tokens=True)}' "
                            f"score={scores_b[kk].item():.4f}"
                        )

                if seqs_b.numel() == 0:
                    next_active_sequences.append(seqs_b)
                    next_active_scores.append(scores_b)
                    continue

                all_done = False
                candidate_list = []

                for k in range(seqs_b.size(0)):
                    log_probs_k = beam_log_probs[k][b]                    # (V,)
                    beam_candidate_scores = scores_b[k] + log_probs_k    # (V,)

                    top_scores_k, top_tokens_k = torch.topk(
                        beam_candidate_scores,
                        min(beam_width, beam_candidate_scores.size(0)),
                        dim=-1
                    )

                    for j in range(top_scores_k.size(0)):
                        token = top_tokens_k[j]
                        score = top_scores_k[j]
                        cand_seq = torch.cat([seqs_b[k], token.view(1)], dim=0)
                        candidate_list.append((score, cand_seq))

                candidate_list.sort(key=lambda t: rank_score(t[0], t[1]), reverse=True)

                if debug and b == debug_batch:
                    print(f"candidates for batch {b}")
                    for ci, (score, cand_seq) in enumerate(candidate_list[:10]):
                        text = self.tokenizer.decode(cand_seq.tolist(), skip_special_tokens=True)
                        print(
                            f"{ci}: text='{text}' raw={score.item():.4f} rank={rank_score(score, cand_seq):.4f} "
                            f"last={cand_seq[-1].item()} eos={(cand_seq[-1].item() == eos_id)}"
                        )

                keep_seq = []
                keep_score = []
                seen = set()

                for score, cand_seq in candidate_list:
                    if cand_seq[-1].item() == eos_id:
                        key = tuple(cand_seq.tolist())
                        if key in seen:
                            continue
                        seen.add(key)
                        completed[b].append((score, cand_seq.clone()))
                    else:
                        keep_seq.append(cand_seq)
                        keep_score.append(score)

                    if len(keep_seq) >= beam_width:
                        break

                if len(keep_seq) == 0:
                    if len(completed[b]) > 0:
                        completed[b].sort(key=lambda t: rank_score(t[0], t[1]), reverse=True)

                        fallback_seqs = []
                        fallback_scores = []
                        for score, seq in completed[b][:beam_width]:
                            fallback_seqs.append(seq.clone())
                            fallback_scores.append(score.clone())

                        next_active_sequences.append(torch.stack(fallback_seqs, dim=0))
                        next_active_scores.append(torch.stack(fallback_scores, dim=0))
                    else:
                        next_active_sequences.append(seqs_b[:1].clone())
                        next_active_scores.append(scores_b[:1].clone())
                else:
                    while len(keep_seq) < beam_width:
                        keep_seq.append(keep_seq[-1].clone())
                        keep_score.append(keep_score[-1].clone())

                    next_active_sequences.append(torch.stack(keep_seq, dim=0))
                    next_active_scores.append(torch.stack(keep_score, dim=0))

            active_sequences = next_active_sequences
            active_scores = next_active_scores
            current_len += 1

            if all(seq_b.size(0) == 0 for seq_b in active_sequences):
                break


        final_sequences = []
        final_scores = []

        for b in range(batch_size):
            completed[b].sort(key=lambda t: rank_score(t[0], t[1]), reverse=True)

            chosen = []
            chosen_scores = []
            seen = set()

            for score, seq in completed[b]:
                key = tuple(seq.tolist())
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(seq)
                chosen_scores.append(score)
                if len(chosen) == beam_width:
                    break

            if len(chosen) < beam_width and active_sequences[b].size(0) > 0:
                scores_b = active_scores[b]
                seqs_b = active_sequences[b]
                order = sorted(
                    range(seqs_b.size(0)),
                    key=lambda idx: rank_score(scores_b[idx], seqs_b[idx]),
                    reverse=True
                )
                for idx in order:
                    seq = seqs_b[idx]
                    score = scores_b[idx]
                    key = tuple(seq.tolist())
                    if key in seen:
                        continue
                    seen.add(key)
                    chosen.append(seq)
                    chosen_scores.append(score)
                    if len(chosen) == beam_width:
                        break

            if len(chosen) == 0:
                chosen.append(x[b].clone())
                chosen_scores.append(torch.tensor(0.0, device=device))

            while len(chosen) < beam_width:
                chosen.append(chosen[-1].clone())
                chosen_scores.append(chosen_scores[-1].clone())
            if debug and b == debug_batch:
                print(f"\nfinal chosen batch {b}")
                for kk, (seq, sc) in enumerate(zip(chosen, chosen_scores)):
                    print(f"final {kk}: '{self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)}' score={sc.item():.4f}")
            max_len_b = max(seq.size(0) for seq in chosen)
            padded_chosen = []

            for seq in chosen:
                if seq.size(0) < max_len_b:
                    pad = torch.full(
                        (max_len_b - seq.size(0),),
                        self.tokenizer.pad_id,
                        dtype=seq.dtype,
                        device=seq.device
                    )
                    seq = torch.cat([seq, pad], dim=0)
                padded_chosen.append(seq)

            final_sequences.append(torch.stack(padded_chosen, dim=0))
            final_scores.append(torch.stack(chosen_scores, dim=0))

        max_final_len = max(seq_group.size(1) for seq_group in final_sequences)

        padded_final_sequences = []
        for seq_group in final_sequences:
            if seq_group.size(1) < max_final_len:
                pad = torch.full(
                    (seq_group.size(0), max_final_len - seq_group.size(1)),
                    self.tokenizer.pad_id,
                    dtype=seq_group.dtype,
                    device=seq_group.device
                )
                seq_group = torch.cat([seq_group, pad], dim=1)
            padded_final_sequences.append(seq_group)

        sequences = torch.stack(padded_final_sequences, dim=0)   # (B, K, T_max)
        scores = torch.stack(final_scores, dim=0)                # (B, K)

        return sequences, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]