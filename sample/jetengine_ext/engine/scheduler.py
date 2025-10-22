from collections import deque
import torch
from torch.nn import functional as F
import numpy as np

from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine_ext.engine.block_manager import BlockManager
from jetengine_ext.layers.sampler import sample_with_temperature_topk_topp
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK, Sample


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.running: list[Sequence] = []
        self.sample_pipe = LogitsPipe([
                                Temperature(),      # Scale logits by temperature
                                TopK(),             # Apply top-k filtering
                                Softmax(),          # Convert logits to probabilities
                                TopP(),             # Apply top-p filtering
                            ])
        self.sample_pipe_topk0 = LogitsPipe([
                        Temperature(),      # Scale logits by temperature
                        Softmax(),          # Convert logits to probabilities
                        TopP(),             # Apply top-p filtering
                        ])

    def add(self, seq: Sequence):
        self.running.append(seq)

    def is_finished(self):
        return not self.running

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # 1. Schedule new sequences for prefill
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for seq in prefill_candidates:
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL   
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
        if denoise_candidates:
            denoise_batch = []
            for seq in denoise_candidates:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
            if denoise_batch:
                return denoise_batch, RunType.DENOISE

        return None, None     

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        
        elif run_type == RunType.DENOISE:
            start_idx = 0
            if self.consistent_sampling_params:
                if seqs[0].top_k > 0:
                    probs = self.sample_pipe(logits, temperature=seqs[0].temperature, top_k=seqs[0].top_k, top_p=seqs[0].top_p) 
                else:
                    probs = self.sample_pipe_topk0(logits, temperature=seqs[0].temperature, top_p=seqs[0].top_p)
            for seq in seqs:
                # Extract the part of the tensors relevant to this sequence
                if seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    if not self.consistent_sampling_params:
                        if seq.top_k > 0:
                            probs = self.sample_pipe(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_k=seq.top_k, top_p=seq.top_p)
                        else:
                            probs = self.sample_pipe_topk0(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_p=seq.top_p)
                        seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1) 
                        seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)    
                    else:
                        seq_x0 = torch.multinomial(probs[start_idx : start_idx + block_len], num_samples=1).squeeze(-1) 
                        seq_x0_p = torch.gather(probs[start_idx : start_idx + block_len], -1, seq_x0.unsqueeze(-1)).squeeze(-1)

                    # ========== PPL EVALUATION MODE ==========
                    if seq.eval_mode and seq.full_oracle_sentence is not None:
                        # Get oracle for CURRENT block
                        oracle_block = seq.get_current_block_oracle()
                        if oracle_block is None:
                            # Beyond sentence length, treat as normal generation
                            seq.eval_mode = False
                        else:
                            oracle_block_tensor = torch.tensor(oracle_block, device=logits.device)
                            
                            # Get probabilities for oracle tokens
                            oracle_probs = torch.gather(
                                probs[start_idx : start_idx + block_len], 
                                -1, 
                                oracle_block_tensor.unsqueeze(-1)
                            ).squeeze(-1)
                            
                            # Find masked positions in current block
                            current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                            mask_index = (current_block_tensor == self.mask_token_id)
                            
                            # Select best masked position based on oracle probabilities
                            masked_oracle_probs = torch.where(
                                mask_index, 
                                oracle_probs, 
                                torch.tensor(-np.inf, device=logits.device)
                            )
                            
                            # For low_confidence_static: select highest probability
                            best_pos = torch.argmax(masked_oracle_probs).item()
                            best_prob = oracle_probs[best_pos]
                            
                            # Record log probability for THIS BLOCK
                            seq.block_log_probs.append(torch.log(best_prob).item())
                            seq.block_unmask_positions.append(best_pos)
                            
                            # Oracle unmask: use ground truth token
                            seq.intermediate_block_tokens[best_pos] = oracle_block[best_pos]
                            
                            seq.current_denoising_step += 1
                            seq.global_denoising_step += 1
                            
                            # Check if current block is complete
                            is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                                (seq.current_denoising_step >= seq.denoising_steps)
                            
                            if is_fully_denoised:
                                seq.status = SequenceStatus.SAVING
                            
                            seq.num_to_transfer = 1
                            start_idx += block_len
                            continue
                    # ========== END PPL EVALUATION MODE ==========
                    
                    current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    mask_index = (current_block_tensor == self.mask_token_id)
                    num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                    
                    transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                    
                    if seq.remasking_strategy == 'sequential':
                        if mask_index.any():
                            first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                            end_pos = min(first_mask_pos + num_to_transfer, block_len)
                            transfer_index[first_mask_pos:end_pos] = True
                    
                    elif 'low_confidence_static' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        # For dynamic, add threshold logic here if desired
                        _, top_indices = torch.topk(confidence, num_to_transfer)
                        transfer_index[top_indices] = True
                    
                    elif 'low_confidence_dynamic' in seq.remasking_strategy:
                        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                        transfer_index = torch.where(confidence > seq.dynamic_threshold, True, False)
                        if sum(transfer_index) < num_to_transfer:
                            _, top_indices = torch.topk(confidence, num_to_transfer)
                            transfer_index[top_indices] = True
                        num_to_transfer = transfer_index.sum().item() if transfer_index.sum().item() > 0 else num_to_transfer
                    elif 'entropy_bounded' in seq.remasking_strategy:
                        block_probs = probs[start_idx : start_idx + block_len]
                        P = block_probs[mask_index]
                        eps = 1e-12
                        entropies = -(P.clamp_min(eps) * (P.clamp_min(eps)).log()).sum(dim=-1)
                        ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
                        cumsum = torch.cumsum(ent_sorted, dim=0)
                        k = torch.searchsorted(cumsum, torch.tensor(seq.eb_threshold, device=P.device), right=False).item()
                        if k == 0:
                            k = 1
                        # print(k)
                        selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        # print(selected_token_indices)
                        transfer_index[selected_token_indices] = True
                        num_to_transfer = k

                    # update
                    new_block_list = current_block_tensor.tolist()
                    accepted_tokens = seq_x0[transfer_index].tolist()
                    original_indices = transfer_index.nonzero(as_tuple=True)[0].tolist()





                    # newly added
                    if seq.block_first_unmask_steps is None or len(seq.block_first_unmask_steps) != block_len:
                        seq.block_first_unmask_steps = [0] * block_len
                    first_time_global = seq.global_denoising_step + 1
                    for idx in original_indices:
                        if seq.block_first_unmask_steps[idx] == 0:
                            seq.block_first_unmask_steps[idx] = first_time_global



                    

                    for idx, token in zip(original_indices, accepted_tokens):
                        new_block_list[idx] = token
                    seq.intermediate_block_tokens = new_block_list
                    
                    seq.current_denoising_step += 1
                    seq.global_denoising_step += 1
                    
                    # Check if block is fully denoised
                    is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                        (seq.current_denoising_step >= seq.denoising_steps)

                    if is_fully_denoised:
                        # Block is done, commit it and check if generation is finished
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    
                elif seq.status == SequenceStatus.SAVING:
                    # Check if in eval mode
                    if seq.eval_mode and seq.full_oracle_sentence is not None:
                        # Commit current block
                        seq.commit_block(seq.intermediate_block_tokens)
                        seq.advance_to_next_block()  # Save block results
                        
                        # Check if there are more blocks to evaluate
                        remaining_tokens = len(seq.full_oracle_sentence) - seq.num_tokens
                        if remaining_tokens > 0:
                            # Start next block
                            seq.start_new_block()
                        else:
                            # All blocks evaluated, finish
                            seq.status = SequenceStatus.FINISHED
                        
                        seq.num_to_transfer = 0
                    else:
                        # Normal saving logic
                        seq.commit_block(seq.intermediate_block_tokens)
                        seq.num_to_transfer = 0
                        if not seq.is_finished:
                            seq.start_new_block()

                start_idx += seq.block_length
                
        # Filter out finished sequences from the running list
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)