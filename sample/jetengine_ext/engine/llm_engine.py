import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
# Added imports for profiling
import torch
from torch import nn
from contextlib import nullcontext
import torch.profiler as torch_profiler

from jetengine_ext.config import Config
from jetengine_ext.sampling_params import SamplingParams
from jetengine_ext.engine.sequence import Sequence, RunType
from jetengine_ext.engine.scheduler import Scheduler
from jetengine_ext.engine.model_runner import ModelRunner
from jetengine_ext.utils.loader import load_from_hf_model


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        config.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        assert config.mask_token_id is not None, "Model tokenizer must have a mask_token_id or pad_token_id"

        self.config = config
        self.scheduler = Scheduler(config)
        self.scheduler.consistent_sampling_params = False
        atexit.register(self.exit)

    def offload_parameters(self, include_buffers: bool = False):
        """
        Replace all parameter (and buffer) storages with meta tensors.
        Keeps shapes/dtypes, frees GPU/CPU memory.
        """

        def offload_parameters_keep_buffers(model: torch.nn.Module):
            """
            Move *parameters* to meta to free memory while keeping buffers unchanged.
            Works for any module tree.
            """
            # 1) Snapshot real buffers (module reference + buffer name + tensor)
            saved_buffers = []
            for mod in model.modules():
                for bname, buf in list(mod._buffers.items()):
                    if buf is not None:
                        saved_buffers.append((mod, bname, buf))

            # 2) Move everything to meta
            model.to_empty(device=torch.device("meta"))

            # 3) Restore the saved, real buffers
            for mod, bname, buf in saved_buffers:
                # Reattach the original tensor (device/dtype preserved)
                mod._buffers[bname] = buf

            torch.cuda.empty_cache()
        if include_buffers:
            self.model_runner.model.to_empty(device=torch.device("meta"))
        else:
            offload_parameters_keep_buffers(self.model_runner.model)

        print("Successfully cleaned old parameters (buffers kept)." if not include_buffers
              else "Successfully cleaned old parameters and buffers.")

    def reload_parameters(self, hf_model: nn.Module):
        load_from_hf_model(self.model_runner.model, hf_model=hf_model)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            if self.tokenizer.pad_token_id in prompt:
                start = prompt.index(self.tokenizer.pad_token_id) + 1
                prompt = prompt[start:]
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params)
        seq.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler.add(seq)

    def step(self):
        scheduled_seqs, run_type = self.scheduler.schedule()
        if scheduled_seqs is None:
            return [], 0 # Nothing to run

        logits = self.model_runner.call("run", scheduled_seqs, run_type)
        self.scheduler.postprocess(scheduled_seqs, logits, run_type)
        
        #finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_seqs if seq.is_finished]
        
        finished_outputs = [
            (seq.seq_id, seq.completion_token_ids, seq.first_unmask_steps)
            for seq in scheduled_seqs
            if seq.is_finished
        ]

        # Throughput calculation needs to be adapted for block-wise generation
        num_tokens = [self.scheduler.running[i].num_to_transfer if hasattr(self.scheduler.running[i], 'num_to_transfer') else 0 for i in range(len(self.scheduler.running))]
        return finished_outputs, sum(num_tokens)

    def is_finished(self):
        return self.scheduler.is_finished()

    def _clean_token_ids(self, token_ids):
        # Accept tensors, numpy ints, etc.
        try:
            token_ids = list(token_ids)
        except Exception:
            token_ids = [token_ids]
        
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        mask_id = getattr(self.config, "mask_token_id", None)

        cleaned = []
        for t in token_ids:
            if t is None or t < 0 or t == mask_id or t >= vocab_size:
                if t not in special_ids:
                    cleaned.append(0)
                    continue
            cleaned.append(t)
        return cleaned

    def _safe_decode(self, token_ids):
        ids = self._clean_token_ids(token_ids)
        # skip_special_tokens can be True or False; doesn't affect the None issue
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        # ... (This method remains largely the same, but the progress bar will update differently) ...
        # The logic inside the `while not self.is_finished()` loop correctly calls `self.step()`
        # and collects outputs.
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        
        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            while not self.is_finished():
                output, num_processed = self.step()
                if profile:
                    prof.step()
                total_generated_tokens += num_processed
                
                throughput = total_generated_tokens / (perf_counter() - start_time)
                if use_tqdm:
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})

                #for seq_id, token_ids in output:
                #    outputs[seq_id] = token_ids
                for seq_id, token_ids, unmask_times in output:
                    outputs[seq_id] = {"token_ids": token_ids, "unmask_times": unmask_times}
                    if use_tqdm:
                        pbar.update(1)

        #outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        #outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {
                "text": self._safe_decode(item["token_ids"]),
                "token_ids": self._clean_token_ids(item["token_ids"]),
                "first_unmask_times": item["unmask_times"],   # 与 token_ids 等长
            }
            for item in outputs
        ]

        if use_tqdm:
            pbar.close()
        return outputs

    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        max_active: int | None = None,
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total
            self.scheduler.consistent_sampling_params = True

        if max_active is None:
            max_active = getattr(self.scheduler, "max_num_seqs", 32)

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}
        pending_idx = 0

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i])
        pending_idx = initial

        total_generated_tokens = 0
        start_time = perf_counter()

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            while not self.is_finished() or pending_idx < total:
                # Top up to capacity before each step
                running = getattr(self.scheduler, "running", [])
                deficit = max_active - len(running)
                while deficit > 0 and pending_idx < total:
                    self.add_request(prompts[pending_idx], sampling_params[pending_idx])
                    pending_idx += 1
                    deficit -= 1

                output, num_processed = self.step()
                if profile:
                    prof.step()
                total_generated_tokens += num_processed

                if use_tqdm:
                    throughput = total_generated_tokens / (perf_counter() - start_time + 1e-6)
                    pbar.set_postfix({"Throughput": f"{int(throughput)} tok/s"})
                    pbar.update(len(output))

                #for seq_id, token_ids in output:
                #    outputs[seq_id] = token_ids
                for seq_id, token_ids, unmask_times in output:
                    outputs[seq_id] = {"token_ids": token_ids, "unmask_times": unmask_times}

        #outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        #results = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs_list]
        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        results = [
            {
                "text": self._safe_decode(item["token_ids"]),
                "token_ids": self._clean_token_ids(item["token_ids"]),
                "first_unmask_times": item["unmask_times"],
            }
            for item in outputs_list
        ]

        if use_tqdm:
            pbar.close()
        return results
    
    def evaluate_perplexity(
        self,
        sentences: list[list[int]],
        block_length: int = 4,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        Evaluate perplexity using oracle unmasking with flexible block length.
        
        Args:
            sentences: List of token sequences, e.g., [[1,2,3,4,5,6,7,8], ...]
            block_length: Fixed block length for evaluation (e.g., 4)
            use_tqdm: Show progress bar
        
        Returns:
            List of dicts with: 'perplexity', 'nll', 'unmask_order', 'log_probs', 'block_ppls'
        """
        import numpy as np
        
        results = []
        
        if use_tqdm:
            pbar = tqdm(total=len(sentences), desc="Evaluating PPL")
        
        for sentence in sentences:
            # Determine number of blocks
            num_blocks = (len(sentence) + block_length - 1) // block_length
            
            # Pad sentence to multiple of block_length if needed
            padded_length = num_blocks * block_length
            if len(sentence) < padded_length:
                # Note: you may want different padding strategy
                sentence_padded = sentence + [self.tokenizer.pad_token_id] * (padded_length - len(sentence))
            else:
                sentence_padded = sentence
            
            # Create sampling params for evaluation
            sampling_params = SamplingParams(
                block_length=block_length,
                denoising_steps=block_length,  # 1 token per step
                remasking_strategy='low_confidence_static',
                eval_mode=True,
                ignore_eos=True,
                max_tokens=len(sentence_padded),
            )
            
            # Determine prefill length (tokens before first block to denoise)
            # For pure left-to-right: prefill = 0 (start from all masks)
            prefill_tokens = []
            
            # Create sequence
            seq = Sequence(prefill_tokens, self.config.mask_token_id, sampling_params)
            seq.eos_token_id = self.tokenizer.eos_token_id
            seq.set_full_oracle_sentence(sentence_padded)
            
            self.scheduler.add(seq)
            
            # Run evaluation
            while not self.is_finished():
                self.step()
            
            # Extract results
            # Only use log_probs for actual sentence (not padding)
            valid_log_probs = seq.step_log_probs[:len(sentence)]
            valid_unmask_order = seq.step_unmask_positions[:len(sentence)]
            
            nll = -np.mean(valid_log_probs)
            ppl = np.exp(nll)
            
            # Compute per-block PPL for analysis
            block_ppls = []
            for i in range(num_blocks):
                start = i * block_length
                end = min(start + block_length, len(sentence))
                block_log_probs = seq.step_log_probs[start:end]
                if block_log_probs:
                    block_nll = -np.mean(block_log_probs)
                    block_ppls.append(np.exp(block_nll))
            
            results.append({
                'sentence': sentence,
                'perplexity': ppl,
                'nll': nll,
                'unmask_order': valid_unmask_order,
                'log_probs': valid_log_probs,
                'block_ppls': block_ppls,
                'num_blocks': num_blocks,
            })
            
            if use_tqdm:
                pbar.update(1)
        
        if use_tqdm:
            pbar.close()
        
        return results