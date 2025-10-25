"""Incremental CoT generation with activation capture."""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional

class IncrementalCoTGenerator:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float16"):
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto", low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
    
    def generate_cot_with_step_activations(
        self, question: str, gold_answer: Optional[str] = None, max_steps: int = 10
    ) -> Dict:
        full_solution = ""
        step_data = []
        
        for step_num in range(1, max_steps + 1):
            if step_num == 1:
                prompt = f'''Solve this step-by-step.

Question: {question}

Solution:
Step 1:'''
            else:
                prompt = f'''{full_solution}
Step {step_num}:'''
            
            step_result = self._generate_single_step(prompt)
            if step_result is None:
                break
            
            step_data.append({
                "step_num": step_num,
                "step_text": step_result["text"],
                "activations_last": step_result["activations_last"],
                "activations_mean": step_result["activations_mean"],
                "activations_max": step_result["activations_max"],
                "mean_logprob": step_result["mean_logprob"],
                "min_logprob": step_result["min_logprob"],
                "std_logprob": step_result["std_logprob"],
                "n_tokens": step_result["n_tokens"]
            })
            
            full_solution += f"\nStep {step_num}: {step_result['text']}"
            if self._is_final_answer(step_result["text"]):
                break
        
        return {
            "question": question,
            "gold_answer": gold_answer,
            "full_solution": full_solution,
            "steps": step_data,
            "n_steps": len(step_data)
        }
    
    def _generate_single_step(self, prompt: str, max_tokens: int = 50) -> Optional[Dict]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        step_tokens = []
        step_logprobs = []
        all_step_hidden_states = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token_id = torch.argmax(logits).item()
                
                token_logprob = torch.log(probs[next_token_id]).item()
                step_logprobs.append(token_logprob)
                
                token_hidden_states = torch.stack([
                    layer[0, -1, :].cpu() for layer in outputs.hidden_states[1:]
                ])
                all_step_hidden_states.append(token_hidden_states)
                step_tokens.append(next_token_id)
                
                decoded = self.tokenizer.decode(step_tokens, skip_special_tokens=True)
                if decoded.endswith("\n\n") or decoded.endswith("\n"):
                    break
                if next_token_id == self.tokenizer.eos_token_id:
                    return None
                
                inputs.input_ids = torch.cat([
                    inputs.input_ids, torch.tensor([[next_token_id]], device=self.device)
                ], dim=1)
        
        if not step_tokens:
            return None
        
        full_step_activations = torch.stack(all_step_hidden_states, dim=0).numpy()
        pooled = self._compute_pooling(full_step_activations)
        step_text = self.tokenizer.decode(step_tokens, skip_special_tokens=True).strip()
        
        return {
            "text": step_text,
            "n_tokens": len(step_tokens),
            **pooled,
            "mean_logprob": np.mean(step_logprobs),
            "min_logprob": np.min(step_logprobs),
            "std_logprob": np.std(step_logprobs) if len(step_logprobs) > 1 else 0.0
        }
    
    def _compute_pooling(self, activations: np.ndarray) -> Dict:
        return {
            "activations_last": activations[-1],
            "activations_mean": activations.mean(axis=0),
            "activations_max": activations.max(axis=0)
        }
    
    def _is_final_answer(self, text: str) -> bool:
        markers = ["therefore", "thus", "so the answer is", "final answer", "the answer is"]
        return any(m in text.lower() for m in markers)
