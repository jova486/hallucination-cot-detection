"""Self-annotation using GPT-4o with structured JSON."""

import json
import os
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from glob import glob

class SelfVerifier:
    def __init__(self, verifier_model: str = "gpt-4o-mini", use_json_mode: bool = True):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        self.verifier_model = verifier_model
        self.use_json_mode = use_json_mode
    
    def verify_step(self, question: str, context_steps: List[str], 
                   current_step: str, gold_answer: str = None) -> Dict:
        system_prompt = '''You are a reasoning verification system.
Output ONLY valid JSON:
{
  "verdict": "CORRECT" or "INCORRECT",
  "confidence": 0-100,
  "reasoning": "brief explanation",
  "error_type": "factual" or "logical" or "arithmetic" or "none"
}'''
        
        context_str = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(context_steps))
        if not context_str:
            context_str = "[No previous steps]"
        
        user_prompt = f'''Question: {question}

Previous steps:
{context_str}

Current step: "{current_step}"

Is this step correct?'''
        
        try:
            response = self.client.chat.completions.create(
                model=self.verifier_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"} if self.use_json_mode else None,
                temperature=0.0,
                max_tokens=300
            )
            
            content = json.loads(response.choices[0].message.content)
            assert content["verdict"] in ["CORRECT", "INCORRECT"]
            assert 0 <= content["confidence"] <= 100
            
            return {
                "label": 0 if content["verdict"] == "CORRECT" else 1,
                "confidence": int(content["confidence"]),
                "reasoning": content["reasoning"],
                "error_type": content.get("error_type", "unknown"),
                "needs_human_review": content["confidence"] < 70
            }
        except Exception as e:
            return {
                "label": -1,
                "confidence": 0,
                "reasoning": f"ERROR: {str(e)}",
                "needs_human_review": True
            }

def self_annotate_dataset_robust(step_data_dir: str, output_path: str, 
                                max_retries: int = 3) -> Tuple[List, List, List]:
    verifier = SelfVerifier()
    all_annotations = []
    needs_review = []
    failed = []
    
    files = sorted(glob(f"{step_data_dir}/*.npz"))
    print(f"Annotating {len(files)} files...")
    
    for f in tqdm(files):
        data = np.load(f, allow_pickle=True)
        question = str(data["question"])
        steps = data["steps_text"]
        
        for i, step in enumerate(steps):
            context = list(steps[:i])
            verification = None
            
            for attempt in range(max_retries):
                try:
                    verification = verifier.verify_step(question, context, step)
                    if verification["label"] != -1:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        verification = {"label": -1, "confidence": 0, 
                                      "reasoning": f"ERROR: {e}", "needs_human_review": True}
            
            ann = {"file": f, "step_idx": i, "step_text": step, **verification}
            all_annotations.append(ann)
            if verification["needs_human_review"]:
                needs_review.append(ann)
            if verification["label"] == -1:
                failed.append(ann)
    
    with open(output_path, "w") as fp:
        json.dump(all_annotations, fp, indent=2)
    
    print(f"\n✅ Annotated: {len(all_annotations)}")
    print(f"⚠️  Review needed: {len(needs_review)} ({100*len(needs_review)/len(all_annotations):.1f}%)")
    print(f"❌ Failed: {len(failed)}")
    
    return all_annotations, needs_review, failed
