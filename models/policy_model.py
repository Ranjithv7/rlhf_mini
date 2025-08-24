import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Dict, Tuple
import logging
import nltk

logger = logging.getLogger(__name__)


class policyModel:
    def __init__(self, model_name: str, cache_dir: str = "models/cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.value_head = None
        self.load_model()
    

    def load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir = self.cache_dir,
            torch_dtype = torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            truncation = True,
            use_fast = True,
            padding_side = "left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)
    

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        outputs = self.model.forward(input_ids, attention_mask, output_hidden_states=True)
        logits = outputs.logits
        last_hidden_state = outputs.hidden_statehidden_states[-1][:, -1, :]
        value = self.value_head(last_hidden_state)
        return logits, value

    def generate_rl_responses(self, prompt: str, max_new_tokens: int = 50, temperature: int = 0.7) -> Tuple[str, list[str]]:
        inputs = self.tokenizer(
            prompt, 
            max_length= 64, 
            truncation= True, 
            return_tensors ="pt"
        )
        with torch.no_grad():
            rl_generated_responses = self.model.generate(
                **inputs, 
                max_new_tokens = max_new_tokens,
                temperature = temperature,
                do_sample = True,
                pad_token_id = self.tokenizer.pad_token_id,
                no_repeat_ngram_size = 2, ## Why ??
                return_dict_in_generate = True, 
                output_scores = True

            )
            response_decoding = self.tokenizer.decode(rl_generated_responses.sequences[0], skip_special_tokens= True)

            # Extracting the steps
            steps = nltk.sent_tokenize(response_decoding[len(prompt):].strip())
            for i, step in enumerate(steps):
                if len(steps.strip()) > 5:
                    steps = f" step {i+1}: {step}"
            if not steps:
                steps = [f" step1 : {response_decoding[len(prompt):].strip()}"]
            return response_decoding, steps
        
    def save_model(self, savepath: str) -> None:
        os.makedirs(savepath, exist_ok= True)
        self.model.save_pretained(savepath)
        self.tokenizer.save_pretrained(savepath)
        torch.save(self.value_head.state_dict(), os.path.join(savepath, "value_head.pt"))
        logger.info("Model saved to {save_path}")
    

    def load_checkpoint(self, checkpointpath: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpointpath, 
            cache_dir = self.cache_dir,
            torch_dtype = torch.bfloat16,

        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpointpath)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.value_head.load_state_dict(torch.load(os.path.join(checkpointpath, "value_head.pt")))
        logger.info(f"Checkpoint loaded from {checkpointpath}")
