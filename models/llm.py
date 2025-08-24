import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple


# device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# print(device)

class LLM:
    def __init__(self, model_name: str, cache_dir: str = "models/cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir = self.cache_dir,
            torch_dtype = torch.bfloat16
            # force_download=True 

        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir = self.cache_dir, 
            use_fast=True,
            padding_side="left"
            # force_download=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        

    def tokenize(self, text: str, max_length: int = 512 ) -> dict:
        return self.tokenizer(text, return_tensors = "pt", max_length = max_length, truncation = True, padding = True)
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7 ) -> str: ## why 
        inputs = self.tokenize(prompt)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, ## why 
                max_new_tokens = max_new_tokens, 
                temperature = temperature, 
                do_sample = True, 
                pad_token_id = self.tokenizer.pad_token_id, ## Why 
                no_repeat_ngram_size=2
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens = True) ## why 
    

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> dict:
        outputs = self.model(
            input_ids = input_ids, 
            attention_mask = attention_mask
        )
        return outputs
    
    def save_model(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok= True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f" Model saved to {save_path} successfully")
    
    def load_checkpoint(self, checkpoint_path: str)-> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, 
            cache_dir = self.cache_dir,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

if __name__ == "__main__":
    model = LLM(model_name="facebook/opt-350m")
    prompt = "Solve 2x + 3 = 7"
    print("Starting generation...", flush=True)
    response = model.generate(prompt)
    print(f" prompt: {prompt} \nThis is the model response: {response}")
    print(f"Prompt: {prompt}\nResponse: {response}", flush=True) 



