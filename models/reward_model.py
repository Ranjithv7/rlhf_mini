import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)

class RewardModel:
    def __init__(self, model_name: str, bert_model_name: str = "bert-base-uncased", cache_dir: str = "models/cache" ):
        self.model_name = model_name
        self.bert_model_name = bert_model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.load_model()

    def load_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype = torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast = True,
            padding_side = "left"  ## Why
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Adding the regression head
        self.model.score = nn.Linear(self.model.config.hidden_size, 1) # To assign scores
        logging.info(" LLM Loaded successfully")


        # It's time to load bert model 
        self.bert_model = AutoModelForCausalLM.from_pretrained(
            self.bert_model_name, 
            cache_dir = self.cache_dir
        )
        self.bert_tokenizer = self.tokenizer.from_pretrained(self.bert_model_name)
        logging.info("Bert Model loaded successfully")

    def forward(self, input_ids: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        outputs = self.model(input_ids, attention_mask, output_hidden_state = True)
        last_hidden_State = outputs.hidden_states[-1][:, -1, :]
        reward = self.model.score(last_hidden_State)
        return reward
    

    def verify_steps(self, steps: list[str]) -> float:
        scores = []
        for step in steps:
            encoding = self.tokenizer(step, return_tensors = "pt", max_length= 128, truncation= True, padding = "max_length" )
            with torch.no_grad():
                outputs = self.bert_model(**encoding)
            score = outputs.last_hidden_state[:,0, :].mean().item()
            scores.append(score)
        return sum(scores) / len(scores)
    
    def save_model(self, savepath: str ) -> None:
        os.makedirs(savepath, exist_ok= True)
        self.model.save_pretrained(savepath)
        self.tokenizer.save_pretrained(savepath)
        logging.info(" Successfully model saved")
        
        




