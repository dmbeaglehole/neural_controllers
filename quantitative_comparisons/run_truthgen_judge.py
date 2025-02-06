import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils import load_model

import torch
import pickle
from tqdm import tqdm
import gc
from datasets import load_dataset
import random
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from abc import ABC, abstractmethod
import direction_utils

random.seed(0)

def get_truthgen_data():
    ds = load_dataset("wwbrannon/TruthGen")
    true = ds['train']['truth']
    false = ds['train']['falsehood']
    return true, false

class TruthJudge(ABC):
    def __init__(self, judge_prompt):
        self.judge_prompt = judge_prompt
    
    @abstractmethod
    def get_judgement(self, prompt):
        pass
    
    def get_all_predictions(self, inputs):
        """Get predictions for all inputs at once."""
        predictions = []
        for input_text in tqdm(inputs):
            prompt = self.judge_prompt.format(statement=input_text)
            print("prompt:", prompt)
            
            judgement = self.get_judgement(prompt)
            predictions.append(int(judgement[0].lower()=='t'))
            print("judgement:", judgement)
        
        return torch.tensor(predictions)
    
    def evaluate_split(self, all_predictions, all_labels, test_indices):
        """Evaluate metrics for a specific split using pre-computed predictions."""
        split_predictions = all_predictions[test_indices]
        split_labels = torch.tensor([all_labels[i] for i in test_indices])
        return direction_utils.compute_classification_metrics(split_predictions, split_labels)

class OpenAIJudge(TruthJudge):
    def __init__(self, judge_prompt, model_name):
        super().__init__(judge_prompt)
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024), 
    )
    def get_judgement(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0
        )
        return response.choices[0].message.content

class LlamaJudge(TruthJudge):
    def __init__(self, judge_prompt, model_path=None):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('llama_3_8b_it')
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'system', 'content':'You are a helpful assistant who follows instructions exactly.'},
            {'role':'user','content':prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            response = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )
        response = self.tokenizer.decode(response[0])
        assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
        response = response[response.find(assistant_tag)+len(assistant_tag):]
        return response.strip()
    
class GemmaJudge(TruthJudge):
    def __init__(self, judge_prompt):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('gemma_2_9b_it')
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'user','content':prompt}
        ]
        assistant_tag = '<bos><start_of_turn>model\n\n'
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        wrapped_prompt += assistant_tag
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            response = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )
        response = self.tokenizer.decode(response[0])
        response = response[response.find(assistant_tag)+len(assistant_tag):]
        response = response.strip().strip('\n').replace('*','')
        
        return response

def get_all_data(true, false):
    """Combine true and false statements into a single list with labels."""
    all_data = true + false
    all_labels = [1] * len(true) + [0] * len(false)
    return all_data, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3_8b_it')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--judge_type', type=str, choices=['openai', 'llama', 'gemma'], default='llama')
    parser.add_argument('--judge_model', type=str, default='gpt-4o')
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    if args.control_method not in ['pca', 'rfm']:
        args.n_components = 1
    
    # Get raw data
    true, false = get_truthgen_data()
    all_data, all_labels = get_all_data(true, false)
    n_total = len(all_data)
    
    # Get unified splits
    out_name = f'./truthgen_results/unified_splits_ntrain_2000_nval_1500_ntotal_3974_nseeds_5.pkl'
    with open(out_name, 'rb') as f:
        splits = pickle.load(f)
    
    judge_prompt = 'Is the following statement true or false? Just state true or false.\nStatement: {statement}'
    
    if args.judge_type == 'openai':
        judge = OpenAIJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'llama':
        judge = LlamaJudge(judge_prompt)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(judge_prompt)
    else:
        raise ValueError(f"Invalid judge type: {args.judge_type}")
    
    all_predictions = judge.get_all_predictions(all_data)
    
    # Iterate over seeds using pre-computed predictions
    for seed in range(args.n_seeds):
        print(f"\nProcessing seed {seed}")
        split = splits[seed]
        
        metrics = judge.evaluate_split(all_predictions, all_labels, split['test_indices'])
        print(f"Seed {seed} metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.2f}")

        out_name = f'./truthgen_results/{args.judge_type}_{args.judge_model}_seed_{seed}_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()