import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_model

import torch
import pickle
from direction_utils import compute_classification_metrics
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from abc import ABC, abstractmethod
from utils import load_model

def read_hallucination_prompts_from_lines(lines):
    import re
    
    dicts = []
    for line in lines:
        line = line[1:-1]
        x = re.findall('".*?"', line)
        
        prompt = {}
        prompt['knowledge'] = x[1].strip('"')
        prompt['question'] = x[3].strip('"')
        prompt['answer'] = x[5].strip('"')
        prompt['hallucination'] = x[7].strip('"')
        dicts.append(prompt)
    return dicts

def clean(prompts):
    new_prompts = []
    for p in prompts:
        if p['question'] == 'question':
            continue
        new_prompts.append(p)
    return new_prompts

def get_halu_eval_data(hal_type):
    if hal_type=='qa':
        data_path = "../data/hallucinations/halu_eval/qa_data.txt"
        template = 'Yes or no, is the answer to the following question factual?\n\nQ: {question}\n\nA: {answer}'

        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)
            n = len(raw_prompts)
            # Only use evaluation data (second half)
            raw_prompts = raw_prompts[int(n//2):]

        prompts = clean(raw_prompts)
        inputs = []
        labels = []
        for prompt in prompts:
            x_pos = template.format(question=prompt['question'], answer=prompt['answer'])
            x_neg = template.format(question=prompt['question'], answer=prompt['hallucination'])
            inputs.append(x_pos)
            inputs.append(x_neg)
            labels += [0,1]
            
    elif hal_type=='general':
        data_path = "../data/hallucinations/halu_eval/general_data.txt"
        template = 'Is the response to the following query factual? Simply state yes or no.\n\nQuery: {query}\n\nResponse: {response}'
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)
            prompts = clean(raw_prompts)
            
        inputs = []
        labels = []
        for prompt in prompts:
            x = template.format(query=prompt['question'], response=prompt['answer'])
            inputs.append(x)
            # Flip the label logic to match the other code
            label = 0 if prompt['hallucination']=='no' else 1
            labels.append(label)
        
    return inputs, labels

class HallucinationJudge(ABC):
    def __init__(self, judge_prompt):
        self.judge_prompt = judge_prompt
    
    @abstractmethod
    def get_judgements(self, prompts):
        pass
    
    def evaluate_inputs(self, test_inputs, test_labels, splits):
        # Get all judgements at once for efficiency
        all_judgements = self.get_judgements(test_inputs)
        
        results = []
        for seed in range(len(splits)):
            split = splits[seed]
            
            # Split judgements according to the same indices used in the original code
            test_indices = split['test_indices']
            split_test_judgements = [all_judgements[i] for i in test_indices]
            split_test_targets = [test_labels[i] for i in test_indices]
            
            # Convert to tensors for compute_classification_metrics
            test_judgements_tensor = torch.tensor(split_test_judgements)
            test_targets_tensor = torch.tensor(split_test_targets)
            
            # Calculate metrics using compute_classification_metrics
            test_metrics = compute_classification_metrics(test_judgements_tensor, test_targets_tensor)
            
            trivial_acc = max(
                (sum(split_test_targets)/len(split_test_targets)),
                1-(sum(split_test_targets)/len(split_test_targets))
            ) * 100
            
            test_metrics['trivial_acc'] = trivial_acc
            
            results.append({
                'seed': seed,
                'test_metrics': test_metrics,
            })
            
        return results

class OpenAIJudge(HallucinationJudge):
    def __init__(self, judge_prompt, model_name):
        super().__init__(judge_prompt)
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024),
    )
    def _get_single_judgement(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0
        )
        return int(response.choices[0].message.content[0].lower() == 'n')
    
    def get_judgements(self, prompts):
        return [self._get_single_judgement(prompt) for prompt in tqdm(prompts)]

class LlamaJudge(HallucinationJudge):
    def __init__(self, judge_prompt, model_path=None):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('llama_3_8b_it')
        
    def get_judgements(self, prompts):
        judgements = []
        for prompt in tqdm(prompts):
            chat = [{'role': 'system', 'content': 'You are a helpful assistant who follows instructions exactly.'},
                    {'role': 'user', 'content': prompt}]
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
            response = response[response.find(assistant_tag)+len(assistant_tag):].strip()
            judgements.append(int(response[0].lower() == 'n'))
            
        return judgements

class GemmaJudge(HallucinationJudge):
    def __init__(self, judge_prompt, model_path=None):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('gemma_2_9b_it')
        
    def get_judgements(self, prompts):
        judgements = []
        for prompt in tqdm(prompts):
            assistant_tag = '<bos><start_of_turn>model\n\n'
            chat = [{'role': 'user', 'content': prompt}]
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

            judgements.append(int(response[0].lower() == 'n'))
            
        return judgements
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_3_8b_it')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--hal_type', type=str, default='qa')
    parser.add_argument('--judge_type', type=str, choices=['openai', 'llama', 'gemma'], default='llama')
    parser.add_argument('--judge_model', type=str, default='gpt-4o')
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    inputs, labels = get_halu_eval_data(args.hal_type)
    
    # Get the same splits as used in the original code
    if args.hal_type == 'qa':
        out_name = f'./halu_eval_results/qa_test_splits_nval_3997_ntotal_7994_nseeds_5.pkl'
    else:
        out_name = f'./halu_eval_results/general_test_splits_nval_2253_ntotal_4507_nseeds_5.pkl'
        
    with open(out_name, 'rb') as f:
        splits = pickle.load(f)
    
    judge_prompt = ''  # Customize if needed
    if args.judge_type == 'openai':
        judge = OpenAIJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'llama':
        judge = LlamaJudge(judge_prompt)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(judge_prompt)
    
    results = judge.evaluate_inputs(inputs, labels, splits)
    
    # Save results for each seed separately to match original code structure
    for result in results:
        seed = result['seed']
        
        test_metrics = result['test_metrics']
        test_out_name = f'./halu_eval_results/{args.judge_type}_{args.judge_model}_seed_{seed}_{args.hal_type}_metrics.pkl'
        with open(test_out_name, 'wb') as f:
            pickle.dump(test_metrics, f)

if __name__ == '__main__':
    main()