import argparse
import os
import sys
from pathlib import Path

notebook_path = Path().absolute()
sys.path.append(str(notebook_path.parent))

from utils import load_model
import re
import json
import torch
import pickle
from tqdm import tqdm
import gc

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from abc import ABC, abstractmethod
import direction_utils
from bs4 import BeautifulSoup

_TAGS = ["entity", "relation", "sentence", "invented", "subjective", "unverifiable"]

def remove_deleted_text(text):
    # Use regex to match text between <delete> and </delete> tags
    regex = r'<delete>.*?</delete>'
    
    # Replace all matches with empty string
    # re.DOTALL flag (s) allows matching across multiple lines
    return re.sub(regex, '', text, flags=re.DOTALL)

def remove_empty_tags(html_content):
    # Pattern to match empty tags with optional whitespace between them
    pattern = r'<(\w+)>\s*</\1>'
    
    # Keep removing empty tags until no more changes are made
    prev_content = None
    current_content = html_content
    
    while prev_content != current_content:
        prev_content = current_content
        current_content = re.sub(pattern, '', current_content)
    
    return current_content

def modify(s):
    s = remove_deleted_text(s)
    s = remove_empty_tags(s)

    indicator = [0, 0, 0, 0, 0, 0]
    soup = BeautifulSoup(s, "html.parser")
    s1 = ""
    for t in range(len(_TAGS)):
        indicator[t] = len(soup.find_all(_TAGS[t]))
    # print(soup.find_all(text=True))
    for elem in soup.find_all(text=True):
        if elem.parent.name != "delete":
            s1 += elem
    return s1, int(sum(indicator)>0)

def get_fava_annotated_data():
    # Specify the path to your JSON file
    file_path = './annotations.json'

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    inputs = []
    labels = []
    for d in data:
        s = d['annotated']
        i, label = modify(s)
        labels.append(label)
        inputs.append(i)
    return inputs, labels


class HallucinationJudge(ABC):
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
            predictions.append(int(judgement[0].lower()=='y'))
            print("judgement:", judgement)
        
        return torch.tensor(predictions)
    
    def evaluate_split(self, all_predictions, all_labels, test_indices):
        """Evaluate metrics for a specific split using pre-computed predictions."""
        split_predictions = all_predictions[test_indices]
        split_labels = torch.tensor([all_labels[i] for i in test_indices])
        return direction_utils.compute_classification_metrics(split_predictions, split_labels)

class OpenAIJudge(HallucinationJudge):
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

class GemmaJudge(HallucinationJudge):
    def __init__(self, judge_prompt):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('gemma_2_9b_it')
        
    def get_judgement(self, prompt):
        assistant_tag = '<bos><start_of_turn>model\n\n'
        chat = [
            {'role':'user','content':prompt}
        ]
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
    
class LlamaJudge(HallucinationJudge):
    def __init__(self, judge_prompt):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model('llama_3_8b_it')
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'system', 'content':'You are a helpful assistant who follows instructions exactly.'},
            {'role':'user', 'content':prompt}
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
        response = response.strip().strip('\n').replace('*','')
        return response

def save_predictions(predictions, judge_type, judge_model):
    """Save all predictions to a file."""
    out_name = f'./fava_annotated_results/{judge_type}_{judge_model}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(predictions, f)

def load_predictions(judge_type, judge_model):
    """Load predictions from file if they exist."""
    out_name = f'./fava_annotated_results/{judge_type}_{judge_model}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3_8b_it')
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--judge_type', type=str, choices=['openai', 'llama', 'gemma'], default='llama')
    parser.add_argument('--judge_model', type=str, default='gpt-4o')
    parser.add_argument('--n_seeds', type=int, default=20)
    parser.add_argument('--n_train', type=int, default=0)
    parser.add_argument('--n_val', type=int, default=360)
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    inputs, labels = get_fava_annotated_data()
    out_name = f'./fava_annotated_results/splits_ntrain_0_nval_360_ntotal_460_nseeds_20.pkl'
    with open(out_name, 'rb') as f:
        splits = pickle.load(f)
    
    judge_prompt = "Consider hallucinations of the following types:\n"
    judge_prompt += (
        "(1a) Entity : Contradictory entity errors are a sub-category within Type 1, "
        "where an entity in a statement is incorrect and changing that single entity "
        "can make the entire sentence factually correct.\n"
        
        "(1b) Relation : Contradictory relation errors are another sub-category within "
        "contradictory statements where a semantic relationship (e.g., verbs, prepositions, "
        "or adjectives) in a statement is incorrect.\n"
        
        "(1c) Sentence : Contradictory sentence errors refer to cases where a full statement "
        "entirely contradicts relevant evidence from the web, and cannot be solved via "
        "phrase-level edits.\n"
        
        "(2) Invented : Invented errors refer to statements where the LM generates an entirely "
        "fabricated entity that doesn't exist based on world knowledge. Fictional entities in "
        "creative work aren't included.\n"
        
        "(3) Subjective : Subjective errors refer to expressions about existing entities that "
        "lack universal validity. These statements often do not contain facts and are influenced "
        "by personal beliefs or opinions.\n"
        
        "(4) Unverifiable : These are statements where the LM output contains facts, but no "
        "retrieved.\n\n"
    )
    judge_prompt += (
        'Based on the above definition, does the following statement contain a hallucination? '
        'Simply state yes or no.\nStatement: {statement}'
    )
    
    if args.judge_type == 'openai':
        judge = OpenAIJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'llama':
        judge = LlamaJudge(judge_prompt)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(judge_prompt)

    all_predictions = judge.get_all_predictions(inputs)
    
    # Iterate over seeds using pre-computed predictions
    for seed in tqdm(range(args.n_seeds)):
        split = splits[seed]
        
        metrics = judge.evaluate_split(all_predictions, labels, split['test_indices'])
        print(f"Seed {seed} metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.2f}")

        out_name = f'./fava_annotated_results/{args.judge_type}_{args.judge_model}_seed_{seed}_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()