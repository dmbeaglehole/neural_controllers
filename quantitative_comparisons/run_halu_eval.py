import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model

import numpy as np
import torch
import pickle
from tqdm import tqdm
import gc
import random
random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

def clean(prompts):
    new_prompts = []
    for p in prompts:
        if p['question'] == 'question':
            continue
        new_prompts.append(p)
    return new_prompts

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

def create_positive_negative_pairs(inputs, labels, max_pairs):
    """
    Creates pairs where each pair consists of one positive and one negative example.
    If one list is shorter, extends it by randomly sampling from itself.

    Args:
        inputs (list of str): The input texts.
        labels (list of int): Corresponding toxicity labels (1 for positive, 0 for negative).
        max_pairs (int): Maximum number of pairs to create.

    Returns:
        Tuple of (pairs, pair_labels)
    """
    if max_pairs is None:
        max_pairs = len(inputs)

    # Separate inputs by label
    positive_inputs = [inp for inp, lbl in zip(inputs, labels) if lbl == 1]
    negative_inputs = [inp for inp, lbl in zip(inputs, labels) if lbl == 0]

    # Debugging statements
    print(f"Number of positive examples: {len(positive_inputs)}")
    print(f"Number of negative examples: {len(negative_inputs)}")

    # Determine which list is longer
    max_length = max(len(positive_inputs), len(negative_inputs))
    num_pairs = min(max_length, max_pairs)

    # Extend shorter list if necessary
    if len(positive_inputs) < max_length:
        additional_needed = max_length - len(positive_inputs)
        positive_inputs.extend(random.choices(positive_inputs, k=additional_needed))
    elif len(negative_inputs) < max_length:
        additional_needed = max_length - len(negative_inputs)
        negative_inputs.extend(random.choices(negative_inputs, k=additional_needed))

    # Take only up to max_pairs
    paired_positive = positive_inputs[:num_pairs]
    paired_negative = negative_inputs[:num_pairs]

    # Create pairs as tuples (positive, negative)
    pairs = list(zip(paired_positive, paired_negative))

    # Create corresponding label pairs (1, 0)
    pair_labels = [[1, 0] for _ in range(len(pairs))]

    return pairs, pair_labels

def get_halu_eval_data(tokenizer, hal_type):
    if hal_type=='qa':
        data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/qa_data.txt'
        unwrapped_template = 'Consider the factuality of the following question and answer.\n\nQ: {question}\n\nA: {answer}'
        chat = [
            {
                "role": "user", 
                "content": unwrapped_template
            },
        ]
        template = tokenizer.apply_chat_template(chat, tokenize=False)

        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)

        n = len(raw_prompts)
        clean_train_prompts = clean(raw_prompts[:int(n//2)])
        clean_eval_prompts = clean(raw_prompts[int(n//2):])
        
        # Generate training data
        train_inputs = []
        train_labels = []
        for prompt in clean_train_prompts:
            x_pos = template.format(question=prompt['question'], answer=prompt['answer'])
            x_neg = template.format(question=prompt['question'], answer=prompt['hallucination'])
            train_inputs.append(x_pos)
            train_inputs.append(x_neg)
            train_labels += [1,0]
        
        test_inputs = []
        test_labels = []
        for prompt in clean_eval_prompts:
            x_pos = template.format(question=prompt['question'], answer=prompt['answer'])
            x_neg = template.format(question=prompt['question'], answer=prompt['hallucination'])
            test_inputs.append(x_pos)
            test_inputs.append(x_neg)
            test_labels += [1,0]
            
    elif hal_type=='general':
        # Get QA data for training
        qa_data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/qa_data.txt'
        unwrapped_qa_template = 'Consider the factuality of the following question and answer.\n\nQ: {question}\n\nA: {answer}'
        chat = [
            {
                "role": "user", 
                "content": unwrapped_qa_template
            },
        ]
        qa_template = tokenizer.apply_chat_template(chat, tokenize=False)

        with open(qa_data_path, 'r') as f:
            lines = f.readlines()
            raw_train_prompts = read_hallucination_prompts_from_lines(lines)
            
        clean_train_prompts = clean(raw_train_prompts)
        n = len(clean_train_prompts)
        train_prompts = clean_train_prompts[:int(n//2)]
        
        # Generate training data from QA data
        train_inputs = []
        train_labels = []
        for prompt in train_prompts:
            x_pos = qa_template.format(question=prompt['question'], answer=prompt['answer'])
            x_neg = qa_template.format(question=prompt['question'], answer=prompt['hallucination'])
            train_inputs.append(x_pos)
            train_inputs.append(x_neg)
            train_labels += [1,0]
            
        # Get general data for evaluation
        data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/general_data.txt'
        unwrapped_template = 'Consider the response to the following query.\n\nQuery: {query}\n\nResponse: {response}'
        chat = [
            {
                "role": "user", 
                "content": unwrapped_template
            },
        ]
        template = tokenizer.apply_chat_template(chat, tokenize=False)
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)
            
        eval_prompts = clean(raw_prompts)
            
        test_inputs = []
        test_labels = []
        for prompt in eval_prompts:
            x = template.format(query=prompt['question'], response=prompt['answer'])
            test_inputs.append(x)
            test_labels.append(0 if prompt['hallucination']=='no' else 1)
    
    return train_inputs, train_labels, test_inputs, test_labels

def get_splits(n_val, n_total, n_seeds, hal_type):
    """
    Generate validation splits for the test set instead of training set.
    n_val: number of validation samples
    n_total: total number of test samples
    n_seeds: number of different random seeds for splits
    hal_type: type of hallucination data
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/{hal_type}_test_splits_nval_{n_val}_ntotal_{n_total}_nseeds_{n_seeds}.pkl'
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        pass

    splits = []
    indices = np.arange(n_total)

    for seed in range(n_seeds):
        np.random.seed(seed)
        val_indices = np.random.choice(indices, size=n_val, replace=False)
        test_indices = np.setdiff1d(indices, val_indices)

        splits.append({
            'test_indices': test_indices,
            'val_indices': val_indices
        })

    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)

    return splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--hal_type', type=str, default='qa')
    parser.add_argument('--n_seeds', type=int, default=5)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    hal_type = args.hal_type
    n_seeds = args.n_seeds
    
    unsupervised = control_method=='pca'
    if control_method not in ['rfm']:
        n_components=1
        
    use_logistic=(control_method=='logistic')
    
    original_control_method = str(control_method)
    use_rfm = False
    if control_method=='logistic_rfm':
        control_method='logistic'
        use_rfm=True
    elif control_method=='linear_rfm':
        control_method='linear'
        use_rfm=True
    elif control_method=='rfm_linear':
        control_method='rfm'
        use_rfm=False
    elif control_method=='rfm':
        use_rfm=True
    else:
        use_rfm=False
        
    if control_method=='logistic_concat':
        use_logistic=True
    
    print("Num components:", n_components)
    
    language_model, tokenizer = load_model(model=model_name)
    
    controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method,
            rfm_iters=5,
            batch_size=2
        )
    
    train_inputs, train_labels, test_inputs, test_labels = get_halu_eval_data(tokenizer, hal_type)

    # Create pairs if unsupervised is True
    if unsupervised:
        train_pairs, train_pair_labels = create_positive_negative_pairs(
            inputs=train_inputs,
            labels=train_labels,
            max_pairs=None  # This will use all possible pairs
        )
        train_inputs = np.concatenate(train_pairs).tolist()
        train_labels = np.concatenate(train_pair_labels).tolist()
            
    try:
        controller.load(concept='halu_eval_detect', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
    except:
        controller.compute_directions(train_inputs, train_labels)
        controller.save(concept='halu_eval_detect', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

    # Changed to split the test set instead of train set
    splits = get_splits(n_val=len(test_inputs)//2, n_total=len(test_inputs), n_seeds=n_seeds, hal_type=hal_type)
    
    for seed in tqdm(range(n_seeds)):
        split = splits[seed]
        
        # Apply split to test data instead of train data
        final_test_inputs = [test_inputs[i] for i in split['test_indices']]
        split_val_inputs = [test_inputs[i] for i in split['val_indices']]
        final_test_labels = [test_labels[i] for i in split['test_indices']]
        split_val_labels = [test_labels[i] for i in split['val_indices']]

        ntrain  = len(train_inputs)
        nval = len(split_val_inputs)
        ntest = len(final_test_inputs)
        out_name = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_results/{hal_type}_{control_method}_data_counts_seed_{seed}.pkl'
        with open(out_name, 'wb') as f:
            counts = {'train':ntrain, 'val':nval, 'test':ntest}
            pickle.dump(counts, f)
        
        assert(len(split_val_inputs)==len(split_val_labels))
        assert(len(final_test_inputs)==len(final_test_labels))
        print("val_inputs", len(split_val_inputs), "test_inputs", len(final_test_inputs), 
            "val_labels", len(split_val_labels), "test_labels", len(final_test_labels))
        
        val_metrics, test_metrics, _ = controller.evaluate_directions(
            split_val_inputs, split_val_labels,
            final_test_inputs, final_test_labels,
            n_components=n_components,
            use_logistic=use_logistic,
            use_rfm=use_rfm,
            unsupervised=unsupervised
        )
        
        trivial_acc = max((sum(final_test_labels)/len(final_test_labels)), 1-(sum(final_test_labels)/len(final_test_labels)))*100
        
        results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_results'
        os.makedirs(results_dir, exist_ok=True)
        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_{hal_type}_val_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(val_metrics, f)

        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_{hal_type}_test_metrics.pkl'
        with open(out_name, 'wb') as f:
            test_metrics['trivial_acc'] = trivial_acc
            pickle.dump(test_metrics, f)
                
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == '__main__':              
    main()