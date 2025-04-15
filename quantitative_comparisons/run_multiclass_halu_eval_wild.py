import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model

import json
import numpy as np
import torch
import pickle
from tqdm import tqdm
import gc
import random
random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

TYPE_MAP = {
    'confused': 'confused / erroneous queries',
    'inappropriate': 'inappropriate content',
    'complex': 'complex reasoning',
    'out-of-scope': 'out-of-scope information',
    'beyond-modality': 'beyond-modality interaction',
    'other': 'other types'
}


def read_json_to_list(file_path):
    """
    Reads a JSON file and returns its contents as a list of dictionaries.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if isinstance(data, list):
        return data
    else:
        raise ValueError("JSON content is not a list.")
        
def create_paired_data(pos_examples, neg_examples):
    # Ensure we have enough examples of each class
    min_len = min(len(pos_examples), len(neg_examples))
    max_len = max(len(pos_examples), len(neg_examples))
    
    # Randomly sample with replacement if we need more examples
    if len(pos_examples) < max_len:
        pos_examples = pos_examples + random.choices(pos_examples, k=max_len-len(pos_examples))
    if len(neg_examples) < max_len:
        neg_examples = neg_examples + random.choices(neg_examples, k=max_len-len(neg_examples))
        
    # Create pairs
    paired_data = list(zip(pos_examples, neg_examples))
    paired_data = [list(x) for x in paired_data]
    
    # Create corresponding labels
    paired_labels = [[1, 0] for _ in range(len(paired_data))]
    
    return paired_data, paired_labels


def get_multiclass_halu_eval_wild_data(controller):
    data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval_wild/HaluEval_Wild_6types.json'
    entries = read_json_to_list(data_path)
    
    # Get unique classes from TYPE_MAP
    classes = list(TYPE_MAP.values())
    num_classes = len(classes)

    print("classes", classes)
    
    inputs = []
    labels = []
    
    for entry in entries:
        query = entry['query']
        qtype = entry['query_type']
        inputs.append(controller.format_prompt(query))
        
        # Create one-hot encoded label
        label = [0] * num_classes
        class_idx = classes.index(qtype)
        label[class_idx] = 1
        labels.append(torch.tensor(label))
    
    labels = torch.stack(labels).reshape(-1, num_classes).cuda().float()

    return inputs, labels


def get_splits(n_train, n_val, n_total, n_seeds):
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_wild_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/splits_ntrain_{n_train}_nval_{n_val}_ntotal_{n_total}_nseeds_{n_seeds}.pkl'
    
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
        return splits
    except:
        pass

    splits = []
    indices = np.arange(n_total)
    
    for seed in range(n_seeds):
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Shuffle indices for this seed
        shuffled_indices = indices.copy()
        np.random.shuffle(shuffled_indices)
        
        # Sample train indices
        train_indices = shuffled_indices[:n_train]
        remaining_indices = shuffled_indices[n_train:]
        
        # Sample validation indices from remaining
        val_indices = np.random.choice(remaining_indices, size=n_val, replace=False)
        test_indices = np.setdiff1d(remaining_indices, val_indices)
        
        splits.append({
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        })

    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)
        
    return splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it', choices=['llama_3_8b_it', 'llama_3.3_70b_4bit_it'])
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_train', type=int, default=300)
    parser.add_argument('--n_val', type=int, default=250)
    parser.add_argument('--n_components', type=int, default=6)
    parser.add_argument('--rfm_iters', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")  

    control_method = args.control_method
    model_name = args.model_name
    n_train = args.n_train
    n_val = args.n_val
    n_seeds = args.n_seeds
    n_components = args.n_components
    
    # Set unsupervised flag based on control method
    unsupervised = (control_method == 'pca')
        
    use_logistic = (control_method == 'logistic')
    
    original_control_method = str(control_method)
    if control_method == 'logistic_rfm':
        control_method = 'logistic'
        use_rfm = True
    elif control_method == 'linear_rfm':
        control_method = 'linear'
        use_rfm = True
    elif control_method == 'rfm_linear':
        control_method = 'rfm'
        use_rfm = False
    elif control_method == 'rfm_logistic':
        control_method = 'rfm'
        use_logistic = True
        use_rfm = False
    elif control_method == 'rfm':
        use_rfm = True
    else:
        use_rfm = False
        
    if control_method=='logistic_concat':
        use_logistic=True
        
    print("Num components:", n_components)
                        
    language_model, tokenizer = load_model(model=model_name)
    controller = NeuralController(
        language_model,
        tokenizer,
        control_method=control_method,
        rfm_iters=args.rfm_iters,
        batch_size=2,
        n_components=8
    )  
    
    inputs, labels = get_multiclass_halu_eval_wild_data(controller)

    splits = get_splits(n_train, n_val, len(labels), n_seeds)
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = range(n_seeds)
        
    for seed in tqdm(seeds):
        split = splits[seed]
        
        train_inputs = [inputs[i] for i in split['train_indices']]
        val_inputs = [inputs[i] for i in split['val_indices']]
        test_inputs = [inputs[i] for i in split['test_indices']]
        
        train_labels = labels[split['train_indices']]
        val_labels = labels[split['val_indices']]
        test_labels = labels[split['test_indices']]

        ntrain  = len(train_inputs)
        nval = len(val_inputs)
        ntest = len(test_inputs)
        out_name = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_wild_results/{control_method}_data_counts_seed_{seed}.pkl'
        with open(out_name, 'wb') as f:
            counts = {'train':ntrain, 'val':nval, 'test':ntest}
            pickle.dump(counts, f)
        
        if unsupervised:
            # Split train data into positive and negative examples
            pos_train = [train_inputs[i] for i, label in enumerate(train_labels) if label == 1]
            neg_train = [train_inputs[i] for i, label in enumerate(train_labels) if label == 0]
            
            # Create paired training data
            train_inputs_processed, train_labels_processed = create_paired_data(pos_train, neg_train)
            train_inputs_processed = np.concatenate(train_inputs_processed).tolist()
            train_labels_processed = np.concatenate(train_labels_processed).tolist()
        else:
            train_inputs_processed = train_inputs
            train_labels_processed = train_labels
        
        assert(len(train_inputs_processed) > 0)
        assert(len(val_inputs) > 0)
        assert(len(test_inputs) > 0)
        
        print("train shapes:", len(train_inputs_processed), train_labels_processed.shape)
        try:
            controller.load(concept=f'halu_eval_wild_multiclass_seed_{seed}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            controller.compute_directions(train_inputs_processed, train_labels_processed)
                
            controller.save(concept=f'halu_eval_wild_multiclass_seed_{seed}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

        val_metrics, test_metrics, _ = controller.evaluate_directions(
            val_inputs, val_labels,
            test_inputs, test_labels,
            n_components=n_components,
            use_logistic=use_logistic,
            use_rfm=use_rfm,
            unsupervised=unsupervised,
        )
        
        results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_wild_results'
        os.makedirs(results_dir, exist_ok=True)
        out_name = f'{results_dir}/multiclass_{model_name}_{original_control_method}_seed_{seed}_val_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(val_metrics, f)
            
        out_name = f'{results_dir}/multiclass_{model_name}_{original_control_method}_seed_{seed}_test_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(test_metrics, f)
                     
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == '__main__':              
    main()