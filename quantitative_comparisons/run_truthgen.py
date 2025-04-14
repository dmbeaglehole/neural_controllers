import argparse
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
from datasets import load_dataset
import random
random.seed(0)

def get_raw_data():
    """Get raw data from the dataset without any processing."""
    ds = load_dataset("wwbrannon/TruthGen")
    true = ds['train']['truth']
    false = ds['train']['falsehood']
    return true, false

def process_data(true, false, controller, indices=None):
    """Process data based on indices and whether it's supervised or unsupervised."""
    # If indices provided, first select the specified samples
    all_data = true + false
    selected_data = [all_data[i] for i in indices]
    selected_labels = [1 if i < len(true) else 0 for i in indices]
    
    inputs = [controller.format_prompt(p) for p in selected_data]
    return inputs, selected_labels

def get_splits_fixed_train(n_train, n_val, n_total, n_seeds):
    """Generate splits with randomized train, val, and test sets for each seed."""
    out_name = f'./truthgen_results/unified_splits_ntrain_{n_train}_nval_{n_val}_ntotal_{n_total}_nseeds_{n_seeds}.pkl'
    
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
        # Randomize train indices for each seed
        train_indices = np.random.choice(indices, size=n_train, replace=False)
        remaining_indices = np.setdiff1d(indices, train_indices)
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=1500)
    parser.add_argument('--n_components', type=int, default=2)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_train = args.n_train
    n_val = args.n_val
    n_seeds = args.n_seeds
    n_components = args.n_components
    
    if control_method not in ['rfm']:
        n_components=1
        
    use_logistic=(control_method=='logistic')
    unsupervised=(control_method in ['pca'])
    
    original_control_method = str(control_method)
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
        
    
    print("Num components:", n_components)
                        
    language_model, tokenizer = load_model(model=model_name)
    controller = NeuralController(
        language_model,
        tokenizer,
        control_method=control_method,
        rfm_iters=8,
        batch_size=2
    )  
    
    # Get raw data
    true, false = get_raw_data()
    n_total = len(true) + len(false)
    
    # Get unified splits
    splits = get_splits_fixed_train(n_train, n_val, n_total, n_seeds)
    
    for seed in tqdm(range(n_seeds)):
        split = splits[seed]
        
        # Process data for current split
        train_inputs, train_labels = process_data(true, false, controller, split['train_indices'])
        val_inputs, val_labels = process_data(true, false, controller, split['val_indices'])
        test_inputs, test_labels = process_data(true, false, controller, split['test_indices'])

        ntrain  = len(train_inputs)
        nval = len(val_inputs)
        ntest = len(test_inputs)
        out_name = f'./truthgen_results/{control_method}_data_counts_seed_{seed}.pkl'
        with open(out_name, 'wb') as f:
            counts = {'train':ntrain, 'val':nval, 'test':ntest}
            pickle.dump(counts, f)

        # Create pairs if unsupervised is True
        if unsupervised:
            train_pairs, train_pair_labels = create_positive_negative_pairs(
                inputs=train_inputs,
                labels=train_labels,
                max_pairs=None  # This will use all possible pairs
            )
            train_inputs = np.concatenate(train_pairs).tolist()
            train_labels = np.concatenate(train_pair_labels).tolist()
        
        assert(len(train_inputs)>0)
        assert(len(val_inputs)>0)
        assert(len(test_inputs)>0)
        
        # Calculate trivial accuracy using all data
        all_labels = [1]*len(true) + [0]*len(false)
        trivial_acc = max((sum(all_labels)/len(all_labels)), 1-(sum(all_labels)/len(all_labels)))*100
        
        try:
            controller.load(concept='truthgen_large_seed_'+str(seed), model_name=model_name, path='../directions/')
        except:
            controller.compute_directions(train_inputs, train_labels)
            controller.save(concept='truthgen_large_seed_'+str(seed), model_name=model_name, path='../directions/')

        
        val_metrics, test_metrics, _ = controller.evaluate_directions(
            val_inputs, val_labels,
            test_inputs, test_labels,
            n_components=n_components,
            use_logistic=use_logistic,
            use_rfm=use_rfm,
            unsupervised=unsupervised
        )
        
        out_name = f'./truthgen_results/{model_name}_{original_control_method}_seed_{seed}_val_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(val_metrics, f)
            
        out_name = f'./truthgen_results/{model_name}_{original_control_method}_seed_{seed}_test_metrics.pkl'
        with open(out_name, 'wb') as f:
            test_metrics['trivial_acc'] = trivial_acc
            pickle.dump(test_metrics, f)
                     
        gc.collect()
        torch.cuda.empty_cache()
        
if __name__ == '__main__':              
    main()