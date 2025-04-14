import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model
import numpy as np

import pickle

from datasets import load_dataset
import random
random.seed(0)
from datasets import load_dataset

import os   
NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
def create_positive_negative_pairs(inputs, labels, max_pairs=None):
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


def get_data(controller):
    # Load the dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")

    train_inputs = [x['user_input'] for x in ds['train']]
    test_inputs = [x['user_input'] for x in ds['test']]
        
    # Format prompts using the controller
    train_inputs = [controller.format_prompt(x) for x in train_inputs]
    test_inputs = [controller.format_prompt(x) for x in test_inputs]
    
    # Extract labels
    train_labels = [x['toxicity'] for x in ds['train']]
    test_labels = [x['toxicity'] for x in ds['test']]
    
    return train_inputs, train_labels, test_inputs, test_labels


def get_splits_fixed_train(n_val, n_total, n_seeds, unsupervised=False):
    # Use same base name for both supervised and unsupervised to ensure same splits
    out_name = f'./toxic_chat_results/full_splits_fixed_train_nval_{n_val}_ntotal_{n_total}_nseeds_{n_seeds}.pkl'
    
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        pass

    splits = []
    indices = np.arange(n_total)

    # Fix the train indices
    np.random.seed(0)  # Fixed seed for reproducibility
    indices = np.arange(n_total)
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        val_indices = np.random.choice(indices, size=n_val, replace=False)
        train_indices = np.setdiff1d(indices, val_indices)

        splits.append({
            'val_indices': val_indices,
            'train_indices': train_indices
        })

    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_components', type=int, default=1)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
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
        batch_size=1
    )  
    controller.name = model_name
    # Get base data first
    train_inputs, train_labels, test_inputs, test_labels = get_data(controller) 

    print('train')
    print("train inputs", len(train_inputs), "train labels", len(train_labels))
    print("test inputs", len(test_inputs), "test labels", len(test_labels))
    
    # Split training data into train and validation sets (60/40 split)
    n_val = int(len(train_labels) * 0.4) 
    splits = get_splits_fixed_train(n_val, len(train_labels), n_seeds)
    
    seed = args.seed
    split = splits[seed]
        
    # Get the base split indices first
    val_indices = split['val_indices']
    train_indices = split['train_indices']
        
    # Split the supervised data
    val_inputs = [train_inputs[i] for i in val_indices]
    train_inputs_split_base = [train_inputs[i] for i in train_indices]
    
    val_labels = [train_labels[i] for i in val_indices]
    train_labels_split_base = [train_labels[i] for i in train_indices]
    
    # If unsupervised, create pairs from the split data
    if unsupervised:
        train_pairs, train_pair_labels = create_positive_negative_pairs(train_inputs_split_base, train_labels_split_base)
        
        train_inputs_split = np.concatenate(train_pairs).tolist()
        train_labels_split = np.concatenate(train_pair_labels).tolist()
    else:
        train_inputs_split = train_inputs_split_base
        train_labels_split = train_labels_split_base

    try:
        controller.load(concept='toxic_chat_full_seed_'+str(seed), model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
    except:
        controller.compute_directions(train_inputs_split, train_labels_split)
        controller.save(concept='toxic_chat_full_seed_'+str(seed), model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')


    ntrain  = len(train_inputs_split)
    nval = len(val_inputs)
    ntest = len(test_inputs)
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/toxic_chat_results'
    os.makedirs(results_dir, exist_ok=True)
    
    out_name = f'{results_dir}/{control_method}_data_counts_seed_{seed}.pkl'
    with open(out_name, 'wb') as f:
        counts = {'train':ntrain, 'val':nval, 'test':ntest}
        pickle.dump(counts, f)
        
    val_metrics, test_metrics, _ = controller.evaluate_directions(
        val_inputs, val_labels,
        test_inputs, test_labels,
        n_components=n_components,
        use_logistic=use_logistic,
        use_rfm=use_rfm,
        unsupervised=unsupervised
    )
    
    out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_val_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(val_metrics, f)
        
    out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_test_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(test_metrics, f)
        
            
if __name__ == '__main__':              
    main()