import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model
import re
import json
import numpy as np
import random
random.seed(0)

import pickle
from tqdm import tqdm
import gc

from bs4 import BeautifulSoup

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

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
    
    # Return slices up to max_n instead of single index
    return paired_data, paired_labels

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

def get_fava_annotated_data(tokenizer):
    # Specify the path to your JSON file
    file_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/annotations.json'

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    inputs = []
    labels = []
    for d in data:
        s = d['annotated']
        i, label = modify(s)
        chat = [
            {'role':'user',
             'content':i
            },
        ]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False))
        labels.append(label)
    return inputs, labels

def get_fava_training_data(tokenizer, unsupervised=False, max_n=10000):
    # Load training data from FAVA dataset
    from datasets import load_dataset
    ds = load_dataset("fava-uw/fava-data")
    completions = ds['train']['completion']

    # Try to load saved shuffle indices
    shuffle_indices_path = f'{NEURAL_CONTROLLERS_DIR}/results/fava_annotated_results/fava_shuffle_indices.pkl'
    try:
        with open(shuffle_indices_path, 'rb') as f:
            indices = pickle.load(f)
            completions = [completions[i] for i in indices]
    except:
        # If loading fails, create new shuffle indices
        random.seed(0)
        indices = list(range(len(completions)))
        
        random.shuffle(indices)
        completions = [completions[i] for i in indices]
        
        # Save the shuffle indices
        os.makedirs(os.path.dirname(shuffle_indices_path), exist_ok=True)
        with open(shuffle_indices_path, 'wb') as f:
            pickle.dump(indices, f)

    inputs = []
    labels = []
    for d in completions:
        i, label = modify(d)
        chat = [{'role':'user','content':i}]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False))
        labels.append(label)

    if unsupervised:
        # Split data and create pairs for training
        pos_train = [x for x, y in zip(inputs, labels) if y == 1]
        neg_train = [x for x, y in zip(inputs, labels) if y == 0]
        train_inputs, train_labels = create_paired_data(pos_train, neg_train)

        print("Number of positive examples:", len(pos_train))
        print("Number of negative examples:", len(neg_train))
        print("Shape of train_inputs after create_paired_data:", np.array(train_inputs).shape)

        # Flatten training data
        train_inputs = np.concatenate(train_inputs).tolist()
        train_labels = np.concatenate(train_labels).tolist()
    else:
        train_inputs = inputs
        train_labels = labels
    
    return train_inputs[:max_n], train_labels[:max_n]


def get_splits(n_train, n_val, n_total, n_seeds):
    """
    Create splits for training, validation, and testing datasets.
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/fava_annotated_results'
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
        np.random.seed(seed)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_train', type=int, default=0)
    parser.add_argument('--n_val', type=int, default=360)
    parser.add_argument('--n_components', type=int, default=3)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_train = args.n_train
    n_val = args.n_val
    n_seeds = args.n_seeds
    n_components = args.n_components
    
    unsupervised = control_method in ['pca']
    
    if control_method not in ['rfm']:
        n_components = 1
    
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
    elif control_method == 'rfm':
        use_rfm = True
    else:
        use_rfm = False
                        
            
    if control_method=='logistic_concat':
        use_logistic=True
        
    print("Num components:", n_components)
                        
    language_model, tokenizer = load_model(model=model_name)
    inputs, labels = get_fava_annotated_data(tokenizer)


    splits = get_splits(n_train, n_val, len(labels), n_seeds)

    seed_list = range(n_seeds)
    for seed in tqdm(seed_list):
        split = splits[seed]
        
        controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method,
            batch_size=4,
            rfm_iters=5
        )
        
        val_inputs = [inputs[i] for i in split['val_indices']]
        test_inputs = [inputs[i] for i in split['test_indices']]
        
        val_labels = [labels[i] for i in split['val_indices']]
        test_labels = [labels[i] for i in split['test_indices']]

        nval = len(val_inputs)
        ntest = len(test_inputs)
        results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/fava_annotated_results'
        os.makedirs(results_dir, exist_ok=True)

        out_name = f'{results_dir}/{control_method}_data_counts_seed_{seed}.pkl'
        with open(out_name, 'wb') as f:
            counts = {'val':nval, 'test':ntest}
            pickle.dump(counts, f)
        
        try:
            controller.load(concept='fava_training', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            train_inputs, train_labels = get_fava_training_data(tokenizer, unsupervised=unsupervised)

            # Create new controller and compute directions
            controller = NeuralController(
                    language_model,
                tokenizer,
                n_components=5,
                control_method=control_method,
                batch_size=2,
                rfm_iters=5
            )
            controller.compute_directions(train_inputs, train_labels)
            controller.save(concept='fava_training', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
              
        assert(len(val_inputs) > 0)
        assert(len(test_inputs) > 0)
        
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
            
        del controller
        gc.collect()

if __name__ == '__main__':              
    main()