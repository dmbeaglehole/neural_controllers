import argparse
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model

import json
import numpy as np
import pickle
import random
random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

def get_data(controller):
    # Function to load a JSON file
    def load_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    benign_val_path = f'{NEURAL_CONTROLLERS_DIR}/data/agentharm/benign_behaviors_validation.json'
    benign_test_path = f'{NEURAL_CONTROLLERS_DIR}/data/agentharm/benign_behaviors_test_public.json'
    harmful_val_path = f'{NEURAL_CONTROLLERS_DIR}/data/agentharm/harmful_behaviors_validation.json'
    harmful_test_path = f'{NEURAL_CONTROLLERS_DIR}/data/agentharm/harmful_behaviors_test_public.json'

    # Load the files
    benign_val_data_ = load_json(benign_val_path)['behaviors']
    benign_test_data_ = load_json(benign_test_path)['behaviors']
    harmful_val_data_ = load_json(harmful_val_path)['behaviors']
    harmful_test_data_ = load_json(harmful_test_path)['behaviors']

    benign_val_data = [entry['prompt'] for entry in benign_val_data_]
    benign_test_data = [entry['prompt'] for entry in benign_test_data_]
    harmful_val_data = [entry['prompt'] for entry in harmful_val_data_]
    harmful_test_data = [entry['prompt'] for entry in harmful_test_data_]

    # Example: Print counts or process further as needed
    print(f"Number of benign validation prompts: {len(benign_val_data)}")
    print(f"Number of benign test prompts: {len(benign_test_data)}")
    print(f"Number of harmful validation prompts: {len(harmful_val_data)}")
    print(f"Number of harmful test prompts: {len(harmful_test_data)}")

    test_inputs = benign_test_data + harmful_test_data
    test_inputs = [controller.format_prompt(x) for x in test_inputs]
    test_labels = [1]*len(benign_test_data) + [0]*len(harmful_test_data)
    
    val_inputs = benign_val_data + harmful_val_data
    val_labels = [1]*len(benign_val_data) + [0]*len(harmful_val_data)
    val_inputs = [controller.format_prompt(x) for x in val_inputs]
    
    return val_inputs, val_labels, test_inputs, test_labels

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it', choices=['llama_3_8b_it', 'llama_3.3_70b_4bit_it'])
    parser.add_argument('--n_components', type=int, default=2)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    
    if control_method not in ['rfm']:
        n_components=1
        
    use_logistic=(control_method=='logistic')
    unsupervised = control_method=='pca'
    
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
        batch_size=2,
        rfm_iters=10
    )
    
    val_inputs, val_labels, test_inputs, test_labels = get_data(controller)
    print("len(val_inputs)", len(val_inputs), "len(test_inputs)", len(test_inputs))

    
    comb = list(zip(val_inputs, val_labels))

    random.seed(0)
    random.shuffle(comb)

    val_inputs, val_labels = zip(*comb)
    val_inputs = list(val_inputs)
    val_labels = list(val_labels)

    nval = len(val_inputs)
    ntest = len(test_inputs)
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/agentharm_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/{control_method}_data_counts.pkl'
    with open(out_name, 'wb') as f:
        counts = {'val':nval, 'test':ntest}
        pickle.dump(counts, f)
    
    if unsupervised:
        # Create paired data for validation and test sets
        train_inputs, train_labels = create_paired_data(val_inputs, val_labels)        

        train_inputs = np.concatenate(train_inputs).tolist()
        train_labels = np.concatenate(train_labels).tolist()
    else:
        train_inputs = val_inputs
        train_labels = val_labels
    
    try:
        controller.load(concept='agentharm', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
    except:
        controller.compute_directions(train_inputs, train_labels)
        controller.save(concept='agentharm', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
            
            
    assert(len(val_inputs)==len(val_labels))
    assert(len(test_inputs)==len(test_labels))
    print("Val inputs:", len(val_inputs), "Test inputs:", len(test_inputs), 
          "Val labels:", len(val_labels), "Test labels:", len(test_labels))
    
    val_metrics, test_metrics, _ = controller.evaluate_directions(
                                    val_inputs, val_labels,
                                    test_inputs, test_labels,
                                    n_components=n_components,
                                    use_logistic=use_logistic,
                                    use_rfm=use_rfm,
                                    unsupervised=unsupervised
                                )

    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/agentharm_results'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/{model_name}_{original_control_method}_val_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(val_metrics, f)

    out_name = f'{results_dir}/{model_name}_{original_control_method}_test_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(test_metrics, f)

if __name__ == '__main__':              
    main()
    