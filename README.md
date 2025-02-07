# Neural Controllers

A Python library for implementing neural controllers with decoder-only Large Language Models (LLMs), as described in [our paper](https://arxiv.org/abs/2502.03708). Our API allows you to steer the output of language models toward desired concepts and generate lightweight detectors for arbitrary pre-defined concepts. The approach can be implemented with any decoder-only LLM, with demonstrated success on models like instruction-tuned [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Llama-3.3-70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), and [Gemma-2-9B](https://huggingface.co/google/gemma-2-9b-it). 

We choose Recursive Feature Machines (RFMs) as our nonlinear predictor at every layer and (often) as our aggregation model. These models are simple, lightweight kernel machines. We also include functionality for our aggregation technology with other baselines including linear/logistic probing and contrastive methods like PCA and difference-in-means. The RFM library can be installed from github (https://github.com/aradha/recursive_feature_machines) using the following command:
```bash
pip install git+https://github.com/aradha/recursive_feature_machines.git@pip_install
```


See the notebooks folder for examples of steering:
- Style transfer capabilities (e.g., English to Shakespearean or Poetic)
- Language transfer capabilities (e.g., English to Spanish, Mandarin to English)
- Harmful content generation (e.g. generating social security numbers)
- Science subjects (e.g. Biology vs. Classical Mechanics)
- Political leaning (e.g. Democratic vs. Republican)
- Name disambiguation (e.g. Cam vs. Isaac Newton)
- Word meaning disambiguation (e.g. (Financial) Bank vs. Riverbank)
- and more...

## Minimum working requirements

- Python 3.10.15
- PyTorch 2.4.0+cu118
- Transformers 4.47.0
- Datasets 3.1.0
- NumPy 1.26.4
- tqdm
- torchmetrics
- scikit-learn
- RFM (https://github.com/aradha/recursive_feature_machines)
- Access to decoder-only LLM weights, such as Llama-3.1-8B-it and Gemma-2-9B-it.

## Our approach
![Neural Controller methodology](figures/main_figure.png)
Methodology for (A) detecting and (B) steering concepts in language models by aggregating layer-wise predictors. (C) Examples of steered generations toward one or multiple concepts. These include harmfulness, Shakespearean/Poetic English, and dishonesty. The produced SSNs in the first box are valid according to online verification services.

## Usage

### Basic Steering Example: Shakespeare

```python
from neural_controllers import NeuralController
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize tokenizer and model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
language_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

# Create neural controller
controller = NeuralController(
    language_model,
    tokenizer,
    rfm_iters=8,
    batch_size=2,
    n_components=5,
    control_method='rfm'
)

# Load pre-trained directions
controller.load(concept=f'shakespeare', 
               model_name='llama_3_8b_it', 
               path='../directions/')

# Generate controlled text
prompt = controller.format_prompt("What can I do to treat flu symptoms?")
controlled_output = controller.generate(
    prompt,
    layers_to_control=list(range(-1, -31, -1)),
    control_coef=0.5,
    max_new_tokens=150
)
```

### Basic Detection Example: Toxicity from the ToxicChat benchmark [1]

```python
from neural_controllers import NeuralController
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Initialize tokenizer and model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
language_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

# Create neural controller
controller = NeuralController(
    language_model,
    tokenizer,
    rfm_iters=8,
    batch_size=2,
    n_components=5,
    control_method='rfm'
)

def get_data(controller):
    # Load the dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")
    
    # Extract and preprocess inputs
    all_train_inputs = [x['user_input'] for x in ds['train']]
    test_inputs = [x['user_input'] for x in ds['test']]

    # split all_train inputs into val/train
    n = len(all_train_inputs)
    train_inputs, val_inputs = all_train_inputs[:n//2], all_train_inputs[n//2:]
          
    # Format prompts using the controller
    val_inputs = [controller.format_prompt(x) for x in val_inputs]
    train_inputs = [controller.format_prompt(x) for x in train_inputs]
    test_inputs = [controller.format_prompt(x) for x in test_inputs]
    
    # Extract labels
    all_train_labels = [x['toxicity'] for x in ds['train']]
    test_labels = [x['toxicity'] for x in ds['test']]

    # split all_train labels into val/train
    train_labels, val_labels = all_train_labels[:n//2], all_train_labels[n//2:]
    
    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels

train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = get_data(controller)
controller.compute_directions(train_inputs, train_labels)

val_metrics, test_metrics, _ = controller.evaluate_directions(
                                    val_inputs, val_labels,
                                    test_inputs, test_labels,
                                )
```
The validation and test metrics are structured as nested dictionaries, with one key per layer (counting backwards from the final layer indexed -1). We also include the aggregated scores. E.g. for Llama-3.1-8B with 31 blocks:

```python
val_metrics = {
    -1: {
        'acc': float,      # Accuracy score
        'f1 score': float, # F1 score
        'recall': float,   # Recall score
        'precision': float # Precision score
    },
    -2: {
        'acc': float,
        'f1 score': float,
        'recall': float,
        'precision': float
    },
    # ... continues through layer -31
    
    'aggregated': {
        'acc': float,      # Aggregated accuracy across layers
        'f1 score': float, # Aggregated F1 score
        'recall': float,   # Aggregated recall
        'precision': float # Aggregated precision
    }
}
```
## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{beaglehole2025aggregateconquerdetectingsteering,
      title={Aggregate and conquer: detecting and steering LLM concepts by combining nonlinear predictors over multiple layers}, 
      author={Daniel Beaglehole and Adityanarayanan Radhakrishnan and Enric Boix-Adserà and Mikhail Belkin},
      year={2025},
      eprint={2502.03708},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.03708}, 
}
```

## References
[1]: Lin, Z., Wang, Z., Tong, Y., Wang, Y., Guo, Y., Wang, Y., & Shang, J. (2023). ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation. *arXiv preprint arXiv:2310.17389*
