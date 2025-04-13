import argparse
import os
from openai import OpenAI

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model, pca_programming_language_dataset, programming_language_dataset

OPENAI_API_KEY=os.environ['OPENAI_API_KEY']

def extract_code(c, extract_type='quotes'):
    if extract_type=='quotes':
        items = c.split("```")
        code = items[1]
        return code
    elif extract_type=='javascript':
        start_idx = c.find('javascript')
        if start_idx==-1:
            start_idx = c.find('python') # if javascript not there, try python
        if start_idx==-1:
            start_idx = c.find('class') # if python not there, try class
        if start_idx==-1:
            start_idx = c.find('function') # if class not there, try function
        
        c = c[start_idx:]
        items = c.split("```")
        code = items[0]
        return code
    
def control_language(program, controller, layers_to_control, coef, dest_lang, tokenizer, num_new_tokens=400):
    # template =  "Re-state the following program. Do not include an explanation. {program}."

    template = "Give a single, different re-writing of this program with the same function. "
    template += "The output will be judged by an expert in all programming languages. "
    template += "Do not include an explanation.\n\n```{program}```"
    
    prompt = template.format(program=program)
    formatted_prompt = controller.format_prompt(prompt, steer=True)
    
    whole_generation = controller.generate(formatted_prompt, 
                                     layers_to_control=layers_to_control, 
                                     control_coef=coef, 
                                     max_new_tokens=num_new_tokens, 
                                     do_sample=False
                                    )
    
    
    print("-"*50 + '\n')
    print("prompt:\n", formatted_prompt)
    generation = whole_generation[len(formatted_prompt):]
    print("generation (without prompt):\n", generation)
    generation = extract_code(generation, extract_type=dest_lang)
    print("extracted code:\n", generation)
        
    return generation, whole_generation

def detect_language(code):
    """
    Determines whether a given code snippet is Python or JavaScript based on content.
    """
    # Keywords and patterns specific to Python
    python_indicators = [
        "def ",  # Python function definition
        "print(",  # Python print function
        "elif ",  # Python elif keyword
        "self.",  # Python instance variable reference.
        "len(",
        "range(",
        "elif"
    ]

    # Keywords and patterns specific to JavaScript
    javascript_indicators = [
        "function ",  # JavaScript function definition
        "console.log(",  # JavaScript console output
        "var ",  # JavaScript variable declaration (ES5)
        "let ", 
        "const ",  # JavaScript variable declaration (ES6)
        "=>",  # JavaScript arrow function
        ".has(",
        "document.",  # JavaScript DOM manipulation
        "||",
        "&&",
        "null",
        "===",
        "if (",
        "else if",
        "while ("
    ]
            
    python_score = sum(1 for indicator in python_indicators if indicator in code)
    javascript_score = sum(1 for indicator in javascript_indicators if indicator in code)

    # Determine language based on scores
    if python_score > javascript_score:
        return "python"
    elif javascript_score > python_score:
        return "javascript"
    else:
        return "ambiguous"

def evaluate_language_translation(original, translation, dest_lang):
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    
    translation_lang = detect_language(translation)
    is_valid_translation = translation_lang==dest_lang
    
    if is_valid_translation:
        prompt="Rate the translation of the original program from 1 to 5. Do not reduce score for name changes."
        prompt+="Give your response in the format '{score}/5. {Reason}'."
        prompt+="-"*60
        prompt+=f"\nORIGINAL: {original}\n"
        prompt+="-"*60
        prompt+=f"\nTRANSLATION: {translation}"
        response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=125,
                temperature=0.0
            )

        overall_score = response.choices[0].message.content

        prompt="Does the translation have the same functionality as the original program and will it compile and run?"
        prompt+=" Simply state yes or no."
        prompt+="-"*60
        prompt+=f"\nORIGINAL: {original}\n"
        prompt+="-"*60
        prompt+=f"\nTRANSLATION: {translation}"
        response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=125,
                temperature=0.0
            )
        functionality_score = response.choices[0].message.content
        return overall_score, functionality_score
    else:
        print(f"Invalid translation. The translated code is: {translation_lang}.")
        return "1/5", 'No'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='python')
    parser.add_argument('--dest', type=str, default='javascript')
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3_8b_it')
    parser.add_argument('--coef', type=float)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
        
    
    source_lang = args.source
    dest_lang = args.dest
    control_method = args.control_method
    model_name = args.model_name
    coef = args.coef
    coef_not_provided = coef is None
    unsupervised = control_method in ['pca']
    
    if coef_not_provided and model_name=='gemma_2_9b_it':
        coef=9.0
        # assistant_tag = '<start_of_turn>model\n' 
    if coef_not_provided and model_name=='llama_3_8b_it':
        coef=0.5
        # assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
        
                        
    language_model, tokenizer = load_model(model=model_name)
    
    try:
        controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method
        )
        controller.load(f'{source_lang}_{dest_lang}', model_name, path='../directions/')
    except:
        print(f"No direction file found for {source_lang} to {dest_lang}.")
        concept_types = [source_lang, dest_lang]
    
        if unsupervised:
            data = pca_programming_language_dataset(concept_types, tokenizer)
        else:
            data = programming_language_dataset(concept_types, tokenizer)

        controllers = {}
        for concept_type in concept_types:

            other_type = [k for k in concept_types if k != concept_type][0]

            train_data = data[concept_type]['train']

            language_controller = NeuralController(
                language_model,
                tokenizer,
                rfm_iters=8,
                batch_size=2,
                control_method=control_method
            )

            language_controller.compute_directions(train_data['inputs'], train_data['labels'])

            controllers[concept_type] = language_controller

        for concept_type in concept_types:
            controller = controllers[concept_type]
            other_type = [k for k in concept_types if k!=concept_type][0]

            controller.save(concept=f'{concept_type}_{other_type}', model_name=model_name, path='../directions/')
        
        controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method
        )
        controller.load(f'{source_lang}_{dest_lang}', model_name, path='../directions/')
    
    
    from datasets import load_dataset
    huggingface_dataset = load_dataset("greengerong/leetcode")
    python_dataset = huggingface_dataset["train"]['python']
    
    programs = []
    for x in python_dataset:
        try: 
            programs.append(extract_code(x))
        except:
            pass
            # print("Program does not have quoted code:")
            # print(x + '\n')
            # print('-'*50 + '\n')
    
    n_to_translate = 100
    programs = programs[:n_to_translate]
    print(f'{len(programs)} total programs to translate') 
    
    if model_name=='gemma_2_9b_it':
        layers_to_control = list(range(-1, -41, -1))
    else:
        layers_to_control = list(range(-1, -31, -1))
        
    print(f"Control coef: {coef}, Layers: min {layers_to_control[0]}, max {layers_to_control[-1]}")

    overall_scores = []
    functionality_scores = []
    translations = []
    generations = []
    for program in programs:
        controlled_out, whole_generation = control_language(program, controller, layers_to_control, 
                                                            coef=coef, dest_lang=dest_lang, tokenizer=tokenizer)
        
        overall_score, functionality_score = evaluate_language_translation(program, controlled_out, dest_lang)
        
        overall_scores.append(overall_score)
        functionality_scores.append(functionality_score)

        translations.append(controlled_out)
        generations.append(whole_generation)
        
    out_name = f'./program_translations/{model_name}_{control_method}_{source_lang}_{dest_lang}_{coef}_translation_ratings.txt'
    with open(out_name, 'w') as f:
        for overall_score in overall_scores:
            f.write(f'{overall_score}\n')
            f.write('-'*50 + '\n')
    
    out_name = f'./program_translations/{model_name}_{control_method}_{source_lang}_{dest_lang}_{coef}_binary_ratings.txt'
    with open(out_name, 'w') as f:
        for functionality_score in functionality_scores:
            f.write(f'{functionality_score}\n')
            f.write('-'*50 + '\n')

    for i, translation in enumerate(translations):
        out_name = f'./program_translations/{model_name}_translations/{model_name}_{control_method}_{source_lang}_{dest_lang}_{coef}_translation_{i}.txt'
        with open(out_name, 'w') as f:
            f.write(f'{translation}\n')
            f.write('-'*50 + '\n')
    
    for i, generation in enumerate(generations):
        out_name = f'./program_translations/{model_name}_generations/{model_name}_{control_method}_{source_lang}_{dest_lang}_{coef}_generation_{i}.txt'
        with open(out_name, 'w') as f:
            f.write(f'{generation}\n')
            f.write('-'*50 + '\n')

if __name__ == '__main__':              
    main()