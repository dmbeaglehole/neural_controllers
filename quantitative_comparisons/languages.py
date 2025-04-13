import os
import time
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from neural_controllers import NeuralController
from utils import load_model, pca_language_dataset, supervised_language_dataset
import re

OPENAI_API_KEY=os.environ['OPENAI_API_KEY']


def control_language_llama(sentence, controller, tokenizer, assistant_tag, num_new_tokens=80, coef=0.4):
    template =  "Give another version of the following sentence with the same meaning: '{sentence}'. Write the version in quotes."
    prompt = template.format(sentence=sentence)
    print("Prompt:",repr(prompt))
    
    formatted_prompt = controller.format_prompt(prompt, steer=True)
    
    whole_generation = controller.generate(formatted_prompt, 
                                     layers_to_control=list(range(-1, -31, -1)), 
                                     control_coef=coef, 
                                     max_new_tokens=num_new_tokens, 
                                     do_sample=False
                                    )
    
    generation = whole_generation[len(formatted_prompt):]
    try:
        generation = generation.split('"')[1]
    except:
        if '。' in generation:
            generation = generation.split('。')[0] + "。"

        if '?' in generation:
            generation = generation.split('?')[0] + "?"

        generation = generation.strip(' ')
        generation = generation.strip('\n')

        if assistant_tag in generation:
            generation = generation.replace(assistant_tag, '')
        
    return generation, whole_generation

def control_language_gemma(sentence, controller, tokenizer, assistant_tag, num_new_tokens=80, coef=9.0):
    template =  "Give another version of the following sentence with the same meaning: '{sentence}'. Write the version in quotes."
    prompt = template.format(sentence=sentence)
    print("Prompt:",repr(prompt))
    
    formatted_prompt = controller.format_prompt(prompt, steer=True)

    whole_generation = controller.generate(formatted_prompt, 
                                     layers_to_control=list(range(-1, -41, -1)), 
                                     control_coef=coef, 
                                     max_new_tokens=num_new_tokens, 
                                     do_sample=False
                                    )
    
    generation = whole_generation[len(formatted_prompt):]
    
    try:
        generation = generation.split('"')[1]
    except:
        if '。' in generation:
            generation = generation.split('。')[0] + "。"

        if '?' in generation:
            generation = generation.split('?')[0] + "?"

        generation = generation.strip(' ')
        generation = re.sub(r'\n.*\n', '\n', generation)
        generation = re.sub(r'\s+', ' ', generation)
        generation = generation.strip('\n')

        end_token = '<end_of_turn>'
        if end_token in generation:
            end_idx = generation.find(end_token)
            generation = generation[:end_idx]

        if assistant_tag in generation:
            generation = generation.replace(assistant_tag, '')
        
    return generation, whole_generation

def evaluate_language_translation(original, translation, dest_lang, assistant_tag):
    
    if assistant_tag in translation:
        translation = translation.replace(assistant_tag, '')
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt="What language are the original and translations in? Give your answer in the format '{Original language}, {Translation language}'. "
    prompt+="If the translation is empty, assume Language 2 is English.\n\n"
    prompt+="-"*40
    prompt+=f"\nORIGINAL: {original}\n"
    prompt+="-"*40
    prompt+=f"\nTRANSLATION: {translation}"


    @retry(
        stop=stop_after_attempt(12), 
        wait=wait_exponential(min=1, max=1024),
    )
    def api_call():
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=25,
            temperature=0
        )

    response = api_call()
    
    verification_out = response.choices[0].message.content
    verification_out = verification_out.replace('{','')
    verification_out = verification_out.replace('}','')
    langs = verification_out.split(',')
    langs = [lang.strip(' ') for lang in langs]
    is_valid_translation = langs[0]!=langs[1]
        
    if is_valid_translation:
        prompt="Rate the translation of the original sentence from 1 to 4 according to the following rubric.\n\n"
        prompt+="Score of 1: The translation is mostly inaccurate or unrelated.\n"
        prompt+="Score of 2: The translation is slightly unrelated to the original.\n"
        prompt+="Score of 3: The translation has mostly the same meaning as the original.\n"
        prompt+="Score of 4: The translation has the same meaning as the original.\n\n"
        prompt+="Give your response in the format '{score}/4.' Do not penalize awkward or excessive wording. "
        prompt+="If the translation is empty, give a score of 0.\n"
        prompt+="-"*40
        prompt+=f"\nORIGINAL: {original}\n"
        prompt+="-"*40
        prompt+=f"\nTRANSLATION: {translation}"
        
        
        @retry(
            stop=stop_after_attempt(12),
            wait=wait_exponential(min=1, max=1024), 
        )
        def api_call():
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0
            )

        response = api_call()

                 
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    else:
        print(f"Invalid translation. The languages are: {langs}.")
        return "0/4"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='english')
    parser.add_argument('--dest', type=str, default='chinese')
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3_8b_it')
    parser.add_argument('--coef', type=float)
    args = parser.parse_args()
    
    source_lang = args.source
    dest_lang = args.dest
    control_method = args.control_method
    model_name = args.model_name
    coef = args.coef
    coef_not_provided = coef is None
    unsupervised = control_method in ['pca']
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
        data_dir = "../data/languages"
        concept_types = [source_lang, dest_lang]
    
        if unsupervised:
            data = pca_language_dataset(data_dir, concept_types, tokenizer)
        else:
            data = supervised_language_dataset(data_dir, concept_types, tokenizer)

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
    
    if model_name=='llama_3_8b_it':
        control_language = control_language_llama
        assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
        if coef_not_provided:
            coef=0.5
    elif model_name=='gemma_2_9b_it':
        control_language = control_language_gemma
        assistant_tag = '<start_of_turn>model\n' 
        if coef_not_provided:
            coef=9.0
    
    with open(f'./language_translations/{source_lang}_sentences.txt', 'r') as f:
        source_sentences = f.readlines()
    source_sentences = [x.replace('\n','').strip(' ') for x in source_sentences]
    
    print(f"Control coef: {coef}")
        
    reviews = []
    translations = []
    generations = []
    for i, sentence in enumerate(source_sentences):
        print(f"Sentence {i+1} out of {len(source_sentences)}")
        controlled_out, whole_generation = control_language(sentence, controller, tokenizer, assistant_tag, num_new_tokens=80, coef=coef)
        review = evaluate_language_translation(sentence, controlled_out, dest_lang, assistant_tag)
        
        generations.append(whole_generation)
        reviews.append(review)
        translations.append(controlled_out)
        
    out_name = f'./language_translations/{model_name}_{control_method}_{source_lang}_{dest_lang}_{coef}_translation_ratings.txt'
    with open(out_name, 'w') as f:
        for review in reviews:
            f.write(f'{review}\n')
            f.write('-'*40 + '\n')

    out_name = f'./language_translations/{model_name}_translations/{model_name}_{control_method}_{source_lang}_{dest_lang}_{coef}_translations.txt'
    with open(out_name, 'w') as f:
        for translation in translations:
            f.write(f'{translation}\n')
            f.write('-'*40 + '\n')
            
    out_name = f'./language_translations/{model_name}_generations/{model_name}_{control_method}_{source_lang}_{dest_lang}_{coef}_translations.txt'
    with open(out_name, 'w') as f:
        for generation in generations:
            f.write(f'{generation}\n')
            f.write('-'*40 + '\n')
            
if __name__ == '__main__':              
    main()