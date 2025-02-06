import torch
from tqdm import tqdm
from datasets import load_dataset

if __name__=='__main__':
    data = load_dataset("wikipedia", "20220301.en")


    isaac_article = data['train']['title'].index('Isaac Newton')
    isaac_text = data['train'][isaac_article]
    text = isaac_text['text']
    lines = text.split('\n')
    sentences = [line.split('. ') for line in lines]
    isaac_sentences = []
    for line in sentences:
        for s in line:
            if len(s.split(' ')) > 4:
                isaac_sentences.append(s)
    print(len(isaac_sentences))
    isaac_sentences[:25]

    cam_article = data['train']['title'].index('Cam Newton')
    cam_text = data['train'][cam_article]
    text = cam_text['text']
    lines = text.split('\n')
    sentences = [line.split('. ') for line in lines]
    cam_sentences = []
    for line in sentences:
        for s in line:
            if len(s.split(' ')) > 4:
                cam_sentences.append(s)
    print(len(all_sentences))
    cam_sentences[:25]