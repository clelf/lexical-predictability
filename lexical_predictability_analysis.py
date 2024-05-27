import sys

from transformers import GPT2TokenizerFast, GPT2LMHeadModel, BatchEncoding
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np


# Load to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


def get_next_word_predictability(model, encoded_input, next_word):
    # next_word is a token_id, i.e. the id of the token within the tokenizer's vocabulary
    with torch.no_grad():  # useful line?
        # do_sample? temperature?
    #     output = model.generate(**encoded_input, max_new_tokens=1, output_scores=True, return_dict_in_generate=True,
    #                             pad_token_id=tokenizer.eos_token_id)
    # preds = F.softmax(output.scores[0], dim=-1)
    #     output = model(**encoded_input)
        output = model(encoded_input)
        preds = F.softmax(output.logits, dim=-1)
    pred_word = preds[:, -1, next_word].item() # Get prediction predictions of last token for the next one
    return pred_word


def crop_context(input, word_id, context_length):
    # input is an Encoding instance containing items like tensor_ids and attention_mask
    # returns an Encoding instance with tensors cropped to a length of context_length before word_id
    if isinstance(input, dict):
        cropped_input = {key: input[key][:, word_id - context_length:word_id] for key in input.keys()}
        return BatchEncoding(cropped_input)
    elif isinstance(input, torch.Tensor):
        cropped_input = input[:, word_id - context_length:word_id]
        return cropped_input


def word_by_word_predictability(model, tokenizer, text_sample, sample_id, level):
    """
    Quantify predictability word by word, varying context length for each word
    :param text_sample:
    :return: DataFrame storing information about sample at stake, and predictability for every word in sample, for each
    varying context length value
    """

    # Tokenize sample
    # encoded_input = tokenizer(text_sample, return_tensors='pt').to(device)
    # encoded_input_ids = encoded_input.input_ids.squeeze()
    encoded_input = tokenizer.encode(text_sample, return_tensors='pt').to(device)
    encoded_input_ids = encoded_input.squeeze().tolist()

    # Store all encoded input later cropped by context
    # inputs = []
    # words_test = []

    # Create list to store predictability scores
    preds = []

    # Create context length scale: logarithmic to optimize evaluation
    context_lengths = np.unique(np.logspace(0, np.log10(tokenizer.model_max_length- 2), num=22, dtype=int))

    # Test word predictability for every word in sample one by one
    for word_id, word in tqdm(enumerate(encoded_input_ids[::10]), total=len(encoded_input_ids), position=0, leave=False, desc="Single word"):
        # Start at second word, to have at least 1 previous word of context
        if word_id == 0: continue

        # For every word tested, vary context length from very local (previous word) to very global (all available previous words)
        context_lengths_word = np.append(context_lengths[context_lengths<word_id], word_id)

        for context_length in context_lengths_word:
            encoded_input_cropped_context = crop_context(encoded_input, word_id, context_length)

            # inputs.append(encoded_input_cropped_context) # modif
            # words_test.append(word) # modif

            pred = get_next_word_predictability(model, encoded_input_cropped_context, word)

            preds.append({
                "disorder_level": level,
                "sample_id": sample_id,
                "word_pos": word_id,
                "context_length": context_length,
                "predictability": pred, # modif
                # "inputs_id": len(inputs), # modif
                "word_token": word,
                "word": tokenizer.decode(word) # [word] ?
            })

    # preds = get_next_word_predictability(model, inputs, words_test)

    return preds


def lexical_predictability_analysis(data_path, compare_original=False):
    # Read text samples
    data = pd.read_csv(os.path.join(data_path, "text_samples.csv"))

    # Define and load model
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()

    # Create list to store predictability scores
    preds = []
    if compare_original: preds_o = []

    # Iterate over disorder level, samples and individual words
    for level in data.disorder_level.unique():
        print(f"Disorder level: {level*100:.0f}%")
        data_level = data[data['disorder_level'] == level]

        for sample_id, sample in tqdm(data_level.iterrows(), total=len(data_level), leave=True, position=1, desc="Samples"):
            # Get word-by-word predictability scores, varying context length for each word, for given sample
            preds_sample = word_by_word_predictability(model, tokenizer, sample['text_shuffled'], sample_id, level)
            preds.extend(preds_sample)

            if compare_original:
                # Also get predictability scores for original sample
                preds_o.extend(word_by_word_predictability(model, tokenizer, sample['text_original'], sample_id, level))

            if sample_id == 1: break # TODO: delete line

        break # TODO: delete too

    # Convert data to DataFrame
    preds = pd.DataFrame(preds)

    if compare_original:
        preds_o = pd.DataFrame(preds_o)
        preds = pd.merge(preds, preds_o, on=['disorder_level', 'sample_id', 'context_length', 'word_pos'],
                     suffixes=('_shuf', '_o'))

    return preds


def visualize_predictability(df_preds):
    # for level in disorder_levels:
    #     plot(preds[level].context_length, preds[level].pred)

    pass


if __name__ == '__main__':
    df_preds = lexical_predictability_analysis(data_path="text_samples_trunc_gpt2tokenfast")
    df_preds.to_csv("pred_scores_1samples.csv")
    visualize_predictability(visualize_predictability(df_preds))
