import concurrent.futures
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, BatchEncoding
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import gc



# Load to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Define set of context_lengths
NUM_TOKENS = 64
set_context_lengths = np.unique(np.logspace(0, np.log10(NUM_TOKENS - 2), num=6, dtype=int))


def get_next_word_predictability(model, encoded_input, next_word):
    # next_word is a GPT-2 token, i.e. the id of the token within the tokenizer's vocabulary
    with torch.no_grad():
        output = model(encoded_input)
        preds = F.softmax(output.logits, dim=-1)
    pred_word = preds[:, -1, next_word].item()  # Get prediction of last token for the next one
    return pred_word


def crop_context(input, word_id, context_length):
    """
    Function that crops a text sequence given as input according to a finish index (word_id), and the length of tokens
     to include before that index (context_length)
    Returns a version of the input cropped to a length of context_length before word_id
     """
    if isinstance(input, dict):
        # In case input is an Encoding instance
        cropped_input = {key: input[key][:, word_id - context_length:word_id] for key in input.keys()}
        return BatchEncoding(cropped_input)
    elif isinstance(input, torch.Tensor):
        # In case input is only the encoded tokens tensor
        cropped_input = input[:, word_id - context_length:word_id]
        return cropped_input
    elif isinstance(input, list):
        # In case input is a list of tokens
        cropped_input = input[word_id - context_length:word_id]
        return cropped_input


def word_by_word_predictability(model, tokenizer, text_sample, sample_id, level, num_tokens=NUM_TOKENS):
    # Encode input with GPT2 tokenizer -- it might give more than 64 tokens
    encoded_input = tokenizer.encode(text_sample, return_tensors='pt').to(model.device)

    # Truncate tokens above num_tokens for equivalent number of tokens across samples
    encoded_input = encoded_input[:, :num_tokens]

    # Get tokens' ids list
    encoded_input_ids = encoded_input.squeeze().tolist()

    # Create list to store prediction scores
    preds = []

    # Wrapping function for multithreading
    def process_word(word_id, word):
        local_preds = []

        # context_lengths_word = np.append(context_lengths[context_lengths<word_id], word_id) # Good idea but adds sparsity in the end, keep for later maybe
        # Only take context length values inferior to position id at stake
        context_lengths_word = set_context_lengths[set_context_lengths < word_id]

        # Test possible context lengths
        for context_length in context_lengths_word:  # for context_length in context_lengths_word
            encoded_input_cropped_context = crop_context(encoded_input, word_id, context_length)
            pred = get_next_word_predictability(model, encoded_input_cropped_context, word)
            local_preds.append({
                "disorder_level": level,
                "sample_id": sample_id,
                "word_pos": word_id,
                "context_length": context_length,
                "predictability": pred,
                "word_token": word,
                "word": tokenizer.decode([word])
            })
        return local_preds

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_word, word_id, word)
            for word_id, word in enumerate(encoded_input_ids)
            if word_id != 0  # Skip first word
        ]
        for future in concurrent.futures.as_completed(futures):
            preds.extend(future.result())

    # Clear memory
    del encoded_input
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return preds


def lexical_predictability_analysis(data_path, results_path, compare_original=False):
    # Read text samples
    data = pd.read_csv(os.path.join(data_path, "text_samples.csv"))

    # Define and load model
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()


    # Iterate over disorder level, samples and individual words
    for level in data.disorder_level.unique():
        if level == 0.1: continue # Skip first level already downloaded
        # Create list to store predictability scores
        preds = []

        print(f"Disorder level: {level * 100:.0f}%")
        data_level = data[data['disorder_level'] == level]

        for sample_id, sample in tqdm(data_level.iterrows(), total=len(data_level), leave=False, position=0,
                                      desc="Samples"):
            # Get word-by-word predictability scores, varying context length for each word, for given sample
            preds_sample = word_by_word_predictability(model, tokenizer, sample['text_shuffled'], sample_id, level)
            preds.extend(preds_sample)

            # Clear memory before next loop
            del preds_sample

        # Convert data to DataFrame
        preds = pd.DataFrame(preds)

        # Save preds
        preds_file = f"{results_path}_level{level*100:.0f}.csv"
        preds.to_csv(preds_file, index=False)


    return preds



if __name__ == '__main__':

    # Compute the word-by-word predictability analysis for all text samples, disorder levels and possible context length
    lexical_predictability_analysis(data_path="/content/drive/MyDrive/lex-pred/text_samples_trunc64", results_path="/content/drive/MyDrive/lex-pred/pred_scores_trunc64_100samples")

