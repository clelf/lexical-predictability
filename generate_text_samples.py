"""
Task 1: We are going to introduce ‘disorder’ artificially by randomly shuffling around the order of words.
For each of 10 disorder levels (10–100%), please create 100 samples that randomly shuffle the order of the
appropriate proportion of words.
For the text sample, please choose a random chapter from a classic book (On the Origin of Species by Charles Darwin,
available here: https://www.gutenberg.org/files/1228/1228-h/1228-h.htm).

"""
import os, shutil, codecs, string, random, re
import json, csv
import numpy as np
import spacy
from tqdm import tqdm
from string import punctuation

EXCLUDE_CHARS = set(punctuation).union(set('’'))

# Initialize spacy analyzer in English
nlp = spacy.load('en_core_web_sm')


def simple_tokeniser(text):
    # Simply splits sentences into single words
    return text.split()


def clean_text(text):
    # Exclude punctuation characters and lowercase all the given text sample.
    for char in EXCLUDE_CHARS:
        text = text.replace(char, ' ')
    return text.lower()


def choose_random_chapter(chapters):
    # Return random choice in range 0 - #chapters (introduction counts as 0)
    chap_id = random.randint(0, len(chapters)-1)
    return chap_id, chapters[chap_id]


def parse_book(bookFile="OntheOriginofSpecies_CharlesDarwin_text.txt"):
    # Load book
    with codecs.open(bookFile, encoding="utf8") as f:
        book = f.read()

    # Crop until beginning of book and remove all '\r'
    book = book[10715:].replace("\r", "")

    # Crop after end of book
    ending_words = ", evolved."
    end_index = book.find(ending_words) + len(ending_words)
    book = book[:end_index]

    # Remove first all single occurrences of '\n', and then reduce multiple successive occurrences into single ones
    book = re.sub(r'(?<!\n)\n(?!\n)', ' ', book)
    book = re.sub(r'\n+', '\n', book)

    # Define identifier of chapters' start: new line starting with 'INTRODUCTION' or 'CHAPTER' until end of the line
    chapter_start = re.compile(r'^(INTRODUCTION|CHAPTER)', re.MULTILINE)

    # Split book into chapters: each cell is a chapter
    # Split results a list as follows: 'Intro', 'text intro', 'Chapter', '1. text...', 'Chapter', '2. text'...
    # So keep only every second text cells
    chapters = chapter_start.split(book)[2::2]

    # Remove remaining '\n' in the introduction chapter, and for the others exclude chapters' description,
    # which is all the text comprised up until the second '\n':
    chapters = [chapters[0].replace('\n', ' ')] + [' '.join(c.split('\n')[2:]) for c in chapters[1:]]

    # Return list of chapters
    return chapters


def shuffle_sample(text_sample, disorder_level, max_tokens=-1):

    # Tokenize words sequence into list of single words
    text_obj = nlp(text_sample) # nlp(clean_text(text_sample)) to clean from punctuation and lowercase

    # Get list of word tokens, up until optional max_tokens limit
    words_all = [token.text for token in text_obj]
    # words = simple_tokeniser(text_sample_cleaned)

    # Choose random starting index to truncate from, that should be the beginning of a sentence
    trunc_start = random.choice(
        [i for i, token in enumerate(words_all)
         if i < len(words_all)-max_tokens and (i == 0 or words_all[i - 1] == ".") and token != "."])

    # Truncate from this start index until this start index + max_tokens
    words = words_all[trunc_start:trunc_start+max_tokens]

    # Create list of words' indices
    words_ids = [i for i in range(len(words))]

    # Define number of words to shuffle according to disorder_level proportion
    k_2shuffle = int(disorder_level * len(words))

    # Sample disorder_level proportion of indices to shuffle within total indices list, and shuffle them
    words_ids_sampled = random.sample(population=words_ids, k=k_2shuffle)
    random.shuffle(words_ids_sampled)

    # Get sampled indices in order, to fill positions one by one
    words_ids_sampled_sorted = sorted(words_ids_sampled)

    # Create copy of initial words to be modified
    words_sample_shuffled = words[:]

    # Insert shuffled words at positions of unshuffled sampled indices
    for w_sampled_shuffled, w_sampled_sorted in zip (words_ids_sampled, words_ids_sampled_sorted):
        words_sample_shuffled[w_sampled_sorted] = words[w_sampled_shuffled]

    # Checks.
    set_ori = set(words)
    set_fin = set(words_sample_shuffled)
    if set_ori != set_fin:
        print("Diff vocab")
        print("Len words, words shuffled: ", len(set_ori), len(set_fin))
        print([(word_ori, word_fin) for word_ori, word_fin in zip(set_ori, set_fin) if word_ori != word_fin])
        print("K_2s: ", k_2shuffle, " / Eff k: ", len(words_ids_sampled))
        print("Ori set: ", list(set_ori)[:10], list(set_ori)[-10:])
        print("Fin set: ", list(set_fin)[:10], list(set_fin)[-10:])
        print("Missing: ", sorted(set_ori - set_fin))
        print("Added :", sorted(set_fin - set_ori))

    # Join words with new order into a sentence again
    sample_shuffled = ' '.join(words_sample_shuffled)

    # Also reconstruct tokenized and truncated original text sample to match tokenization effect
    text_original = ' '.join(words)

    # Return shuffled sample as one string, and as a list of words
    return sample_shuffled, words_sample_shuffled, text_original, words

def compute_shuffling_efficacy(words_original, words_shuffled):
    """
    Computes effective proportion of words shuffled from original order
    :param words_original: list of words (tokens) in original order
    :param words_shuffled: list of words (tokens) after random shuffling of a certain proportion of words
    :return: effective proportion of words that were shuffled
    """
    if len(set(words_original))!=len(set(words_shuffled)):
        print(len(set(words_original)), len(set(words_shuffled)))
        raise ValueError("Number of words between original and shuffled sample not matching")

    shuffle_prop_eff = sum([words_original[i] != words_shuffled[i] for i in range(len(words_original))]) / len(
        words_original)

    return shuffle_prop_eff

def test_shuffling_efficacy(disorder_level, n=50):
    # Parse book
    chapters = parse_book()

    avg_eff_shuff_prop = []

    for i in tqdm(range(n)):
        # Select one chapter randomly
        chap_id, text_sample = choose_random_chapter(chapters)

        # Reconstruct original text sample to match tokenization effect on reconstructed shuffled sample
        text_obj = nlp(text_sample)
        words_original = [token.text for token in text_obj]
        text_original = ' '.join(words_original)

        text_sample_shuffled, words_shuffled = shuffle_sample(text_sample, disorder_level)

        if n==1:
            print("Original text, chapter: ", chap_id)
            print(text_sample[:50], " ... ", text_sample[-50:], '\n')

        shuffle_prop_eff = compute_shuffling_efficacy(words_original, words_shuffled)
        print("Shuffling efficacy: ", shuffle_prop_eff)

        # Checks.
        set_ori = set(words_original)
        set_fin = set(words_shuffled)
        if set_ori != set_fin:
            print("Diff vocab")
            print("Len set words, len set words shuffled: ", len(set_ori), len(set_fin))
            print("Ori set: ", list(set_ori)[:10], list(set_ori)[-10:])
            print("Fin set: ", list(set_fin)[:10], list(set_fin)[-10:])
            print("Missing: ", sorted(set_ori - set_fin))
            print("Added :", sorted(set_fin - set_ori))
            print([(word_ori, word_fin) for word_ori, word_fin in zip(set_ori, set_fin) if word_ori != word_fin])
            print("K_2s: ", disorder_level*len(words_original), " / Eff k: ", sum([words_original[i] != words_shuffled[i] for i in range(len(words_original))]))

        avg_eff_shuff_prop.append(shuffle_prop_eff)

        if n==1 or shuffle_prop_eff > 0.5:
            print("Cleaned original text: ")
            print(text_original[:50], " ... ", text_original[-50:], '\n')

            print(f"Disorder level: {disorder_level * 100:.0f}%, shuffled text with"
                  f" {sum([words_original[i] != words_shuffled[i] for i in range(len(words_original))])}/{len(words_original)}"
                  f" (={sum([words_original[i] != words_shuffled[i] for i in range(len(words_original))]) / len(words_original) * 100:.1f}%)"
                  f" words shuffled: ")

            print(text_sample_shuffled[:50], " ... ", text_sample_shuffled[-50:])

    print(f"Average effective shuffled proportion for desired {disorder_level*100:.0f}%: {np.mean(avg_eff_shuff_prop)*100:.1f}")

    return np.mean(avg_eff_shuff_prop), avg_eff_shuff_prop


def generate_samples(n_k=100, data_path=None, max_tokens=-1):
    """
    For each of 10 disorder levels (10–100%), create 100 samples that randomly
    shuffle the order of the appropriate proportion of words.

    :param n_k:
    :return:
    """

    # Parse book to read chapters as separate text samples
    chapters = parse_book()

    # Create the 10 disorder levels (from 10% to 100%, i.e. 0.1 to 1)
    disorder_levels = [round(i/10, 1) for i in range(1,11)]

    # Define where to store generated text samples (overwrite destination if already existing)
    if data_path is None:
        data_path = "text_samples"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

    dataset = []
    for level in tqdm(disorder_levels):
        for i in tqdm(range(n_k)):
            # Select a random chapter to use as text sample
            chap_id, text_sample = choose_random_chapter(chapters)

            # Shuffle text sample according to disorder level
            text_shuffled, words_shuffled, text_original, words_original = shuffle_sample(text_sample, level, max_tokens=max_tokens)

            dataset.append({
                "disorder_level": level,
                "sample_id": i,
                "shuffled_prop_eff": compute_shuffling_efficacy(words_original, words_shuffled),
                "chap_id": chap_id,
                "text_original": text_original,
                "text_shuffled": text_shuffled
                    })

    # Save original and shuffled sample, as well as chapter indice for retrieval
    # with open(os.path.join(data_path, f"text_samples.json"), 'w', encoding='utf-8') as data_file:
    #     json.dump(dataset, data_file)

    with open(os.path.join(data_path, f"text_samples.csv"), 'w', encoding='utf-8', newline='') as data_file:
        writer = csv.DictWriter(data_file, fieldnames=dataset[0].keys())
        writer.writeheader()
        writer.writerows(dataset)


if __name__ == '__main__':

    # test_shuffling_efficacy(disorder_level=0.5, n=5)
    generate_samples(n_k=100, data_path="text_samples_trunc64", max_tokens=64)
