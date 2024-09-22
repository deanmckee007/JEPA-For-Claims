# data/vocab.py
from collections import Counter

def create_vocab(df, min_freq=5):
    cpt_counter = Counter()
    icd_counter = Counter()
    ttnc_counter = Counter()

    for sequence in df['input']:
        for token in sequence:
            if token.startswith('cpt_'):
                cpt_counter[token] += 1
            elif token.startswith('icd_'):
                icd_counter[token] += 1
            elif token.startswith('ttnc_'):
                ttnc_counter[token] += 1

    # Create vocabularies with a minimum frequency threshold
    def build_vocab(counter, reserved_tokens=['<PAD>', '<UNK>'], min_freq=1):
        vocab = {}
        index = 0
        for token in reserved_tokens:
            vocab[token] = index
            index += 1
        for token, freq in counter.items():
            if freq >= min_freq and token not in vocab:
                vocab[token] = index
                index += 1
        return vocab

    cpt_vocab = build_vocab(cpt_counter, reserved_tokens=['<PAD>', '<UNK>'], min_freq=min_freq)
    icd_vocab = build_vocab(icd_counter, reserved_tokens=['<PAD>', '<UNK>'], min_freq=min_freq)
    ttnc_vocab = build_vocab(ttnc_counter, reserved_tokens=['<PAD>', '<UNK>'], min_freq=min_freq)

    # Reverse mapping for unified vocabulary (if needed)
    unified_vocab_reverse = {idx: token for token, idx in {**cpt_vocab, **icd_vocab, **ttnc_vocab}.items()}

    return cpt_vocab, icd_vocab, ttnc_vocab, unified_vocab_reverse, cpt_counter, icd_counter, ttnc_counter

def calculate_rarity(cpt_vocab, icd_vocab, ttnc_vocab, cpt_counter, icd_counter, ttnc_counter):
    def calculate_rarity_score(counter, vocab):
        total_count = sum(counter.values())
        token_rarity = {}
        for token_str, count in counter.items():
            token_id = vocab.get(token_str)
            if token_id is not None:
                rarity = 1 / (count / total_count)
                token_rarity[token_id] = rarity
        return token_rarity

    cpt_rarity = calculate_rarity_score(cpt_counter, cpt_vocab)
    icd_rarity = calculate_rarity_score(icd_counter, icd_vocab)
    ttnc_rarity = calculate_rarity_score(ttnc_counter, ttnc_vocab)

    return cpt_rarity, icd_rarity, ttnc_rarity


