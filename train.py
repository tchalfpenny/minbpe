import os
import time

text = open("taylorswift.txt", "r").read()
tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience

def get_stats(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]): # Pythonic way to iterate consecutive pairs
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
# print(sorted(((v,k) for k,v in stats.items()), reverse=True))

top_pair = max(stats, key=stats.get)
print(f"Top tokens: {top_pair}, chr pair: '{chr(top_pair[0])}{chr(top_pair[1])}'") 

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train(token_dataset, vocab_size):
    num_merges = vocab_size - 256
    ids = list(token_dataset) # copy so we don't destroy the original list
    merges = {} # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        # print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return ids, merges

t0 = time.time()
bpe_tokens, bpe_merges = train(tokens, 400)
# print(bpe_merges)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds.")
print("tokens length:", len(tokens))
print("bpe_tokens length:", len(bpe_tokens))
print(f"compression ratio: {len(tokens) / len(bpe_tokens):.2f}X")