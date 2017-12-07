import numpy as np

def ndcg(documents, rels, cutoff):
    '''
    documents: list of document IDs
    rels:      dict indicating the relevance grade of a document ID
    cutoff:    cutoff rank
    '''
    ideal = [r[0] for r in sorted(rels.items(),
        key=lambda x: x[1], reverse=True)]
    real_dcg = dcg(documents, rels, cutoff)
    ideal_dcg = dcg(ideal, rels, cutoff)
    if ideal_dcg > 0:
        return real_dcg / ideal_dcg
    else:
        return 0.0

def dcg(documents, rels, cutoff):
    result = 0.0
    for idx, d in enumerate(documents[:cutoff]):
        rank = idx + 1
        gain = rels[d]
        decay = 1.0 / np.log2(rank + 1)
        result += gain * decay
    return result
