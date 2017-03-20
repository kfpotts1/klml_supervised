import numpy as np


def gen_hash_features(name, B, FIX, method='basic'):
    features = np.zeros(B)
    if method == 'basic':
        for i in range(FIX):
            feature_str = "pre" + name[:i]
            features[hash(feature_str) % B] = 1
            feature_str = "suf" + name[-1 * i:]
            features[hash(feature_str) % B] = 1
    else:
        for i in range(1, FIX):
            if i == 1:
                feature_str = "pre" + name[:i]
                features[hash(feature_str) % B] += 2
            feature_str = "suf" + name[-1 * i:]
            features[hash(feature_str) % B] += 1
        for i in range(len(name) - 1):
            features[np.abs(ord(name[i]) - ord(name[i + 1])) % B] = 2
            
        if name[-1] in 'aeiouyAEIOUY':
            features[hash('last_is_vowel') % B] += 2.2
            
        if name[0] not in 'aeiouyAEIOUY':
            features[hash('first_is_vowel') % B] += 2
    return features


def names_to_features(names_list, B=256, FIX=4):
    n = len(names_list)
    x = np.zeros((n, B))
    for i in range(n):
        x[i, :] = gen_hash_features(names_list[i], B, FIX)
    return x

