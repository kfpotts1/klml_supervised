import numpy as np


def gen_hash_features(name, B=256, FIX=4, method='basic'):
    """
    Generates a vector of features given a person's name.

    Extracts and hashes features for a given name.
    Focuses on the beginning and endings of the words (prefix and suffix).

    Syntax:
        features[i] = gen_hash_features(names[i])
        features = gen_hash_features('Muhammad')

    Args:
        name: (string)
            intended to be a person's name or similar style word
        B: (int)
            number of unique features to be used - hyperparameter
        FIX: (int)
            length of prefixes and suffixes to be used - hyperparameter
        method: (string)
            'basic' for a simple prefix/suffix hash
            'other' for added features: vowel locations, letter distances, modified weights

    Returns:
        features: (array_like)
            a 1xB array of features for the given name
    """

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
    """
        Generates a nxB matrix of features given a list of n names

        Extracts and hashes features for each given name.

        Syntax:
            features = gen_hash_features(names)

        Args:
            names_list: (array_like)
                intended to be a list of people's names or similar style words
            B: (int)
                number of unique features to be used - hyperparameter
            FIX: (int)
                length of prefixes and suffixes to be used - hyperparameter

        Returns:
            features: (array_like)
                a nxB matrix of features for the n given names
                where the ith row is the feature vector for the ith name
        """
    n = len(names_list)
    x = np.zeros((n, B))
    for i in range(n):
        x[i, :] = gen_hash_features(names_list[i], B, FIX)
    return x

