import pandas as pd
import os
import math
import numpy as np

cnetnb = None


def cosine_similarity(vector1, vector2):

    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0

    return dot_product/magnitude

# 'data/numberbatch-en.txt'
def load_cnet_numbatch(filepath):
	cnetnb = pd.read_csv(filepath, delim_whitespace=True, skiprows=1, header=None, index_col=0)
	print("Concept-Net numberbatch-en loaded ", cnetnb.shape)
	return cnetnb , cnetnb.as_matrix()

def test_cnet_numbatch(cnetnb, word1, word2):
	v1 = cnetnb.loc[word1].as_matrix()
	v2 = cnetnb.loc[word2].as_matrix()
	print('Cosine Similarity between ', word1, 'and ', word2, ' is : ')
	print(cosine_similarity(v1,v2))

if __name__ == '__main__':
	cnetnb , cnetnb_mat  = load_cnet_numbatch('conceptnet_embeddings/numberbatch-en.txt')
	print(cnetnb_mat.shape)
	test_cnet_numbatch(cnetnb, 'cumberbatch', 'otter')
	test_cnet_numbatch(cnetnb, 'cumberbatch', 'actor')
	test_cnet_numbatch(cnetnb, 'cumberbatch', 'cumberbatch')

			