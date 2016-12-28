from __future__ import print_function, division
import sys
import argparse
from we import *
from sklearn.svm import LinearSVC
import json
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Learn gender specific words

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

parser = argparse.ArgumentParser()
parser.add_argument("embedding_filename", help="The name of the embedding")
parser.add_argument("NUM_TRAINING", type=int)
parser.add_argument("GENDER_SPECIFIC_SEED_WORDS")
parser.add_argument("outfile")

args = parser.parse_args()

embedding_filename = args.embedding_filename
NUM_TRAINING = args.NUM_TRAINING
GENDER_SPECIFIC_SEED_WORDS = args.GENDER_SPECIFIC_SEED_WORDS
OUTFILE = args.outfile

with open(GENDER_SPECIFIC_SEED_WORDS, "r") as f:
    gender_seed = json.load(f)

print("Loading embedding...")
E = WordEmbedding(embedding_filename)

print("Embedding has {} words.".format(len(E.words)))
print("{} seed words from '{}' out of which {} are in the embedding.".format(
    len(gender_seed),
    GENDER_SPECIFIC_SEED_WORDS,
    len([w for w in gender_seed if w in E.words]))
)

gender_seed = set(w for i, w in enumerate(E.words) if w in gender_seed or (w.lower() in gender_seed and i<NUM_TRAINING))
labeled_train = [(i, 1 if w in gender_seed else 0) for i, w in enumerate(E.words) if (i<NUM_TRAINING or w in gender_seed)]
train_indices, train_labels = zip(*labeled_train)
y = np.array(train_labels)
X = np.array([E.vecs[i] for i in train_indices])
C = 1.0
clf = LinearSVC(C=C, tol=0.0001)
clf.fit(X, y)
weights = (0.5 / (sum(y)) * y + 0.5 / (sum(1 - y)) * (1 - y))
weights = 1.0 / len(y)
score = sum((clf.predict(X) == y) * weights)
print(1 - score, sum(y) * 1.0 / len(y))

pred = clf.coef_[0].dot(X.T)
direction = clf.coef_[0]
intercept = clf.intercept_

is_gender_specific = (E.vecs.dot(clf.coef_.T) > -clf.intercept_)

full_gender_specific = list(set([w for label, w in zip(is_gender_specific, E.words)
                            if label]).union(gender_seed))
full_gender_specific.sort(key=lambda w: E.index[w])

with open(OUTFILE, "w") as f:
    json.dump(full_gender_specific, f)
