import we
import json
import numpy as np
import argparse
import sys
if sys.version_info[0] < 3:
    import io
    open = io.open
    from __future__ import print_function, division
"""
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
                E.vecs[E.index[a]] = z * gender_direction + y
                E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("definitional_filename", help="JSON of definitional pairs")
    parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (one per line)")
    parser.add_argument("equalize_filename", help="???.bin")
    parser.add_argument("debiased_filename", help="???.bin")

    args = parser.parse_args()
    print(args)

    with open(args.definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(args.equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(args.gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(args.embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if args.embedding_filename[-4:] == args.debiased_filename[-4:] == ".bin":
        from gensim.models import word2vec
        sys.stdout.write("Saving in word2vec format as "+args.debiased_filename)
        sys.stdout.write(".\n    Step 1. Loading old embedding.")
        sys.stdout.flush()
        model = word2vec.Word2Vec.load_word2vec_format(args.embedding_filename, binary=True)
        d = E.vecs[0].shape[0]
        sys.stdout.write(".\n    Step 2. Copying data [0.0%]")
        sys.stdout.flush()
        for counter, w in enumerate(E.words):
            if counter % 1000 == 0:
                sys.stdout.write("\r    Step 2. Copying data [" + str(int(counter * 1000 / len(E.words)) / 10.0) + "%]")
                sys.stdout.flush()
            v = E.v(w)
            for i in range(d):
                model[w][i] = v[i]
        sys.stdout.write("\n     Step 3. Writing file")
        sys.stdout.flush()
        model.save_word2vec_format(args.debiased_filename, binary=True)
    else:
        E.save(args.debiased_filename)

    print("\n\nDone!\n")
