# Debiaswe: try to make word embeddings less sexist

Here we have the code and data for the following paper:
[Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf) by 
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. Proceedings of [NIPS 2016](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings).

**Just looking to download a debiased embedding?**

You can [download](https://drive.google.com/file/d/0B5vZVlu2WoS5ZTBSekpUX0RSNDg/view?usp=sharing) hard debiased version of the Google's Word2Vec embedding trained on Google News (Origin: GoogleNews-vectors-negative300.bin.gz found [here](https://code.google.com/archive/p/word2vec/)).

**Python scripts:**
- **learn_gender_specific.py**: given a word embedding and a seed set of gender-specific words (like <i>king</i>, <i>she</i>, etc.), it learns a much larger list of gender-specific words
- **debias.py**: given a word embedding, sets of gender-pairs, gender-specific words, and pairs to equalize, it outputs a new word embedding. This version basically reads/writes word2vec binary file format.  

```
python learn_gender_specific.py ../embeddings/GoogleNews-vectors-negative300.bin 50000 ../data/gender_specific_seed.json gender_specific_full.json
```

```
python debias.py ../embeddings/GoogleNews-vectors-negative300.bin ../data/definitional_pairs.json ../data/gender_specific_full.json ../data/equalize_pairs.json ../embeddings/GoogleNews-vectors-negative300-hard-debiased.bin
```


We also have seed data used to debias and crowd data used to evaluate the embeddings.

**Data files:**
- **gender_specific_seed.json**: A list of 218 gender-specific words
- **gender_specific_full.json**: A list of 1441 gender-specific words
- **definitional_pairs.json**: The ten pairs of words we use to define the gender direction
- **equalize_pairs.json**: Some crowdsourced F-M pairs of words that represent gender direction

This work only considers M-F gender biases. What about other biases, like racial biases, other genders, ageism, etc.?
