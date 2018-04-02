# Debiaswe: try to make word embeddings less sexist

&#x1F534;[FAT* 2018 tutorial slides](https://drive.google.com/file/d/1IxIdmreH4qVYnx68QVkqCC9-_yyksoxR/view?usp=sharing)


Here we have the code and data for the following paper:
[Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf) by 
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. Proceedings of [NIPS 2016](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings).

**Just looking to download a debiased embedding?**

You can download [binary](https://drive.google.com/file/d/0B5vZVlu2WoS5ZTBSekpUX0RSNDg)/[txt](https://drive.google.com/open?id=1_PvT4ZvtZjhq4HPywA8-u06epht9ccOw) hard debiased version of the Google's Word2Vec embedding trained on Google News (Origin: GoogleNews-vectors-negative300.bin.gz found [here](https://code.google.com/archive/p/word2vec/)).

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


&#x1F535; This is only a partial repo at the moment. I will add more features as I get time.


(All external files that I refer within this repo can be found in [this folder](https://drive.google.com/drive/folders/0B5vZVlu2WoS5dkRFY19YUXVIU2M?usp=sharing).)
