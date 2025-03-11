# How To Guide for GLOSSAM Scripts

This file provides a basic overview of GLOSSAM scripts to enable their use with new corpora. It is expected that users will be familiar with Python, capable of installing necessary dependencies, and already working in a virtual environment (or happy to install the necessary dependencies in their base envionment). It also assumes that users have access to an onboard CUDA enabled GPU or TPU, and have already installed the CUDA toolkit on their hardware.

## Dependencies

The following libraries will need to be installed in the working environment (using pip, conda, etc.).

1. Natural Language Toolkit (NLTK): `pip install nltk`
2. scikit-learn: `pip install scikit-learn`
3. pandas: `pip install pandas`
4. Plotly: `pip install plotly`
5. Pytorch: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (see [https://pytorch.org/](https://pytorch.org/))
6. Sentence Transformers: `pip install sentence-transformers`

## Description of Files

The following files comprise the GLOSSAM scripts. They contain all Python functions written for this project to date:

### Find_lemmas.py

Contains all Python functions necessary to identify and link lemmata from various manuscripts to a single base-text.

### ID_tokens.py

Contains the Python function which tokenises base-text files. The function separates word-tokens, applies `<w>...</w>` tags to words and `<pc>...</pc>` tags to punctuation, then provides each token with a unique identifier based on the original gloss/sentence number. This allows glosses to be linked to lemmata within a given base-text by the functions contained in `Find-lemmas.py`.

This file assumes that base-texts are already annotated to the expected TEI standard.

### MakeGS.py

Contains the Python functions which generate a Gold Standard from Evina Steinová's digital edition of the [_Etymologiae of Isidore of Seville_](https://db.innovatingknowledge.nl/edition/#right-network).

This Gold Standard can be further broken down as necessary. Currently, the Gold Standard is evenly divided into a *Development* set, and a *Test* set. This allows for the development of text-comparison models which do not require training or fine-tuning using roughly half the corpus, while a further half remains untouched for later testing.

For the sake of training/fine-tuning of models in the future, it is intended that a further 20% of the *Test* set (10% of the overall content) will be added to the *Development* set to create a *Training* set. This *Training* set will comprise about 60% of the overall corpus. The remainder of the *Test* set will be split evenly into a *Validation* set and a new *Test* set, each of which will comprise 20% of the overall corpus.

### TextSim.py

Contains all functions necessary to compare gloss similarity by various means. These currently include the following methods:

1. Levenshtein Distance Method
2. Longest Common Substring Method
3. LLM Method

## Description of Functions

The following is a basic walkthrough of the basic functions required to compare gloss similarity using various methods.

### Generate or Load the Gold Standard

First it is necessary to generate the Gold Standard from Steinová's edition:

Run `gen_gs()`

This will create pickle files for both the development set and the test set.

Once the gold standard has been initially generated, the development set can be loaded:

Run `dev_set = load_gs("Gold Standard Dev.pkl")`

### Comparing glosses

All glosses are compared using the `compare_glosses()` function, however, the arguments which must be passed to the function vary depending on the comparison method being employed. Using the large-language-model method also requires that extra steps be taken before the `compare_glosses()` function can be run.

The following are the arguments and possible values which can be passed to the `compare_glosses()` function:

1. `glosses` (required)
   1. `dev_set`
2. `method` (required)
   1. `"ED"` - Edit Distance
   2. `"LCS"` - Lowest Common Substring
   3. `"LLM"` - Large Language Model
3. `gloss_vec_mapping`
   1. `None` (default)
   2. `gloss_dict` (a dictionary mapping the text of glosses to their embeddings)
4. `model`
   1. `None` (default)
   2. `llm` (the large language model from Hugging Face used to create the embeddings)
5. `cutoff_percent`
   1. `50` (default)
   2. `int` (any integer value between 0 and 100)

To use the Levenshtein Distance (edit distance) comparison method, run:

`print(compare_glosses(dev_set, "ED", cutoff_percent=100))`

To use the Longest Common Substring comparison method, run:

`print(compare_glosses(dev_set, "LCS", cutoff_percent=82))`

To use the Large-Language-Model comparison method, there are a few precursors to running the comparison function

* First create a set of all unique glosses to be embedded:

    Run `glosses_to_embed = sorted(list(set([g[0] for g in dev_set] + [g[1] for g in dev_set])))`

* Next identify potential models to use to create embeddings:

    Run `m1 = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin"`

    Run `m2 = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"`

* Next select a specific model to use:

    Run `llm = SentenceTransformer(m1)`

* Next create embeddings for glosses:

    Run `embedded_sentences = llm.encode(glosses_to_embed)`

* Next generate a dictionary mapping glosses to their embeddings:

    Run

      for gloss_index, gloss in enumerate(glosses_to_embed):
          gloss_dict[gloss] = embedded_sentences[gloss_index]

* Finally, use the LLM to compare glosses

    Run `print(compare_glosses(dev_set, "LLM", gloss_dict, llm, 55))`

Regardless of the gloss comparison method employed, the results of the `compare_glosses()` function include:

1. Total number of True Positives achieved (TP)
2. Total number of False Positives achieved (FP)
3. Total number of True Negatives achieved (TN)
4. Total number of False Negatives achieved (FN)
5. Accuracy score
6. Precision score
7. Recall score
8. f-measure
