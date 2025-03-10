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
