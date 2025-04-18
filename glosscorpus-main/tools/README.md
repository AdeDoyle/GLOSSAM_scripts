# Guide for GLOSSAM Scripts

This file provides a basic overview of GLOSSAM scripts to enable their use with new corpora.

## Prerequisites

Certain folders and paths are necessary to ensure the correct functioning of the scripts described here. It is also expected that users will be familiar with Python, and capable of installing necessary dependencies. Some models used here also require that users have access to an onboard CUDA enabled GPU or TPU, and have already installed the CUDA toolkit on their hardware.

### Folders

The following folders and their contents are required:

* `basetexts`
  * This folder contains `.xml` files for primary texts (formatted as required for the Gloss Corpus website)
* `collections`
  * This folder contains sub-folders for each `.xml` file in `basetexts`
    * e.g. `aelfric`, `felire`, `isidore`, `josephus`, `priscian`
    * Each of these sub-folders contains `.xml` files for relevant gloss collections (formatted as required for the Gloss Corpus website)
* `gloss_comparisons`
  * This folder can be used to run gloss similarity analysis on specific gloss collections
    * These collections must be formatted as required for the Gloss Corpus website
    * These gloss collections should be related to the same primary text in `basetexts`

Note, these folders must be named exactly as they are here, and their names are case-sensitive.

### Dependencies

The following libraries will also need to be installed in the working environment (using pip, conda, etc.).

1. Natural Language Toolkit (NLTK): `pip install nltk`
2. Beautiful Soup: `pip install beautifulsoup4`
3. scikit-learn: `pip install scikit-learn`
4. pandas: `pip install pandas`
5. OpenPyXL: `pip install openpyxl`
6. Plotly: `pip install plotly`
7. Pytorch: `pip3 install torch torchvision torchaudio` (see [https://pytorch.org/](https://pytorch.org/) for version compatibility with CUDA, and OS-specific compatibility issues)
8. Sentence Transformers: `pip install sentence-transformers`

It will also be necessary to ensure that a version of CUDA toolkit is installed which is compatible with the Pytorch installation.

### Test CUDA and GPU

To ensure that the Pytorch installation is utilising CUDA run:

    import torch
    print(torch.cuda.is_available())

To ensure that Pytorch has access to the GPU to increase the speed of model training/fine-tuning, etc. run:

    print(torch.cuda.get_device_name(0))

# TL;DR - Quick-run Gloss Similarity Model

To run the best performing gloss similarity model on a group of gloss collections, take the following steps:

1. Add the gloss collections to the `gloss_comparisons` folder.
2. Run the following:


    from Apply_models import apply_bestmod
    apply_bestmod()

Note: The gloss collections must be linked to the same primary text file in the `basetexts` folder.

## Quick-run specific models

To run other models with optimised parameters this can be done by specifying the required model. For example, to use the edit distance model, run the following:

    apply_bestmod(model="ED")

Alternatively, to use the longest common substring model, run the following:

    apply_bestmod(model="LCS")

If you would like to run an LLM model other than the one found to be optimised for this task, a particular model from Hugging Face can be specified like so:

    model_link = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin"
    apply_bestmod(model="LLM", llm=model_link)

It is also possible to run all available models at once, with optimised parameters for each model type, like so:

    from Apply_models import apply_allmods
    apply_allmods()

# Detailed Description of Files and Functions

This section is intended to support users who want to adjust hyperparameters of models, or otherwise interact on a deeper level with the GLOSSAM code pipeline.

## Description of Python Files

The following Python files comprise the GLOSSAM scripts. They contain all functions written for this project to date.

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

Contains all functions necessary to compare gloss similarity by various means for the purpose of assessing models. These currently include the following methods:

1. Levenshtein Distance Method
2. Longest Common Substring Method
3. LLM Method

### Apply_models.py

Contains functions to apply gloss similarity models to real-world gloss collections.

## Description of Functions

The following is a basic walkthrough of the basic functions required to compare gloss similarity using various methods.

### Generate and Load the Gold Standard

First it is necessary to generate the Gold Standard from Steinová's edition. This requires that the `.xml` file for Steinová's edition be in the current working directory, and that it be named `Isidore_Gold.xml`.

Run:

    from MakeGS import gen_gs
    gen_gs()

This will create pickle files for both the development set and the test set in the current working directory. These will be named `Gold Standard Dev.pkl` and `Gold Standard Test.pkl` respectively.

Once the gold standard has been initially generated, the development set can be loaded and set to a variable.

Run:

    from MakeGS import load_gs
    dev_set = load_gs("Gold Standard Dev.pkl")

The format of the development set is a list of sub-lists. Each sub-list contains a pair of glosses as the first and second elements, and a label as the third element which indicates whether the pair of glosses are related or unrelated. For example:

    [
        ['id est viventem', 'vivum', 'Unrelated'],
        ['velaminibus', 'opertoriis', 'Unrelated'],
        ['non aperta', 'perturbata', 'Unrelated'],
        ['id est circumlongus', 'circumlongus', 'Related'],
        ['pro navi', 'navis de pine', 'Unrelated'],
        ['circumlongus', 'circumlongus', 'Related'],
        ['dispensationem', 'id est dispensativam', 'Unrelated']
    ]

To use any of the following comparison methods with a new corpus, or to generate sentence embeddings for any new corpus using the following functions, the corpus must first be rendered in the same format as this development set.

### Comparing glosses

All glosses are compared using the `compare_glosses()` function, however, the arguments which must be passed to the function vary depending on the comparison method being employed. The arguments `glosses` and `method` are required to be supplied any time the function is run. Default values are supplied for the arguments `gloss_vec_mapping`, `model`, and `cutoff_percent`, which will be used as necessary by the function, depending which type of comparison method is being used. The default values for these arguments can be replaced by supplying new values when running the function. Using the large-language-model method also requires that extra steps be taken before the `compare_glosses()` function can be run.

The following are the arguments and possible values which can be passed to the `compare_glosses()` function:

1. `glosses` (required)
   1. `dev_set`
2. `method` (required)
   1. `"ED"` - Edit Distance (i.e. Levenshtein Distance)
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

To use this function, first load it in by running:

    from TextSim import compare_glosses

### The Levenshtein Distance Comparsion Method

To use this method, run:

    print(compare_glosses(dev_set, "ED"))

The `cutoff_percent` argument for this method represents the threshold for the percentage of difference which is tolerable between two strings (glosses) while still considering them to be "related". The default value is `100` (i.e. 100%), so only strings which are exact matches will be considered to be related by default. The value for `cutoff_percent` can be altered by supplying it when running the function, like so:

    print(compare_glosses(dev_set, "ED", cutoff_percent=70))

### The Longest Common Substring Comparison Method

To use this method, run:

    print(compare_glosses(dev_set, "LCS"))

The `cutoff_percent` argument for this method represents the percentage of one of the two strings (glosses) which must be comprised of the longest common substring in order for the two glosses to be considered "related". The default value is `82` (i.e. 82%). The value for `cutoff_percent` can be altered by supplying it when running the function, like so:

    print(compare_glosses(dev_set, "LCS", cutoff_percent=70))

### The Large-Language-Model (LLM) Comparison Method

If using the LLM comparison method, there are a few precursors to running the comparison function.

First import Sentence Transformers library:

    from sentence_transformers import SentenceTransformer

Next create a set of all unique glosses to be embedded:

    glosses_to_embed = sorted(list(set(
        [g[0] for g in dev_set] + [g[1] for g in dev_set]
    )))

Next identify potential models to use to create embeddings:

    m1 = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin"
    m2 = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"

Next select a specific LLM model to use:

    llm = SentenceTransformer(m1)

Note: The first time this function is run, it will download the sentence transformers library. This may take several minutes. It may also require a token to be set up in your Python interpreter from a Hugging Face account for certain models depending on licencing.

Next create embeddings for glosses:

    embedded_sentences = llm.encode(glosses_to_embed)

Next generate a dictionary mapping glosses to their embeddings:

    gloss_dict = dict()
    for gloss_index, gloss in enumerate(glosses_to_embed):
        gloss_dict[gloss] = embedded_sentences[gloss_index]

Finally, use the LLM to compare glosses:

    print(compare_glosses(dev_set, "LLM", gloss_dict, llm))

It is necessary to supply non-null values for the arguments `gloss_vec_mapping` (a mapping of the text of glosses to their embeddings) and `model` (the LLM being employed) when comparing glosses using the LLM comparison method.

The `cutoff_percent` argument for this method represents the minimum percentage of semantic similarity required between two gloss embeddings in order to consider the two glosses to be "related". The default value is `50` (i.e. 50% semantic similarity). The value for `cutoff_percent` can be altered by supplying it when running the function, like so:

    print(compare_glosses(dev_set, "LLM", gloss_dict, llm, cutoff_percent=70))

### Results

Regardless of the gloss comparison method employed, the results of the `compare_glosses()` function take the form of a list. This list includes, in order:

1. Total number of True Positives achieved (TP)
2. Total number of False Positives achieved (FP)
3. Total number of True Negatives achieved (TN)
4. Total number of False Negatives achieved (FN)
5. Accuracy score
6. Precision score
7. Recall score
8. f-measure

Changing the value of `cutoff_percent` will alter the quality of results for each comparison method. Default values for `cutoff_percent` are currently set to be roughly optimised for each comparison method, however, other parameters may also affect the output (for example, using m2 instead of m1, or using a different LLM entirely).

The `organise_output()` function can be used in tandem with the `compare_glosses()` function to make this output more readable. For example:

    from TextSim import organise_output
    print(organise_output(compare_glosses(dev_set, "ED")))
    print(organise_output(compare_glosses(dev_set, "LCS")))
    print(organise_output(compare_glosses(dev_set, "LLM", gloss_dict, llm)))

### Saving Results for All Models

The `save_all_outputs()` function can be used to save the output of all models and variables to spreadsheets without running the individual models separately. It uses all possible values for `cutoff_percent` between 0 and 100 for each model. It can be called without passing any arguments, like so:

    from TextSim import save_all_outputs
    save_all_outputs()

### Visualising LLM Embeddings

The multidimensional sentence embeddings produced by the LLM comparison method can be visualised by applying a clustering algorithm, then projecting the results into 2D.

First, load the necessary functions:

    from TextSim import apply_clustering, plot_clusters

Next, apply the clustering algorithm to the embeddings:

    clusters = apply_clustering(embedded_sentences)

By default, the `apply_clustering` function utilises k-means clustering, which requires the number of clusters to be manually specified. The number of specified clusters is set to 2 by default, however, different numbers of clusters can be selected like so:

    clusters = apply_clustering(embedded_sentences, num_clusters=6)

It is also possible to apply DBSCAN (density-based spatial clustering of applications with noise) instead of k-means clustering. DBSCAN does not require the number of clusters to be specified, and can be applied like so:

    clusters = apply_clustering(embedded_sentences, clustering_method="DBSCAN")

Once clustering has been applied to the embeddings, apply dimensionality reduction and generate scatterplot:

    plot = plot_clusters(glosses_to_embed, embedded_sentences, clusters)

The default name for the plot is set to `"Latin Gloss Embeddings"`, however, a different name can be selected like so:

    plot = plot_clusters(glosses_to_embed, embedded_sentences, clusters, plot_name="Greek Gloss Embeddings")

Finally, display the plot:

    plot.show()

### Saving and Loading Embedding Plots

Once a plot has been set to a variable, as above, it can be saved to a pickle file as follows and later reloaded as follows.

First, load the necessary functions:

    from TextSim import save_cluster_plot, load_cluster_plot

To save a plot run:

    save_cluster_plot(plot)

By default, the name of the saved file will be `Scatter Plot.pkl`, however, an alternative name can be selected like so:

    save_cluster_plot(plot, file_name="Plot of Clusters")

To later reload a saved plot, set it to a new variable, then display it, run:

    loaded_plot = load_cluster_plot(file_name="Scatter Plot.pkl")
    loaded_plot.show()
