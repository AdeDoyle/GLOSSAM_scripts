from nltk import edit_distance as ed
from MakeGS import load_gs
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
import torch
from sentence_transformers import SentenceTransformer
import pickle as pkl
import os


def norm_ld(s1, s2):
    """Calculate the percentage difference between two strings"""

    lev_dist = ed(s1, s2)  # Get the edit distance

    l1 = len(s1)
    l2 = len(s2)
    max_dif = max(l1, l2)  # Find the length of the larger of the two strings (this is the max possible edit distance)

    lev_norm = (lev_dist/max_dif)*100  # Normalise the edit distance, then render as a percentage of difference

    return lev_norm


def ed_compare(str1, str2, n=100):
    """Compares two strings, predicts whether they're related based on edit distance"""

    cutoff = n

    lev_norm = norm_ld(str1, str2)
    if lev_norm >= cutoff:
        result = "Related"
    else:
        result = "Unrelated"

    return result


def lcs(s1, s2):
    """finds the longest common substring of two strings"""

    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    max_length = 0
    end_index = 0

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i

    return s1[end_index - max_length:end_index]


def slcs(s1, s2):
    """finds the longest, then the second-longest common substring of two strings"""

    lcsubstr = lcs(s1, s2)

    if lcsubstr:
        s1_minus = "".join(s1.split(lcsubstr))
        s2_minus = "".join(s2.split(lcsubstr))
        slcsubstr = lcs(s1_minus, s2_minus)
    else:
        slcsubstr = ""

    return lcsubstr, slcsubstr


def lcs_compare(s1, s2, n=82):
    """Compares two strings, predicts whether they're related based on longest (and 2nd longest) common substring"""

    lcs1, lcs2 = slcs(s1, s2)
    combo_lcs = len(lcs1) + len(lcs2)

    len_s1, len_s2 = len(s1), len(s2)
    min_len = min(len_s1, len_s2)

    if n == 0:
        cutoff = 0
    else:
        cutoff = min_len*(n/100)

    if combo_lcs >= cutoff:
        result = "Related"
    else:
        result = "Unrelated"

    return result


def apply_clustering(emb_array, clustering_method="KMeans", num_clusters=2):
    """Applies a clustering algorithm to gloss vectors to identify semantically linked glosses"""

    embedded_clusters = None

    if clustering_method == "KMeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        embedded_clusters = kmeans.fit_predict(emb_array)

    elif clustering_method == "DBSCAN":
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        embedded_clusters = dbscan.fit_predict(emb_array)

    return embedded_clusters


def plot_clusters(sentences, embeddings, clustered_embeddings, plot_name="Latin Gloss Embeddings"):
    """Reduces dimensionality of (gloss) sentence embeddings to 2D and creates a scatterplot for them"""

    tsne_embedded = TSNE(n_components=2).fit_transform(embeddings)

    df_embeddings = pd.DataFrame(tsne_embedded)
    df_embeddings = df_embeddings.rename(columns={0: 'x', 1: 'y'})
    df_embeddings = df_embeddings.assign(label=clustered_embeddings)
    df_embeddings = df_embeddings.assign(text=sentences)

    fig = px.scatter(
        df_embeddings, x='x', y='y',
        color='label', labels={'color': 'label'},
        hover_data=['text'], title=plot_name
    )

    return fig


def llm_compare(gloss_1, gloss_2, gloss_vec_mapping, model, n=50):
    """Compares two glosses, predicts whether they're related based on semantic similarity"""

    similarity_score = model.similarity(gloss_vec_mapping.get(gloss_1), gloss_vec_mapping.get(gloss_2))
    similarity_score = similarity_score.item()

    if n == 0:
        cutoff = 0
    else:
        cutoff = n / 100

    if similarity_score >= cutoff:
        result = "Related"
    else:
        result = "Unrelated"

    return result


def save_cluster_plot(fig, file_name="Scatter Plot"):
    """Saves a scatter plot generated using the plot_clusters() function as a pickle file"""
    file_name = file_name + ".pkl"
    with open(file_name, 'wb') as spl:
        pkl.dump(fig, spl)


def load_cluster_plot(file_name="Scatter Plot.pkl"):
    """Loads a scatter plot from a pickle file"""
    with open(file_name, 'rb') as loadfile:
        file_loaded = pkl.load(loadfile)

    return file_loaded


def compare_glosses(glosses, method, gloss_vec_mapping=None, model=None, cutoff_percent=None):
    """
    Compares two sets of glosses, predicts whether each gloss is related or unrelated
    Scores predictions by comparison to known correct labels to calculate Accuracy, precision, recall and f-measure
    """

    labels = [g[2] for g in glosses]

    # If a cutoff is supplied
    if isinstance(cutoff_percent, int):
        # If using the edit distance method
        if method == "ED":
            results = [ed_compare(g[0], g[1], cutoff_percent) for g in glosses]
        # If using the longest common substring method
        elif method == "LCS":
            results = [lcs_compare(g[0], g[1], cutoff_percent) for g in glosses]
        # If using a large language model method
        elif method == "LLM":
            results = [llm_compare(g[0], g[1], gloss_vec_mapping, model, cutoff_percent) for g in glosses]
        # If using any other method
        else:
            results = labels
    else:
        # If using the edit distance method
        if method == "ED":
            results = [ed_compare(g[0], g[1]) for g in glosses]
        # If using the longest common substring method
        elif method == "LCS":
            results = [lcs_compare(g[0], g[1]) for g in glosses]
        # If using a large language model method
        elif method == "LLM":
            results = [llm_compare(g[0], g[1], gloss_vec_mapping, model) for g in glosses]
        # If using any other method
        else:
            results = labels

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for check_no, check_result in enumerate(results):
        if check_result == labels[check_no] and check_result == "Related":
            tp += 1
        elif check_result == labels[check_no] and check_result == "Unrelated":
            tn += 1
        elif check_result != labels[check_no] and check_result == "Related":
            fp += 1
        elif check_result != labels[check_no] and check_result == "Unrelated":
            fn += 1

    total = tp + tn + fp + fn
    accuracy = round(((tp + tn)/total), 2)
    precision = round((tp/(tp + fp)), 2)
    recall = round((tp/(tp + fn)), 2)
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = round(2 * ((precision * recall) / (precision + recall)), 2)

    return [tp, fp, tn, fn, accuracy, precision, recall, f_measure]


def organise_output(comparison_data):
    """Takes output list from compare_glosses function and displays it in a more readable way"""

    tp, fp, tn, fn, accuracy, precision, recall, f_measure = comparison_data

    return (
        f"        TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        f"        Accuracy: {accuracy},\n        Precision: {precision},\n"
        f"        Recall: {recall},\n        f-measure: {f_measure}"
    )


def save_all_outputs():
    """Saves outputs for all models with all potential variables"""

    main_dir = os.getcwd()
    if not os.path.isdir("Model Outputs"):
        os.mkdir("Model Outputs")

    dev_set = load_gs("Gold Standard Dev.pkl")

    methods = ["ED", "LCS"]
    for method in methods:
        data = [compare_glosses(dev_set, method, cutoff_percent=i) for i in range(101)]

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save to Excel
        os.chdir(os.path.join(main_dir, "Model Outputs"))
        df.to_excel(f"{method} data.xlsx", index=False, header=False)
        os.chdir(main_dir)

    glosses_to_embed = sorted(list(set(
        [g[0] for g in dev_set] + [g[1] for g in dev_set]
    )))

    models = {
        "m1": "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin",
        "m2": "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"
    }

    for model in models:

        trans_model = SentenceTransformer(models.get(model))

        embeddings = trans_model.encode(glosses_to_embed)

        gloss_dict = dict()
        for gloss_index, gloss in enumerate(glosses_to_embed):
            gloss_dict[gloss] = embeddings[gloss_index]

        data = [compare_glosses(dev_set, "LLM", gloss_dict, trans_model, i) for i in range(101)]

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save to Excel
        os.chdir(os.path.join(main_dir, "Model Outputs"))
        df.to_excel(f"{model} data.xlsx", index=False, header=False)
        os.chdir(main_dir)

    return "Process complete!"


if __name__ == "__main__":

    dev_set = load_gs("Gold Standard Dev.pkl")

    print(organise_output(compare_glosses(dev_set, "ED", cutoff_percent=100)))
    print(organise_output(compare_glosses(dev_set, "LCS", cutoff_percent=82)))

    # Select text to embed
    glosses_to_embed = sorted(list(set(
        [g[0] for g in dev_set] + [g[1] for g in dev_set]
    )))

    # Identify models
    m1 = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin"
    m2 = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"

    llm = SentenceTransformer(m1)

    embedded_sentences = llm.encode(glosses_to_embed)
    gloss_dict = dict()
    for gloss_index, gloss in enumerate(glosses_to_embed):
        gloss_dict[gloss] = embedded_sentences[gloss_index]

    print(organise_output(compare_glosses(dev_set, "LLM", gloss_dict, llm, 55)))

    # Create a plot from the development set
    clusters = apply_clustering(embedded_sentences, "KMeans", 6)
    plot = plot_clusters(glosses_to_embed, embedded_sentences, clusters)
    plot.show()

    print(save_all_outputs())
