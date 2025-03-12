from nltk import edit_distance as ed
from MakeGS import load_gs
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
import torch
from sentence_transformers import SentenceTransformer
import pickle as pkl


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

    return (
        f"        TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n"
        f"        Accuracy: {accuracy},\n        Precision: {precision},\n"
        f"        Recall: {recall},\n        f-measure: {f_measure}"
    )


if __name__ == "__main__":

    dev_set = load_gs("Gold Standard Dev.pkl")

    print(compare_glosses(dev_set, "ED", cutoff_percent=100))
    print(compare_glosses(dev_set, "LCS", cutoff_percent=82))

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

    print(compare_glosses(dev_set, "LLM", gloss_dict, llm, 55))

    # Create a plot from the development set
    clusters = apply_clustering(embedded_sentences, "KMeans", 6)
    plot = plot_clusters(glosses_to_embed, embedded_sentences, clusters)
    plot.show()

    """
    The following results were recorded using the edit distance method to compare glosses from the development set.
    Any normalised edit distance higher than the cutoff (n) is considered to indicate related glosses, so the results
    are given for various values for n.
    
    For this task, the f-measure is more important than the other metrics.
    
    n = 1
        TP: 490, FP: 1265, TN: 1, FN: 282
        Accuracy: 0.24,
        Precision: 0.28,
        Recall: 0.63,
        f-measure: 0.39
    n = 3
        TP: 489, FP: 1265, TN: 1, FN: 283
        Accuracy: 0.24,
        Precision: 0.28,
        Recall: 0.63,
        f-measure: 0.39
    n = 5
        TP: 484, FP: 1265, TN: 1, FN: 288
        Accuracy: 0.24,
        Precision: 0.28,
        Recall: 0.63,
        f-measure: 0.39
    n = 10
        TP: 473, FP: 1265, TN: 1, FN: 299
        Accuracy: 0.23,
        Precision: 0.27,
        Recall: 0.61,
        f-measure: 0.37
    n = 15
        TP: 456, FP: 1259, TN: 7, FN: 316
        Accuracy: 0.23,
        Precision: 0.27,
        Recall: 0.59,
        f-measure: 0.37
    n = 20
        TP: 444, FP: 1258, TN: 8, FN: 328
        Accuracy: 0.22,
        Precision: 0.26,
        Recall: 0.58,
        f-measure: 0.36
    n = 30
        TP: 402, FP: 1247, TN: 19, FN: 370
        Accuracy: 0.21,
        Precision: 0.24,
        Recall: 0.52,
        f-measure: 0.33
    n = 40
        TP: 356, FP: 1218, TN: 48, FN: 416
        Accuracy: 0.2,
        Precision: 0.23,
        Recall: 0.46,
        f-measure: 0.31
    n = 50
        TP: 253, FP: 1180, TN: 86, FN: 519
        Accuracy: 0.17,
        Precision: 0.18,
        Recall: 0.33,
        f-measure: 0.23
    n = 60
        TP: 133, FP: 1085, TN: 181, FN: 639
        Accuracy: 0.15,
        Precision: 0.11,
        Recall: 0.17,
        f-measure: 0.13
    n = 70
        TP: 64, FP: 885, TN: 381, FN: 708
        Accuracy: 0.22,
        Precision: 0.07,
        Recall: 0.08,
        f-measure: 0.07
    n = 80
        TP: 15, FP: 569, TN: 697, FN: 757
        Accuracy: 0.35,
        Precision: 0.03,
        Recall: 0.02,
        f-measure: 0.02
    n = 90
        TP: 2, FP: 189, TN: 1077, FN: 770
        Accuracy: 0.53,
        Precision: 0.01,
        Recall: 0.0,
        f-measure: 0.0
    n = 100
        TP: 0, FP: 37, TN: 1229, FN: 772
        Accuracy: 0.6,
        Precision: 0.0,
        Recall: 0.0,
        f-measure: 0
    """

    """
        The following results were recorded using the lowest common substring method:

        n = 82
            TP: 661, FP: 85, TN: 1181, FN: 109
            Accuracy: 0.9,
            Precision: 0.89,
            Recall: 0.86,
            f-measure: 0.87
    """

    """
        The following results were recorded using the LLM method:

        n = 0
            TP: 770, FP: 1202, TN: 64, FN: 0
            Accuracy: 0.41,
            Precision: 0.39,
            Recall: 1.0,
            f-measure: 0.56
        n = 10
            TP: 767, FP: 1008, TN: 258, FN: 3
            Accuracy: 0.5,
            Precision: 0.43,
            Recall: 1.0,
            f-measure: 0.6
        n = 20
            TP: 756, FP: 652, TN: 614, FN: 14
            Accuracy: 0.67,
            Precision: 0.54,
            Recall: 0.98,
            f-measure: 0.7
        n = 30
            TP: 735, FP: 367, TN: 899, FN: 35
            Accuracy: 0.8,
            Precision: 0.67,
            Recall: 0.95,
            f-measure: 0.79
        n = 40
            TP: 712, FP: 184, TN: 1082, FN: 58
            Accuracy: 0.88,
            Precision: 0.79,
            Recall: 0.92,
            f-measure: 0.85
        n = 43
            TP: 706, FP: 145, TN: 1121, FN: 64
            Accuracy: 0.9,
            Precision: 0.83,
            Recall: 0.92,
            f-measure: 0.87
        n = 45
            TP: 690, FP: 126, TN: 1140, FN: 80
            Accuracy: 0.9,
            Precision: 0.85,
            Recall: 0.9,
            f-measure: 0.87
        n = 47
            TP: 677, FP: 112, TN: 1154, FN: 93
            Accuracy: 0.9,
            Precision: 0.86,
            Recall: 0.88,
            f-measure: 0.87
        n = 50
            TP: 662, FP: 100, TN: 1166, FN: 108
            Accuracy: 0.9,
            Precision: 0.87,
            Recall: 0.86,
            f-measure: 0.86
        n = 53
            TP: 640, FP: 84, TN: 1182, FN: 130
            Accuracy: 0.89,
            Precision: 0.88,
            Recall: 0.83,
            f-measure: 0.85
        n = 55
            TP: 633, FP: 69, TN: 1197, FN: 137
            Accuracy: 0.9,
            Precision: 0.9,
            Recall: 0.82,
            f-measure: 0.86
        n = 57
            TP: 623, FP: 63, TN: 1203, FN: 147
            Accuracy: 0.9,
            Precision: 0.91,
            Recall: 0.81,
            f-measure: 0.86
        n = 60
            TP: 603, FP: 51, TN: 1215, FN: 167
            Accuracy: 0.89,
            Precision: 0.92,
            Recall: 0.78,
            f-measure: 0.84
        n = 70
            TP: 542, FP: 28, TN: 1238, FN: 228
            Accuracy: 0.87,
            Precision: 0.95,
            Recall: 0.7,
            f-measure: 0.81
        n = 80
            TP: 404, FP: 12, TN: 1254, FN: 366
            Accuracy: 0.81,
            Precision: 0.97,
            Recall: 0.52,
            f-measure: 0.68
        n = 90
            TP: 329, FP: 7, TN: 1259, FN: 441
            Accuracy: 0.78,
            Precision: 0.98,
            Recall: 0.43,
            f-measure: 0.6
        n = 100
            TP: 166, FP: 3, TN: 1263, FN: 604
            Accuracy: 0.7,
            Precision: 0.98,
            Recall: 0.22,
            f-measure: 0.36
        """
