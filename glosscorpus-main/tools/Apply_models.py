import os
import string
from bs4 import BeautifulSoup
from TextSim import ed_compare, lcs_compare, llm_compare
from sentence_transformers import SentenceTransformer
import pandas as pd


def prep_files(folder_path="default", normalise=False):
    """Prepare TEI encoded project files for comparison"""

    # Get filenames
    if folder_path == "default":
        curdir = os.getcwd()
        folder_path = os.path.join(curdir, "gloss_comparisons")
    filenames = os.listdir(folder_path)

    # Load file data
    filetexts = list()
    for filename in filenames:
        if filename.startswith('.'):
            continue
        with open(os.path.join(folder_path, filename), 'r', encoding="utf-8") as loadfile:
            file_loaded = loadfile.read()
        filetexts.append(file_loaded[file_loaded.find("<text>"):file_loaded.find("</text>") + len("</text>")])

    # Split file data into lists of data for individual glosses
    glosses = list()
    for filetext in filetexts:
        filetext = filetext[filetext.find("<note "):filetext.rfind("</note>") + len("</note>")]
        # Remove notes
        filetext_split = filetext.split("<!--")
        filetext_split = [i if "-->" not in i else i[i.find("-->") + 3:] for i in filetext_split]
        filetext = "".join(filetext_split)
        # Tidy up html
        file_glosses = ["<note n=" + i[:i.rfind("</note>") + len("</note>")] for i in filetext.split("<note n=") if i]
        file_glosses = [" ".join(i.split("\n")) for i in file_glosses]
        file_glosses = [" ".join(i.split("\t")) for i in file_glosses]
        for i_indx, i in enumerate(file_glosses):
            while "  " in i:
                i = " ".join(i.split("  "))
            file_glosses[i_indx] = i.strip()
        glosses.append(file_glosses)

    # Refine the data for individual glosses
    glosses_data = list()
    for collection_index, gloss_collection in enumerate(glosses):
        this_file = filenames[collection_index].strip(".xml")
        for gloss_html in gloss_collection:
            gloss_soup = BeautifulSoup(gloss_html, 'html.parser')
            gloss_id = gloss_soup.find('note')['n']
            lemma_id = gloss_soup.find('note')['target']
            try:
                gloss_text = gloss_soup.find("gloss").text
            except AttributeError:
                continue
            glosses_data.append([this_file, gloss_id, lemma_id, gloss_text])
    if normalise:
        glosses_data = [[gd[0], gd[1], gd[2], gd[3].lower()] for gd in glosses_data]
        glosses_data = [[gd[0], gd[1], gd[2], "u".join(gd[3].split("v"))] for gd in glosses_data]
        glosses_data = [[gd[0], gd[1], gd[2], "i".join(gd[3].split("j"))] for gd in glosses_data]
        for punct in [p for p in string.punctuation + "«»"]:
            glosses_data = [[gd[0], gd[1], gd[2], "".join(gd[3].split(punct))] for gd in glosses_data]
        glosses_data = [[gd[0], gd[1], gd[2], " ".join(gd[3].split("  "))] for gd in glosses_data]

    # Pair glosses on the same lemma from different manuscripts, and remove all unpaired glosses
    gloss_pairs = list()
    for gl_index, gloss_datum in enumerate(glosses_data):
        for remaining_gloss in glosses_data[gl_index:]:
            if gloss_datum[0] != remaining_gloss[0] and gloss_datum[2] == remaining_gloss[2]:
                gloss_pairs.append([gloss_datum, remaining_gloss])

    prepped = gloss_pairs

    return prepped


def apply_bestmod(folder_path="default", model="default", cutoff="default", llm="default", output_filename="default",
                  include_false=True, normalise=False):
    """Uses the best performing model to perform text similarity analysis on glosses"""

    full_gloss_pairs = prep_files(folder_path, normalise=normalise)
    basic_gloss_pairs = [[i[0][3], i[1][3]] for i in full_gloss_pairs]

    if model == "default":
        model = "LLM"

    results = []

    if model == "ED":
        if cutoff == "default":
            cutoff = 60
        results = [ed_compare(g[0], g[1], cutoff) for g in basic_gloss_pairs]

    elif model == "LCS":
        if cutoff == "default":
            cutoff = 82
        results = [lcs_compare(g[0], g[1], cutoff) for g in basic_gloss_pairs]

    elif model == "LLM":

        if cutoff == "default":
            cutoff = 55
        if llm == "default":
            llm = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"

        glosses_to_embed = sorted(list(set(
            [g[0] for g in basic_gloss_pairs] + [g[1] for g in basic_gloss_pairs]
        )))

        llm = SentenceTransformer(llm)
        embedded_glosses = llm.encode(glosses_to_embed)

        gloss_dict = dict()
        for gloss_index, gloss in enumerate(glosses_to_embed):
            gloss_dict[gloss] = embedded_glosses[gloss_index]

        results = [llm_compare(g[0], g[1], gloss_dict, llm, cutoff) for g in basic_gloss_pairs]

    related_glosses = list()
    for result_index, result in enumerate(results):
        if result == "Related":
            related_glosses.append(full_gloss_pairs[result_index])
    related_glosses = [
        [pair[0][0], pair[0][1], pair[0][3], pair[1][0], pair[1][1], pair[1][3], "Related"] for pair in related_glosses
    ]

    output_glosses = related_glosses

    if include_false:
        unrelated_glosses = list()
        for result_index, result in enumerate(results):
            if result == "Unrelated":
                unrelated_glosses.append(full_gloss_pairs[result_index])
        unrelated_glosses = [
            [
                pair[0][0], pair[0][1], pair[0][3], pair[1][0], pair[1][1], pair[1][3], "Unrelated"
            ] for pair in unrelated_glosses
        ]
        output_glosses = related_glosses + unrelated_glosses

    if output_filename == "default":
        if llm == "default":
            if normalise:
                output_filename = f"Related Gloss Output ({model} - Text Normalised).xlsx"
            else:
                output_filename = f"Related Gloss Output ({model}).xlsx"
        else:
            fixed_llm = "_".join(llm.split("/"))
            fixed_llm = "_".join(fixed_llm.split("\\"))
            if normalise:
                output_filename = f"Related Gloss Output ({fixed_llm} - Text Normalised).xlsx"
            else:
                output_filename = f"Related Gloss Output ({fixed_llm}).xlsx"
    else:
        output_filename = output_filename + ".xlsx"

    df = pd.DataFrame(
        output_glosses, columns=[
            "Gl. 1 MS", "Gl. 1 no.", "Gloss 1", "Gl. 2 MS", "Gl. 2 no.", "Gloss 2", "Predicted Relationship"
        ]
    )
    df.to_excel(output_filename, index=False)


def apply_allmods(include_false=True, normalise=False):
    """Applies all variants of all models at once, using optimised hyperparameters"""

    for model_type in ["ED", "LCS", "LLM"]:
        if model_type == "LLM":
            for hf_model in ["Latin", "Medieval Latin"]:
                if hf_model == "Latin":
                    lat_mod = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"
                elif hf_model == "Medieval Latin":
                    lat_mod = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin"
                else:
                    hf_model = "default"
                    lat_mod = "default"
                if normalise:
                    out_file = f"Related Gloss Output ({model_type} - {hf_model} - Text Normalised)"
                else:
                    out_file = f"Related Gloss Output ({model_type} - {hf_model})"
                apply_bestmod(model=model_type, llm=lat_mod, output_filename=out_file,
                              include_false=include_false, normalise=normalise)
        else:
            apply_bestmod(model=model_type, include_false=include_false, normalise=normalise)


if __name__ == "__main__":

    apply_allmods(include_false=False)
