import os
import string
from bs4 import BeautifulSoup
from TextSim import ed_compare, lcs_compare, llm_compare
from sentence_transformers import SentenceTransformer
import pandas as pd


def prep_files(normalise=False):
    """
        Prepare contents of TEI encoded gloss collection files for comparison using models

        Output's a dictionary: {'text_name': [list of gloss pairs], 'text_name': [list of gloss pairs], ...}
        Lists of gloss pairs in the format: [[[gl_1 info][gl_2 info]], [[gl_1 info][gl_2 info]], ...]
        Gloss info for gl_1 and gl_2 is in the format: ['file_name', 'gloss_ID', 'lemma_id', 'gloss_text']

        e.g. {'priscian': [[['r1_lemmatised', '11v19 nn', '#II_28.7__7', 'solus'],
                            ['f3_lemmatised', '13b36 z', '#II_28.7__7', 'Greca']],
                           [['r1_lemmatised', '11v19 nn1', '#II_28.7__7', '.oros. mons uel uisio'],
                            ['f3_lemmatised', '13b36 z', '#II_28.7__7', 'Greca']], ...]}
    """

    # Map Directories
    tools_dir = os.getcwd()
    tools_parent = os.path.dirname(tools_dir)
    texts_dir = os.path.join(tools_parent, "data", "texts")

    # Create lists to collect directories for each base text and associated gloss collections
    base_texts = os.listdir(texts_dir)
    gloss_collections = list()

    for text in base_texts:
        text_path = os.path.join(texts_dir, text)
        collections_path = os.path.join(text_path, "gloss_collections")
        for gloss_coll in os.listdir(collections_path):
            if "_lemmatised.xml" in gloss_coll:
                gloss_collections.append(os.path.join(collections_path, gloss_coll))

    # Get gloss collection data for each base text, and add each separately to a dictionary
    texts_dict = dict()
    for text in base_texts:
        filetexts = list()
        for filename in gloss_collections:
            # Load files and add text (between text tags) for each collection to a list
            if filename.startswith('.'):
                continue
            elif "_lemmatised.xml" not in filename:
                continue
            elif text in filename:
                with open(filename, 'r', encoding="utf-8") as loadfile:
                    file_loaded = loadfile.read()
                filetexts.append(file_loaded[file_loaded.find("<text>"):file_loaded.find("</text>") + len("</text>")])

        # Split each collection file data into lists of data for individual glosses
        glosses = list()
        for filetext in filetexts:
            filetext = filetext[filetext.find("<note "):filetext.rfind("</note>") + len("</note>")]
            # Remove notes
            filetext_split = filetext.split("<!--")
            filetext_split = [i if "-->" not in i else i[i.find("-->") + 3:] for i in filetext_split]
            filetext = "".join(filetext_split)
            # Tidy up html
            file_glosses = [
                "<note n=" + i[:i.rfind("</note>") + len("</note>")] for i in filetext.split("<note n=") if i
            ]
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
            this_file = os.path.basename(gloss_collections[collection_index]).strip(".xml")
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

        # Replace null glosses with empty strings
        glosses_data = [gd if not pd.isnull(gd[3]) else [gd[0], gd[1], gd[2], ""] for gd in glosses_data]

        # Remove double spaces in glosses
        gloss_list = list()
        for gd in glosses_data:
            if "  " not in gd[3]:
                gloss_list.append(gd)
            else:
                new_gl = gd[3]
                while "  " in new_gl:
                    new_gl = " ".join(new_gl.split("  "))
                new_gd = [gd[0], gd[1], gd[2], new_gl]
                gloss_list.append(new_gd)

        # remove spacing before punctuation in glosses
        glosses_data = [[gd[0], gd[1], gd[2], ",".join(gd[3].split(" ,"))] for gd in gloss_list]

        # Pair glosses on the same lemma from different manuscripts, and remove all unpaired glosses
        gloss_pairs = list()
        for gl_index, gloss_datum in enumerate(glosses_data):
            for remaining_gloss in glosses_data[gl_index:]:
                if gloss_datum[0] != remaining_gloss[0] and gloss_datum[2] == remaining_gloss[2]:
                    gloss_pairs.append([gloss_datum, remaining_gloss])

        texts_dict[text] = gloss_pairs

    return texts_dict


def apply_bestmod(model="default", cutoff="default", llm="default", output_filename="default",
                  include_false=True, normalise=False):
    """Uses the best performing model to perform text similarity analysis on glosses"""

    base_text_dict = prep_files(normalise=normalise)
    base_texts = [i for i in base_text_dict if base_text_dict.get(i)]

    output_dir = os.path.join(os.getcwd(), "similarity_models", "models_output")

    for base_text in base_texts:
        full_gloss_pairs = base_text_dict.get(base_text)
        basic_gloss_pairs = [[i[0][3], i[1][3]] for i in full_gloss_pairs]

        if model == "default":
            model = "LLM"

        results = []

        if model == "ED":
            if cutoff == "default":
                cutoff = 41
            results = [ed_compare(g[0], g[1], cutoff) for g in basic_gloss_pairs]

        elif model == "LCS":
            if cutoff == "default":
                cutoff = 81
            results = [lcs_compare(g[0], g[1], cutoff) for g in basic_gloss_pairs]

        elif model == "LLM":

            if cutoff == "default":
                if llm in ["default", "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"]:
                    cutoff = 54
                elif llm == "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-medieval-latin":
                    cutoff = 49
                else:
                    raise RuntimeError("Cutoff must be specified for LLM models")
            if llm == "default":
                model_used = "silencesys/paraphrase-xlm-r-multilingual-v1-fine-tuned-for-latin"
            else:
                model_used = llm

            glosses_to_embed = sorted(list(set(
                [g[0] for g in basic_gloss_pairs] + [g[1] for g in basic_gloss_pairs]
            )))

            model_used = SentenceTransformer(model_used)
            embedded_glosses = model_used.encode(glosses_to_embed)

            gloss_dict = dict()
            for gloss_index, gloss in enumerate(glosses_to_embed):
                gloss_dict[gloss] = embedded_glosses[gloss_index]

            results = [llm_compare(g[0], g[1], gloss_dict, model_used, cutoff) for g in basic_gloss_pairs]

        related_glosses = list()
        for result_index, result in enumerate(results):
            if result == "Related":
                related_glosses.append(full_gloss_pairs[result_index])
        related_glosses = [
            [
                pair[0][1], pair[0][2], pair[0][3], pair[1][1], pair[1][2], pair[1][3], "Related"
            ] for pair in related_glosses
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
                    out_file = f"Paired Glossses for {base_text} ({model} - Text Normalised).xlsx"
                else:
                    out_file = f"Paired Glossses for {base_text} ({model}).xlsx"
            else:
                fixed_llm = "_".join(llm.split("/"))
                fixed_llm = "_".join(fixed_llm.split("\\"))
                if normalise:
                    out_file = f"Paired Glossses for {base_text} ({fixed_llm} - Text Normalised).xlsx"
                else:
                    out_file = f"Paired Glossses for {base_text} ({fixed_llm}).xlsx"
        elif "*base_text_name*" in output_filename:
            out_file = base_text.join(output_filename.split("*base_text_name*"))
            out_file = out_file + ".xlsx"
        else:
            out_file = output_filename + ".xlsx"

        df = pd.DataFrame(
            output_glosses, columns=[
                "Gl. 1 MS", "Gl. 1 no.", "Gloss 1", "Gl. 2 MS", "Gl. 2 no.", "Gloss 2", "Predicted Relationship"
            ]
        )
        df.to_excel(os.path.join(output_dir, out_file), index=False)


def apply_allmods(include_false=True, normalise=False, outfile_namemod="default"):
    """Applies all variants of all models at once, using optimised hyperparameters"""

    if outfile_namemod == "default":
        nm = ""
    else:
        nm = "(" + outfile_namemod + ") "

    for model_type in ["ED", "LCS", "LLM"]:
        cutoff = "default"
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
                    out_file = f"Paired Glossses for *base_text_name* {nm}({model_type} - {hf_model} - Text Normalised)"
                else:
                    out_file = f"Paired Glossses for *base_text_name* {nm}({model_type} - {hf_model})"
                apply_bestmod(model=model_type, cutoff=cutoff, llm=lat_mod, output_filename=out_file,
                              include_false=include_false, normalise=normalise)
        else:
            apply_bestmod(model=model_type, include_false=include_false, normalise=normalise)


if __name__ == "__main__":

    apply_allmods(include_false=False)
