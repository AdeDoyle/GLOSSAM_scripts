import pandas as pd
import numpy as np
import os
import random
import pickle as pkl
import re
from MakeGS import load_gs
from Apply_models import prep_files
from TextSim import ed_compare, lcs_compare, llm_compare
from sentence_transformers import SentenceTransformer


def load_excel(files):

    files_list = list()
    for file in files:
        df = pd.read_excel(file, engine='openpyxl')
        df = df.values.tolist()
        df = [i for i in df if i[6] == "Related" and not np.isnan(i[7])]
        df = [i[:6] + [int(i[7])] for i in df]
        files_list.append(df)

    return files_list


def get_corrections(files):

    models = ["".join(model.split("Output - ")) for model in files]
    models = ["".join(model.split(".xlsx")) for model in models]
    lists = load_excel(files)

    correction_errors = list()
    for file_ind, file in enumerate(lists):
        for gloss_pair in file:
            gp = gloss_pair[:6]
            gp_reversed = gloss_pair[3:6] + gp[:3]
            result = gloss_pair[6]
            for rem_ind, rem_file in enumerate(lists[file_ind + 1:]):
                for rem_gloss_pair in rem_file:
                    rp = rem_gloss_pair[:6]
                    rem_result = rem_gloss_pair[6]
                    if gp == rp or gp_reversed == rp:
                        if result != rem_result:
                            correction_errors.append(
                                f"Correction error found:\n"
                                f"  {models[file_ind]} model has relation correction,\n"
                                f"    {gp} = {result}\n"
                                f"  but {models[file_ind + 1 + rem_ind]} model has correction relation,\n"
                                f"    {rp} = {rem_result}\n"
                            )

    return correction_errors


def calculate_results(files, tolerate_discrepancies=False):

    models = ["".join(model.split("Output - ")) for model in files]
    models = ["".join(model.split(".xlsx")) for model in models]
    lists = load_excel(files)

    if not tolerate_discrepancies:
        discrepancies = get_corrections(excel_files)
        if len(discrepancies) != 0:
            disc_str = "\n".join(str(item) for item in discrepancies)
            raise RuntimeError(f"Discrepancies found between human annotations for different models:\n\n{disc_str}")

    results = dict()
    for file_ind, file in enumerate(lists):
        model = models[file_ind]
        tp = 0
        fp = 0
        for gloss_pair in file:
            result = gloss_pair[6]
            if result == 1:
                tp += 1
            elif result == 0:
                fp += 1
            else:
                raise RuntimeError(f"Unexpected annotation.\n    Expected 0 or 1, got: {result}")
        precision = tp/(tp+fp)
        results[model] = {precision}

    return results


def generate_gs(annotated_file, text_name="Unnamed Textfile"):
    """
        Turns the Priscian annotation into a "gold standard"

        outputs it in the format required by the apply_allmods function
    """

    related_glosseslist = list()
    unrelated_glosslist = list()
    df = pd.read_excel(annotated_file, engine='openpyxl')
    df = df.values.tolist()
    for gloss_pair in df:
        if gloss_pair[7] == 1:
            if gloss_pair[6] == "Related":
                related_glosseslist.append(gloss_pair[:-1])
            elif gloss_pair[6] == "Unrelated":
                unrelated_glosslist.append(gloss_pair[:-1])
        elif gloss_pair[6] == "Related":
            gloss_pair[6] = "Unrelated"
            unrelated_glosslist.append(gloss_pair[:-1])
        elif gloss_pair[6] == "Unelated":
            gloss_pair[6] = "Related"
            related_glosseslist.append(gloss_pair[:-1])
    gloss_list = related_glosseslist + unrelated_glosslist
    random.shuffle(gloss_list)
    gloss_list = [
        [["Lemma Unknown", gl[0], gl[1], gl[2]], ["Lemma Unknown", gl[3], gl[4], gl[5]], gl[6]] for gl in gloss_list
    ]

    gloss_list = [
        gp if not pd.isnull(gp[0][3]) else [[gp[0][0], gp[0][1], gp[0][2], ""], gp[1], gp[2]] for gp in gloss_list
    ]
    gloss_list = [
        gp if not pd.isnull(gp[1][3]) else [gp[0], [gp[1][0], gp[1][1], gp[1][2], ""], gp[2]] for gp in gloss_list
    ]

    gloss_list = [
        gp if " ," not in gp[0][3] else [
            [gp[0][0], gp[0][1], gp[0][2], ",".join(gp[0][3].split(" ,"))], gp[1], gp[2]] for gp in gloss_list
    ]
    gloss_list = [
        gp if " ," not in gp[1][3] else [
            gp[0], [gp[1][0], gp[1][1], gp[1][2], ",".join(gp[1][3].split(" ,"))], gp[2]] for gp in gloss_list
    ]

    gold_list = list()
    for gd in gloss_list:
        if "  " not in gd[0][3] and "  " not in gd[1][3]:
            gold_list.append(gd)
        else:
            new_gp = list()
            if "  " in gd[0][3]:
                new_gl = gd[0][3]
                while "  " in new_gl:
                    new_gl = " ".join(new_gl.split("  "))
                new_gp.append([gd[0][0], gd[0][1], gd[0][2], new_gl])
            else:
                new_gp.append(gd[0])
            if "  " in gd[1][3]:
                new_gl = gd[1][3]
                while "  " in new_gl:
                    new_gl = " ".join(new_gl.split("  "))
                new_gp.append([gd[1][0], gd[1][1], gd[1][2], new_gl])
            else:
                new_gp.append(gd[1])
            new_gp.append(gd[2])
            gold_list.append(new_gp)

    textdict = {text_name: [gp[:-1] for gp in gold_list]}
    gold_labels = [gp[-1] for gp in gold_list]

    cur_dir = os.getcwd()
    gold_dir = os.path.join(cur_dir, "similarity_models", "gold_data")

    # Save the test set to a PKL file
    with open(os.path.join(gold_dir, f'{text_name}.pkl'), 'wb') as testfile:
        pkl.dump(textdict, testfile)
    with open(os.path.join(gold_dir, f'{text_name} labels.pkl'), 'wb') as testfile:
        pkl.dump(gold_labels, testfile)

    return (f"Gold standard file created: {text_name}.pkl\n     {gold_dir}\n"
            f"Gold standard labels file created: {text_name} labels.pkl\n     {gold_dir}")


def apply_bestmod(model="default", cutoff="default", llm="default", output_filename="default",
                  normalise=False, apply_to="default"):
    """Uses the best performing model to perform text similarity analysis on glosses"""

    if apply_to == "default":
        base_text_dict = prep_files(normalise=normalise)
    else:
        base_text_dict = apply_to
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

        output_glosses = list()
        for result_index, result in enumerate(results):
            output_glosses.append(full_gloss_pairs[result_index] + [result])
        output_glosses = [
            [
                pair[0][1], pair[0][2], pair[0][3], pair[1][1], pair[1][2], pair[1][3], pair[2]
            ] for pair in output_glosses
        ]

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


def apply_allmods(normalise=False, apply_to="default", outfile_namemod="default"):
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
                              normalise=normalise, apply_to=apply_to)
        else:
            apply_bestmod(model=model_type, normalise=normalise, apply_to=apply_to)


def check_output(file_name, labels):

    cur_dir = os.getcwd()
    output_dir = os.path.join(cur_dir, "similarity_models", "models_output")
    found_files = os.listdir(output_dir)
    output_filenames = [i for i in found_files if file_name in i]
    models = [re.findall(r'\(([^)]+)\)', outfile)[0] for outfile in output_filenames]

    all_files = dict()
    files_check = list()
    for mod in models:
        for outfile in output_filenames:
            if f"({mod})" in outfile:
                df = pd.read_excel(os.path.join(output_dir, outfile), engine='openpyxl')
                df = df.values.tolist()
                all_files[mod] = [results[-1] for results in df]
                files_check.append([check_file[:-1] for check_file in df])
    if not all(sublist == files_check[0] for sublist in files_check):
        raise RuntimeError("Gloss pairs are not identical, or do not occur in identical order in files for all models.")

    all_results = dict()
    for mod in models:
        these_results = all_files.get(mod)
        if len(these_results) != len(labels):
            raise RuntimeError(f"Number of results for test file ({len(these_results)}) "
                               f"is not equal to the number of labels ({len(labels)}).")
        else:
            result_comparison = list()
            for res_index, result in enumerate(these_results):
                matched_label = labels[res_index]
                if result == matched_label:
                    if result == "Related":
                        result_comparison.append("tp")
                    elif result == "Unrelated":
                        result_comparison.append("tn")
                elif result != matched_label:
                    if result == "Related":
                        result_comparison.append("fp")
                    elif result == "Unrelated":
                        result_comparison.append("fn")
            tp = result_comparison.count("tp")
            fp = result_comparison.count("fp")
            tn = result_comparison.count("tn")
            fn = result_comparison.count("fn")

            total = tp + tn + fp + fn
            accuracy = ((tp + tn) / total)
            precision = (tp / (tp + fp))
            recall = (tp / (tp + fn))
            if precision + recall == 0:
                f_measure = 0
            else:
                f_measure = 2 * ((precision * recall) / (precision + recall))

            all_results[mod] = {
                "accuracy": round(accuracy, 4), "precision": round(precision, 4),
                "recall": round(recall, 4), "f_measure": round(f_measure, 4)
            }

    return all_results


def export_results(results):

    headings = [
        ["", "Accuracy", "Precision", "Recall", "F-Measure"]
    ]

    for mod in results:
        res = results.get(mod)
        ac = res.get("accuracy")
        pr = res.get("precision")
        rl = res.get("recall")
        fm = res.get("f_measure")
        headings.append([mod, ac, pr, rl, fm])

    df = pd.DataFrame(headings[1:], columns=headings[0])

    main_dir = os.getcwd()
    outputs_dir = os.path.join(main_dir, "similarity_models", "models_scores")
    os.chdir(outputs_dir)
    df.to_excel(f"Realworld Priscian Results.xlsx", index=False)
    os.chdir(main_dir)

    return "Process complete!"


if __name__ == "__main__":

    excel_files = ["Output - LevDist.xlsx", "Output - LCS.xlsx", "Output - LLM Lat.xlsx", "Output - LLM MedLat.xlsx"]
    test_data = load_gs("Priscian Gold Standard.pkl")
    labels = load_gs("Priscian Gold Standard labels.pkl")
    real_world_results = check_output("Priscian Gold Standard", labels)

    # print(calculate_results(excel_files))
    # print(generate_gs("Output - LLM Lat.xlsx", "Priscian Gold Standard"))
    # apply_allmods(normalise=False, apply_to=test_data)

    print(export_results(real_world_results))
