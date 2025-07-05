import pandas as pd
import numpy as np
import os
import random
import pickle as pkl
from MakeGS import load_gs


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


def generate_gs(annotated_file):

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
    gloss_list = [[gl[2], gl[5], gl[6]] for gl in gloss_list]

    gloss_list = [gl if not pd.isnull(gl[0]) else ["", gl[1], gl[2]] for gl in gloss_list]
    gloss_list = [gl if not pd.isnull(gl[1]) else [gl[0], "", gl[2]] for gl in gloss_list]

    gold_list = list()
    for gd in gloss_list:
        new_gd = list()
        if "  " not in gd[0]:
            new_gd.append(gd[0])
        else:
            new_gd0 = gd[0]
            while "  " in new_gd0:
                new_gd0 = " ".join(new_gd0.split("  "))
            new_gd.append(new_gd0)
        if "  " not in gd[1]:
            new_gd.append(gd[1])
        else:
            new_gd1 = gd[1]
            while "  " in new_gd1:
                new_gd1 = " ".join(new_gd1.split("  "))
            new_gd.append(new_gd1)
        new_gd.append(gd[2])
        gold_list.append(new_gd)
    gold_list = [[",".join(gd[0].split(" ,")), ",".join(gd[1].split(" ,")), gd[2]] for gd in gold_list]

    cur_dir = os.getcwd()
    gold_dir = os.path.join(cur_dir, "similarity_models", "gold_data")

    # Save the test set to a PKL file
    with open(os.path.join(gold_dir, 'Priscian Gold Standard.pkl'), 'wb') as testfile:
        pkl.dump(gold_list, testfile)

    # return f"Gold standard file created: Priscian Gold Standard.pkl\n     {gold_dir}"
    return gold_list


if __name__ == "__main__":

    excel_files = ["Output - LevDist.xlsx", "Output - LCS.xlsx", "Output - LLM Lat.xlsx", "Output - LLM MedLat.xlsx"]

    # print(calculate_results(excel_files))
    # print(generate_gs("Output - LLM Lat.xlsx"))
    print(load_gs("Priscian Gold Standard.pkl"))
