import pandas as pd
import numpy as np


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


if __name__ == "__main__":

    excel_files = ["Output - LevDist.xlsx", "Output - LCS.xlsx", "Output - LLM Lat.xlsx", "Output - LLM MedLat.xlsx"]

    print(calculate_results(excel_files))
