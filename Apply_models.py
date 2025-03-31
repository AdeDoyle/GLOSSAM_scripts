import os


def prep_files(folder_path):
    """Prepare TEI encoded project files for comparison"""

    # Get filenames
    filenames = os.listdir(folder_path)

    # Load file data
    filetexts = list()
    for filename in filenames:
        with open(os.path.join(folder_path, filename), 'r', encoding="utf-8") as loadfile:
            file_loaded = loadfile.read()
        filetexts.append(file_loaded[file_loaded.find("<text>"):file_loaded.find("</text>") + len("</text>")])

    # Split file data into data for individual glosses
    glosses = list()
    for filetext in filetexts:
        filetext = filetext[filetext.find("<note "):filetext.rfind("</note>") + len("</note>")]
        filetext_split = filetext.split("<!--")
        filetext_split = [i if "-->" not in i else i[i.find("-->") + 3:] for i in filetext_split]
        filetext = "".join(filetext_split)
        file_glosses = ["n=" + i[:i.rfind("</note>")] for i in filetext.split("<note n=") if i]
        file_glosses = [" ".join(i.split("\n")) for i in file_glosses]
        file_glosses = [" ".join(i.split("\t")) for i in file_glosses]
        for i_indx, i in enumerate(file_glosses):
            while "  " in i:
                i = " ".join(i.split("  "))
            file_glosses[i_indx] = i.strip()
        glosses.append(file_glosses)

    prepped = glosses

    return prepped


if __name__ == "__main__":
    curdir = os.getcwd()
    filesdir = os.path.join(curdir, "Mary Testfiles")

    # prep_files(filesdir)
    print(prep_files(filesdir))
