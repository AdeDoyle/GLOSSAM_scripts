import os


def open_xml(file_name):

    file_name = file_name + ".xml"
    cur_dir = os.getcwd()
    if file_name in os.listdir(cur_dir):
        file_path = os.path.join(cur_dir, file_name)
    else:
        raise RuntimeError(f"No file named {file_name} appears in the current directory.")

    with open(file_path, 'r', encoding="utf-8") as xml_file:
        xml_content = xml_file.read()

    return xml_content


def get_xml_lemmata(file_name):

    # Load xml file
    gold_xml = open_xml(file_name)

    # Isolate relevant contents
    text_range = gold_xml[gold_xml.find("<body>"):gold_xml.find("</body>") + len("</body>")]

    # Create a copy of the text content from the relevant range which can be altered
    text_reduce = text_range[:]

    # Remove undesirable tags from the text content
    for xml_tag in [
        "<app>", "</app>", "<add>", "</add>", "<del>", "</del>", "</foreign>", "<mentioned>", "</mentioned>", "</rdg>",
        "<subst>", "</subst>", "</quote>", "<lg>", "</lg>", "<l>", "</l>"
    ]:
        text_reduce = "".join(text_reduce.split(xml_tag))
    text_reduce = text_reduce.split("<pb")
    for tr_num, tr_split in enumerate(text_reduce):
        if tr_num != 0:
            text_reduce[tr_num] = tr_split[tr_split.find("/>") + 2:]
    text_reduce = "".join(text_reduce)
    text_reduce = text_reduce.split("<foreign")
    for tr_num, tr_split in enumerate(text_reduce):
        if tr_num != 0:
            text_reduce[tr_num] = tr_split[tr_split.find(">") + 1:]
    text_reduce = "".join(text_reduce)
    text_reduce = text_reduce.split("<rdg")
    for tr_num, tr_split in enumerate(text_reduce):
        if tr_num != 0:
            text_reduce[tr_num] = tr_split[tr_split.find(">") + 1:]
    text_reduce = "".join(text_reduce)
    text_reduce = text_reduce.split("<quote")
    for tr_num, tr_split in enumerate(text_reduce):
        if tr_num != 0:
            text_reduce[tr_num] = tr_split[tr_split.find(">") + 1:]
    text_reduce = "".join(text_reduce)
    while "<lem>" in text_reduce:
        text_reduce = text_reduce[:text_reduce.find("<lem>")] + text_reduce[text_reduce.find("</lem>") + len("</lem>"):]

    # Remove all new line characters and multiple spacing from the text contents
    text_reduce = " ".join(text_reduce.split("\n"))
    while "  " in text_reduce:
        text_reduce = " ".join(text_reduce.split("  "))

    # Add each sentence to a list of sentences
    sent_list = list()

    for _ in range(text_range.count("</ab>")):
        sentence = text_reduce[text_reduce.find("<ab "):text_reduce.find("</ab>") + len("</ab>")]
        sent_list.append(sentence)
        text_reduce = text_reduce[text_reduce.find("</ab>") + len("</ab>"):]

    # Refine the sentence list to a list of lemmata
    lem_list = list()
    for sent in sent_list:
        if '<seg type="lemma"' in sent:
            while '<seg type="lemma"' in sent:
                sent = sent[sent.find('<seg type="lemma"') + len('<seg type="lemma"'):]
                id_no = sent[sent.find("xml:id=") + len("xml:id=") + 1:sent.find(">") - 1]
                id_no = id_no.strip()
                sent = sent[sent.find(">") + 1:]
                lemma_check = sent[:sent.find("</seg>")]
                # Check for embedded lemmata tags, and handle
                if '<seg type="lemma"' in lemma_check:
                    second_close = sent.find("</seg>", sent.find("</seg>") + 1)
                    if second_close > sent.find('<seg type="lemma"'):
                        full_lemma = sent[:second_close]
                        untagged_lemma = full_lemma[:]
                        while "<" in untagged_lemma:
                            untagged_lemma = (
                                    untagged_lemma[:untagged_lemma.find("<")] +
                                    untagged_lemma[untagged_lemma.find(">") + 1:]
                            )
                        embedded = full_lemma[full_lemma.find('<seg type="lemma"'):full_lemma.find("</seg>") + 6]
                        embedded_id = embedded[embedded.find("xml:id=") + len("xml:id=") + 1:sent.find(">") - 1]
                        embedded_id = embedded_id.strip()
                        embedded = embedded[embedded.find(">") + 1:]
                        embedded_lemma = embedded[:embedded.find("</seg>")]
                        sent = sent[second_close + len("</seg>"):]
                        lem_list.append((embedded_id, embedded_lemma))
                        lem_list.append((id_no, untagged_lemma))
                    else:
                        raise RuntimeError("An unexpected error occurred with an embedded lemma tag.")
                else:
                    lemma = lemma_check
                    sent = sent[sent.find("</seg>") + len("</seg>"):]
                    lem_list.append((id_no, lemma))

    # Ensure all lemma IDs are unique
    id_list = [i[0] for i in lem_list]
    id_set = set(id_list)
    if len(id_list) != len(id_set):
        print([j for j in id_set if id_list.count(j) > 1])
        raise RuntimeError("Lemma IDs not all unique.")

    return lem_list


def get_xml_glosses(file_name):

    # Load xml file
    gold_xml = open_xml(file_name)

    # Isolate relevant contents
    text_range = gold_xml[gold_xml.find("<hi:listGlossGrp>"):gold_xml.find("</hi:listGlossGrp>") + len("</hi:listGlossGrp>")]

    # Add each gloss grouping to a list of gloss groups
    group_list = list()
    text_reduce = text_range[:]
    for _ in range(text_range.count("</hi:glossGrp>")):
        gloss_group = text_reduce[text_reduce.find("<hi:glossGrp "):text_reduce.find("</hi:glossGrp>") + len("</hi:glossGrp>")]
        group_list.append(gloss_group)
        text_reduce = text_reduce[text_reduce.find("</hi:glossGrp>") + len("</hi:glossGrp>"):]

    # for grp in group_list:
    #     print([grp])

    return ""


def gen_gs():
    """
    Generates a Gold Standard Test Set from

    :return:
    """

    # gold_xml_lemmata = get_xml_lemmata("Isidore_Gold")
    gold_gloss_data = get_xml_glosses("Isidore_Gold")

    return gold_gloss_data

# Create an algorithm to produce gold standard for glossing relationships, based on Evina’s data.
# First extract all gloss info (glosses, lemmata, IDs).
# Next extract relationship info between glosses (related gloss IDs, weights, etc.).
#     Note: w=1-4 represented as `weight="4"`


if __name__ == "__main__":

    print(gen_gs())
