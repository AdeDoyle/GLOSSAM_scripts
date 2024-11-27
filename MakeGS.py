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

    # Add each sentence to a list of sentences
    sent_list = list()
    text_reduce = text_range[:]
    for _ in range(text_range.count("</ab>")):
        sentence = text_reduce[text_reduce.find("<ab "):text_reduce.find("</ab>") + len("</ab>")]
        sent_list.append(sentence)
        text_reduce = text_reduce[text_reduce.find("</ab>") + len("</ab>"):]

    # Refine the sentence list to a list of lemmata
    lem_list = list()
    for sent in sent_list:
        if '<seg type="lemma"' in sent:
            for _ in range(sent.count('<seg type="lemma"')):
                sent = sent[sent.find('<seg type="lemma"') + len('<seg type="lemma"'):]
                id_no = sent[sent.find("xml:id=") + len("xml:id="):sent.find(">")]
                id_no = id_no.strip()
                sent = sent[sent.find(">") + 1:]
                lemma = sent[:sent.find("</seg>")]
                sent = sent[sent.find("</seg>") + len("</seg>"):]
                lem_list.append((id_no, lemma))

    for lem in lem_list:
        print(lem)

    return ""


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

    gold_xml_lemmata = get_xml_lemmata("Isidore_Gold")
    gold_gloss_data = get_xml_glosses("Isidore_Gold")

    return gold_gloss_data

# Create an algorithm to produce gold standard for glossing relationships, based on Evina’s data.
# First extract all gloss info (glosses, lemmata, IDs).
# Next extract relationship info between glosses (related gloss IDs, weights, etc.).
#     Note: w=1-4 represented as `weight="4"`


if __name__ == "__main__":

    print(gen_gs())
