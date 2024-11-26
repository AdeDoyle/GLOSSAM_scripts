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

    for sent in sent_list:
        print([sent])

    return ""


def gen_gs():
    """
    Generates a Gold Standard Test Set from

    :return:
    """

    # Load xml file
    gold_xml = get_xml_lemmata("Isidore_Gold")

    return gold_xml

# Create an algorithm to produce gold standard for glossing relationships, based on Evina’s data.
# First extract all gloss info (glosses, lemmata, IDs).
# Next extract relationship info between glosses (related gloss IDs, weights, etc.).
#     Note: w=1-4 represented as `weight="4"`


if __name__ == "__main__":

    print(gen_gs())
