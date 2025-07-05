import os
import random
import pickle as pkl


def open_xml(file_name):

    file_name = file_name + ".xml"
    cur_dir = os.getcwd()
    gold_dir = os.path.join(cur_dir, "similarity_models", "gold_data")
    if not os.path.isdir("similarity_models"):
        os.mkdir("similarity_models")
    if not os.path.isdir(gold_dir):
        os.mkdir(gold_dir)
    if file_name in os.listdir(gold_dir):
        file_path = os.path.join(gold_dir, file_name)
    else:
        raise RuntimeError(f"No file named {file_name} appears in the directory {gold_dir}.")

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

    # Isolate relevant text contents from XML file
    text_range = gold_xml[
                 gold_xml.find("<hi:listGlossGrp>"):gold_xml.find("</hi:listGlossGrp>") + len("</hi:listGlossGrp>")
                 ]

    # Add each gloss grouping to a list of gloss groups
    group_list = list()
    text_reduce = text_range[:]
    for _ in range(text_range.count("</hi:glossGrp>")):
        gloss_group = text_reduce[
                      text_reduce.find("<hi:glossGrp "):text_reduce.find("</hi:glossGrp>") + len("</hi:glossGrp>")
                      ]
        group_list.append(gloss_group)
        text_reduce = text_reduce[text_reduce.find("</hi:glossGrp>") + len("</hi:glossGrp>"):]

    # Refine the gloss groups to separate lists of glosses and gloss clusters
    gloss_list = list()
    clusters = list()
    for grp in group_list:
        grp_reduce = grp[:]
        if '<gloss ' in grp_reduce:

            # Remove undesirable tags from the text content
            for xml_tag in [
                "<add>", "</add>", "<choice>", "</choice>", "<corr>", "</corr>", "<del>", "</del>", "</seg>", "<sic>",
                "</sic>", "<subst>", "</subst>", "<supplied>", "</supplied>", "<unclear>", "</unclear>"
            ]:
                grp_reduce = "".join(grp_reduce.split(xml_tag))

            grp_reduce = grp_reduce.split("<gap")
            for gr_num, gr_split in enumerate(grp_reduce):
                if gr_num != 0:
                    grp_reduce[gr_num] = gr_split[gr_split.find("/>") + 2:]
            grp_reduce = "".join(grp_reduce)
            grp_reduce = grp_reduce.split("<seg")
            for gr_num, gr_split in enumerate(grp_reduce):
                if gr_num != 0:
                    grp_reduce[gr_num] = gr_split[gr_split.find(">") + 1:]
            grp_reduce = "".join(grp_reduce)
            while "<note>" in grp_reduce:
                grp_reduce = (
                        grp_reduce[:grp_reduce.find("<note>")] +
                        grp_reduce[grp_reduce.find("</note>") + len("</note>"):]
                )

            # Remove all new line characters and multiple spacing from the text contents
            grp_reduce = " ".join(grp_reduce.split("\n"))
            while "  " in grp_reduce:
                grp_reduce = " ".join(grp_reduce.split("  "))

            while '<gloss ' in grp_reduce:
                grp_reduce = grp_reduce[grp_reduce.find('<gloss ') + len('<gloss '):]
                id_no = grp_reduce[grp_reduce.find("xml:id=") + len("xml:id=") + 1:grp_reduce.find('" corresp')]
                id_no = id_no.strip()
                grp_reduce = grp_reduce[grp_reduce.find(">") + 1:]
                gloss_text = grp_reduce[:grp_reduce.find("</gloss>")]
                gloss_text = gloss_text.strip()
                grp_reduce = grp_reduce[grp_reduce.find("</gloss>") + len("</gloss>"):]
                gloss_list.append((id_no, gloss_text))
        grp_reduce = grp[:]
        if '<hi:glossCluster ' in grp_reduce:
            while '<hi:glossCluster ' in grp_reduce:
                grp_reduce = grp_reduce[grp_reduce.find('<hi:glossCluster ') + len('<hi:glossCluster '):]
                grp_reduce = grp_reduce[grp_reduce.find('>') + 1:]
                this_group = grp_reduce[:grp_reduce.find('</hi:glossCluster>')]
                this_group = this_group.strip()
                grouped_glosses = list()
                while '<ptr ' in this_group:
                    this_group = this_group[this_group.find('<ptr '):]
                    id_no = this_group[this_group.find("target=") + len("target=") + 2:this_group.find("/>") - 1]
                    grouped_glosses.append(id_no)
                    this_group = this_group[this_group.find("/>") + len("/>"):]
                grp_reduce = grp_reduce[grp_reduce.find('</hi:glossCluster>') + len('</hi:glossCluster>'):]
                clusters.append(grouped_glosses)

    # Ensure all lemma IDs are unique
    id_list = [i[0] for i in gloss_list]
    id_set = set(id_list)
    if len(id_list) != len(id_set):
        print([j for j in id_set if id_list.count(j) > 1])
        raise RuntimeError("Gloss IDs not all unique.")
    else:
        gloss_dict = {i[0]: i[1] for i in gloss_list}

    return [gloss_dict, clusters]


def gen_gs(development_set=True, verbose=True):
    """
    Generates a Gold Standard Test Set (optionally including a development set) from Isidore XML file

    :return:
    """

    # Get glosses
    gold_gloss_data = get_xml_glosses("Isidore_Gold")
    gloss_dict = gold_gloss_data[0]
    gloss_clusters = gold_gloss_data[1]

    # Split related glosses into pairs
    gloss_pairs = list()
    for cl in gloss_clusters:
        if len(cl) == 2:
            gloss_pairs.append(cl)
        elif len(cl) > 2:
            new_pairs = list()
            for id_no, id_label in enumerate(cl):
                for id2_no, id2_label in enumerate(cl):
                    if id2_no > id_no:
                        new_pairs.append([id_label, id2_label])
            for np in new_pairs:
                gloss_pairs.append(np)

    # Get the text of related glosses, and compile into a list
    related_glosses = [[gloss_dict.get(gl_id) for gl_id in gl_pair] for gl_pair in gloss_pairs]
    related_glosses = [gl_pair for gl_pair in related_glosses if gl_pair[0] != '' and gl_pair[1] != '']

    for i in gloss_pairs:
        if len(i) != 2:
            raise RuntimeError

    # Compile a list of glosses on the same lemma which may or may not be related (later remove related glosses)
    lemma_ids = [i[:i.find("_")] for i in gloss_dict]
    multi_lemma_ids = [i for i in lemma_ids if lemma_ids.count(i) > 1]
    ml_id_set = list()
    for ml_id in multi_lemma_ids:
        if ml_id not in ml_id_set:
            ml_id_set.append(ml_id)
    multi_lemma_gloss_ids = [i for i in gloss_dict if i[:i.find("_")] in ml_id_set]
    unrelated_gl_groups = list()
    for lemma_id in ml_id_set:
        this_gl_group = list()
        for gl_id in multi_lemma_gloss_ids:
            if gl_id[:gl_id.find("_")] == lemma_id:
                this_gl_group.append(gl_id)
        unrelated_gl_groups.append(this_gl_group)
    # Split gloss groups into pairs glosses which share a single lemma
    unrelated_gl_pairs = list()
    for gr in unrelated_gl_groups:
        if len(gr) == 2:
            unrelated_gl_pairs.append(gr)
        elif len(gr) > 2:
            new_pairs = list()
            for id_no, id_label in enumerate(gr):
                for id2_no, id2_label in enumerate(gr):
                    if id2_no > id_no:
                        new_pairs.append([id_label, id2_label])
            for np in new_pairs:
                unrelated_gl_pairs.append(np)

    # Get the text of related glosses, and compile into a list
    unrelated_gl_texts = [[gloss_dict.get(gl_id) for gl_id in gl_pair] for gl_pair in unrelated_gl_pairs]

    # Remove related glosses from unrelated gloss list
    unrelated_gl_texts = [
        i for i in unrelated_gl_texts if i not in related_glosses and [
            i[1], i[0]
        ] not in related_glosses and i[0] != '' and i[1] != ''
    ]

    # Add Labels to identify related and unrelated glosses
    related_glosses = [i + ["Related"] for i in related_glosses]
    unrelated_gl_texts = [i + ["Unrelated"] for i in unrelated_gl_texts]

    # Output stats for glosses
    if verbose:
        onepc = 100 / (len(related_glosses) + len(unrelated_gl_texts))
        print(f"Related gloss pairs: {len(related_glosses)}, {round(onepc * len(related_glosses), 2)}%")
        print(f"Unrelated gloss pairs: {len(unrelated_gl_texts)}, {round(onepc * len(unrelated_gl_texts), 2)}%")

    # Shuffle related glosses and unrelated glosses separately
    random.shuffle(related_glosses)
    random.shuffle(unrelated_gl_texts)

    if development_set:
        # Split each gloss-pair list in half
        dev_related = related_glosses[:int(len(related_glosses) / 2)]
        dev_unrelated = unrelated_gl_texts[:int(len(unrelated_gl_texts) / 2)]
        test_related = related_glosses[int(len(related_glosses) / 2):]
        test_unrelated = unrelated_gl_texts[int(len(unrelated_gl_texts) / 2):]

        # Create development set by combining half of related and half of unrelated gloss pairs
        dev_set = dev_related + dev_unrelated
        # Shuffle related and unrelated glosses
        random.shuffle(dev_set)

        # Create test set by combining half of related and half of unrelated gloss pairs
        test_set = test_related + test_unrelated
        # Shuffle related and unrelated glosses
        random.shuffle(test_set)

    else:
        # Create test set by combining half of related and half of unrelated gloss pairs
        test_set = related_glosses + unrelated_gl_texts
        # Shuffle related and unrelated glosses
        random.shuffle(test_set)

    # Remove any null text values form the test set
    test_set = [i for i in test_set if i[0] and i[1]]

    cur_dir = os.getcwd()
    gold_dir = os.path.join(cur_dir, "similarity_models", "gold_data")

    # Save the test set to a PKL file
    with open(os.path.join(gold_dir, 'Gold Standard Test.pkl'), 'wb') as testfile:
        pkl.dump(test_set, testfile)

    if development_set:

        # Remove any null text values form the dev set
        dev_set = [i for i in dev_set if i[0] and i[1]]

        # Save the test set to a PKL file
        with open(os.path.join(gold_dir, 'Gold Standard Dev.pkl'), 'wb') as devfile:
            pkl.dump(dev_set, devfile)


def load_gs(gs_file):
    """
    Load saved PKL files
    """

    cur_dir = os.getcwd()
    gold_dir = os.path.join(cur_dir, "similarity_models", "gold_data")

    # Loading the PKL file
    with open(os.path.join(gold_dir, gs_file), 'rb') as loadfile:
        file_loaded = pkl.load(loadfile)

    return file_loaded


if __name__ == "__main__":

    # # Generate new Dev and Test files
    # gen_gs()

    # Load premade Dev and Test files
    print(load_gs('Gold Standard Test.pkl'))
    print(load_gs('Gold Standard Dev.pkl'))
