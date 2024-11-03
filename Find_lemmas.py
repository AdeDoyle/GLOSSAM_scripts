import os
import fnmatch
from nltk import edit_distance as ed


def norm_ld(s1, s2):
    """Calculate the percentage difference between two strings"""

    lev_dist = ed(s1, s2)  # Get the edit distance

    l1 = len(s1)
    l2 = len(s2)
    max_dif = max(l1, l2)  # Find the length of the larger of the two strings (this is the max possible edit distance)

    lev_norm = (lev_dist/max_dif)*100  # Normalise the edit distance, then render as a percentage of difference
    lev_norm = round(lev_norm, 2)  # Round to two decimal places

    return lev_norm


def find_lems():
    """
    Searches for all tokenised .xml files in the first-level subdirectories of the directory this .py file is placed in,
    and links individual glosses to the lemmata.

    Where it is not possible to find the correct lemma, it marks the lemma as unknown.
    """

    # Search for tokenised .xml files in the base-texts directory
    base_files = {}
    base_path = os.path.join(os.getcwd(), "basetexts")
    if os.path.isdir(base_path):  # Check if it's a directory
        for filename in fnmatch.filter(os.listdir(base_path), '*.xml'):
            if "_tokenised.xml" in filename:
                file_path = os.path.join(base_path, filename)
                if os.path.isfile(file_path):
                    base_files[filename[:filename.find("_tokenised.xml")]] = file_path

    # For each .xml file found in the first-level subdirectories
    for basefile in base_files:

        # Open and read the content of the base-text .xml file
        with open(base_files.get(basefile), 'r', encoding="utf-8") as base_xml_file:
            base_content = base_xml_file.read()
        base_content_text = base_content[base_content.find("<body>"):base_content.find("</body>") + len("</body>")]

        # Collect paths to gloss collections' locations
        gloss_collections = []
        gloss_path = os.path.join(os.getcwd(), "collections")
        if os.path.isdir(gloss_path):  # Check if it's a directory
            for foldername in os.listdir(gloss_path):
                this_gloss_path = os.path.join(gloss_path, foldername)
                if os.path.isdir(this_gloss_path):  # Check if it's a directory
                    for filename in fnmatch.filter(os.listdir(this_gloss_path), '*.xml'):
                        if "_lemmatised" not in filename:
                            gloss_file_path = os.path.join(this_gloss_path, filename)
                            if os.path.isfile(gloss_file_path):
                                gloss_collections.append(gloss_file_path)

        # For each collection of glosses found
        for gloss_file in gloss_collections:

            # Open and read the content of the xml file
            with open(gloss_file, 'r', encoding="utf-8") as gloss_xml_file:
                gloss_content = gloss_xml_file.read()

            # Determine the base-text to which this collection relates
            match_name = gloss_content[gloss_content.find("xml:base=") + 10:]
            match_name = match_name[:match_name.find("\n") - 1]
            match_name = match_name[match_name.rfind("basetexts") + 10:match_name.rfind(".xml")]

            # If the current base-text is not a match for the base-text identified in this gloss collection, move on
            if match_name != basefile:
                continue
            # Otherwise, isolate the text content between the <body> tags
            else:
                gloss_content_text = gloss_content[
                                     gloss_content.find("<body>"):gloss_content.find("</body>") + len("</body>")
                                     ]

            # This section is for testing only. It eliminates all non-test collections. Remove after testing.
            select_collections = [  # Remove to apply to all collections
                "C:\\Users\\admd9\\PycharmProjects\\GLOSSAM_scripts\\collections\\priscian\\E BK1.xml",
                "C:\\Users\\admd9\\PycharmProjects\\GLOSSAM_scripts\\collections\\priscian\\K BK1.xml",
                "C:\\Users\\admd9\\PycharmProjects\\GLOSSAM_scripts\\collections\\priscian\\L BK1.xml"
            ]
            if match_name != "priscian" or gloss_file not in select_collections:
                continue

            # Find each sequential <note> tag which occurs in the text file (these are used to identify lemmata)
            tagset = []
            iend = 0
            for _ in range(gloss_content_text.count("<")):
                istart = gloss_content_text.find("<", iend)
                iend = gloss_content_text.find(">", istart) + 1
                tagset.append(gloss_content_text[istart:iend])
            gloss_tags = [tag for tag in tagset if '<note ' in tag]

            # Create an empty text list to build up with altered gloss content, to be recompiled into the text later
            textlist = []
            # Create a copy of the text content for the gloss which can be altered without affecting hte original text
            reduce_text = gloss_content_text[:]
            # Track the current base-text segment and the current token number, start out with null values
            cur_seg_id = ""
            cur_tok_no = None

            # For each <note> tag in the text (i.e. for each lemma identified in the text)
            for tag_no, found_tag in enumerate(gloss_tags):
                find_pos = reduce_text.find(found_tag)
                end_pos = reduce_text[find_pos:].find("</note>") + len("</note>") + find_pos
                found_gloss = reduce_text[find_pos:end_pos]

                # If no lemma is identified by a gloss
                if found_gloss.count("<term>") == 0:
                    fg_lemma = "No Lemma Tagged for Gloss"
                    ft_lemRef = ""

                # If one or more lemmata are identified by a gloss
                else:
                    # Isolate the lemma identified by the gloss
                    fg_lemma = found_gloss[found_gloss.find("<term>") + len("<term>"):found_gloss.find("</term>")]
                    # Remove undesirable characters and tags from the identified lemma
                    fg_lemma = "".join(fg_lemma.split("["))
                    fg_lemma = "".join(fg_lemma.split("]"))
                    fg_lemma = "".join(fg_lemma.split("*"))
                    fg_lemma = "".join(fg_lemma.split("+"))
                    fg_lemma = "".join(fg_lemma.split("("))
                    fg_lemma = "".join(fg_lemma.split(")"))
                    fg_lemma = "".join(fg_lemma.split("<add>"))
                    fg_lemma = "".join(fg_lemma.split("</add>"))
                    fg_lemma = "".join(fg_lemma.split("<ex>"))
                    fg_lemma = "".join(fg_lemma.split("</ex>"))
                    fg_lemma = "".join(fg_lemma.split(".."))
                    if "<del>" in fg_lemma:
                        fg_lemma = fg_lemma[:fg_lemma.find("<del>")] + fg_lemma[fg_lemma.find("</del>") + 6:]
                    # Replace abbreviations with full forms in the identified lemma
                    fg_lemma = "et".join(fg_lemma.split("⁊"))
                    # Remove any spacing at the beginning/end of the identified lemma
                    fg_lemma = fg_lemma.strip()
                    # Render the identified lemma in lower-case
                    fg_lemma = fg_lemma.lower()

                    # Isolate the target line in the base-text identified by the gloss
                    ft_lemRef = found_tag[found_tag.find('target="'):]
                    ft_lemRef = 'xml:id="' + ft_lemRef[:ft_lemRef.find('"', ft_lemRef.find('"') + 1) + 1][9:]

                # If this is a second, third, etc. gloss on a base-text segment referenced already by an earlier gloss
                if cur_seg_id == ft_lemRef[len('xml:id="'):-1]:
                    new_base_segment = False
                # If this is a new base-text segment, update the current segment identifier
                else:
                    new_base_segment = True
                    cur_seg_id = ft_lemRef[len('xml:id="'):-1]

                # Using the target found above, find the correct text segment in the base-text
                lemma_line = base_content_text[base_content_text.find(ft_lemRef) + len(ft_lemRef):]
                lemma_line = lemma_line[lemma_line.find("<w "):lemma_line.find("</ab>")]
                # Replace abbreviations with full forms in the text segment
                lemma_line = "et".join(lemma_line.split("⁊"))
                # Split the text segment into separate words
                lemma_list = lemma_line.split("</w>")
                lemma_list = [lem[lem.find("<w "):] for lem in lemma_list if lem not in ["", " "]]

                # Create a lookup list of tuples containing tokens from the base-text, their xml id numbers, and index
                # Put the token itself in lower-case and swap out v's for u's
                lemma_lookup = [
                    (
                        "u".join(lem[lem.find(">") + 1:].split("v")).lower(),
                        lem[lem.find("<"):lem.find(">") + 1],
                        i_lem + 1
                    ) for i_lem, lem in enumerate(lemma_list)
                ]
                # Remove undesirable characters from the base-text lookup list
                unds_list = [".", ",", ":", ";", "«", "»"]
                lemma_lookup = [fresh_lem for fresh_lem in lemma_lookup if fresh_lem[0] not in unds_list]

                # If this is the same base-text segment as last time,
                # Remove lemmata up to the last matched token from the lookup list
                if not new_base_segment:
                    if cur_tok_no:
                        lemma_lookup = [fresh_lem for fresh_lem in lemma_lookup if fresh_lem[2] >= cur_tok_no]
                    else:
                        raise RuntimeError("Expected current token number from last pass")
                # If this is a different base-text segment to last time, reset the token counter
                else:
                    cur_tok_no = None

                lemma_key = "None Found"
                lemma_note = None

                # If no corresponding text segment can be found in the base-text
                if ft_lemRef and not lemma_line:
                    lemma_note = f"<!-- re-examine: No line with ID {ft_lemRef} occurs in the base-text -->"

                # If no <term></term> tags occur within the gloss
                elif fg_lemma == "No Lemma Tagged for Gloss":
                    lemma_note = f"<!-- re-examine: {fg_lemma} -->"

                # If only one match for the stated lemma occurs in the segment from the base text
                elif lemma_line.lower().count(f">{fg_lemma}</w>") == 1:
                    for lem_tok_inst in lemma_lookup:
                        lem_tok = lem_tok_inst[0]
                        # If a base-text token matches the lemma exactly
                        if lem_tok == fg_lemma:
                            lemma_key = lem_tok_inst[1]
                            lemma_key = lemma_key[lemma_key.find('xml:id="') + len('xml:id="'):]
                            lemma_key = f"#{lemma_key[:lemma_key.find('"')]}"
                            cur_tok_no = int(lemma_key[lemma_key.find("__")+2:])
                            break

                # If no match for the stated lemma occurs in the segment from the base text
                elif lemma_line.lower().count(f">{fg_lemma}</w>") == 0:

                    # First check to see if standardising u/v allows a match to be found
                    u_match = False
                    this_fg_lemma = "u".join(fg_lemma.split("v"))
                    for lem_tok_inst in lemma_lookup:
                        lem_tok = lem_tok_inst[0]
                        if lem_tok == this_fg_lemma:
                            u_match = True
                            lemma_key = lem_tok_inst[1]
                            lemma_key = lemma_key[lemma_key.find('xml:id="') + len('xml:id="'):]
                            lemma_key = f"#{lemma_key[:lemma_key.find('"')]}"
                            cur_tok_no = int(lemma_key[lemma_key.find("__") + 2:])
                            break

                    # If no match is found still, use Levenshtein distance to find the closest match
                    if not u_match:
                        lev_list = [
                            (
                                ed(this_fg_lemma, lem[0]), norm_ld(this_fg_lemma, lem[0]), lem
                            ) for lem in lemma_lookup
                        ]
                        min_lev = min(lev_list, key=lambda t: t[0])
                        min_lev_norm = min_lev[1]
                        min_lev_info = min_lev[2][1]
                        lemma_key = min_lev_info[min_lev_info.find('xml:id="') + len('xml:id="'):]
                        lemma_key = f"#{lemma_key[:lemma_key.find('"')]}"
                        lemma_note = (f"<!-- re-examine: Levenshtein distance used to find the closest match - "
                                      f"Edit distance: {min_lev[0]} ('{this_fg_lemma}' to '{min_lev[2][0]}') - "
                                      f"Difference: {min_lev_norm}% -->")
                        cur_tok_no = int(lemma_key[lemma_key.find("__") + 2:])

                # If more than one match for the stated lemma occurs in the segment from the base text
                # Match the token to all possible lemmata
                elif lemma_line.lower().count(f">{fg_lemma}</w>") > 1:
                    for lem_tok_inst in lemma_lookup:
                        lem_tok = lem_tok_inst[0]
                        lem_id = lem_tok_inst[1]
                        # If this is the first lemma matched in the base-text segment
                        if lem_tok == fg_lemma and lemma_key == "None Found":
                            lemma_key = lem_id[lem_id.find('xml:id="') + len('xml:id="'):]
                            lemma_key = f"#{lemma_key[:lemma_key.find('"')]}"
                            cur_tok_no = int(lemma_key[lemma_key.find("__") + 2:])
                        # If this is not the first lemma matched in the base-text segment
                        elif lem_tok == fg_lemma and "#" in lemma_key:
                            additional_lemma_key = lem_id[lem_id.find('xml:id="') + len('xml:id="'):]
                            lemma_key = f"{lemma_key} #{additional_lemma_key[:additional_lemma_key.find('"')]}"
                        lemma_note = "<!-- re-examine: Multiple possible matches found -->"

                split_tag = found_tag.split('target="')
                split_tag[1] = split_tag[1][split_tag[1].find('"') + 1:]
                updated_tag = split_tag[0] + f'target="{lemma_key}"' + split_tag[1]
                updated_gloss = updated_tag + found_gloss[len(found_tag):]
                if lemma_note:
                    if "No Lemma Tagged for Gloss" in lemma_note:
                        updated_gloss = found_gloss
                    updated_gloss = lemma_note + "\n\t\t\t\t" + updated_gloss

                textlist.append(reduce_text[:find_pos])
                textlist.append(updated_gloss)
                reduce_text = reduce_text[end_pos:]

                if tag_no + 1 == len(gloss_tags):
                    if reduce_text:
                        textlist.append(reduce_text)

            new_xml_body = "".join(textlist)
            new_xml_file = (
                    gloss_content[:gloss_content.find("<body>")] +
                    new_xml_body +
                    gloss_content[gloss_content.find("</body>") + len("</body>"):]
            )

            # Create a new file path with "_lemmatised" appended to the filename
            directory, filename = os.path.split(gloss_file)
            new_filename = os.path.splitext(filename)[0] + '_lemmatised.xml'
            new_file_path = os.path.join(directory, new_filename)

            # Save the modified content to the new file path
            with open(new_file_path, 'w', encoding="utf-8") as new_file:
                new_file.write(new_xml_file)


if __name__ == "__main__":

    find_lems()