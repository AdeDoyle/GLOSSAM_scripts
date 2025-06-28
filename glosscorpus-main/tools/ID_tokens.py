import os
import re
import string
from nltk import wordpunct_tokenize as tze


def id_tokens():
    """
    Searches for all base-text files and tokenises their contents, tagging and numbering each word.

    Creates an annotated copy of the file within the same directory as the original files..
    """

    # Search for .xml files in the base-texts directory
    xml_files = {}
    tools_dir = os.getcwd()
    tools_parent = os.path.dirname(tools_dir)
    texts_dir = os.path.join(tools_parent, "data", "texts")
    if os.path.isdir(texts_dir):  # Check if it's a directory
        base_texts = os.listdir(texts_dir)
        for base_txt in base_texts:
            txt_dir = os.path.join(texts_dir, base_txt)
            base_txt_path = os.path.join(txt_dir, "basetext.xml")
            if "_tokenised" not in base_txt_path:  # Need better way to ensure base-texts not already tokenised (!!!)
                if os.path.isfile(base_txt_path):
                    xml_files[base_txt] = base_txt_path
                else:
                    raise RuntimeError(f"Could not find file path: {base_txt_path}")

    # For each .xml file found in the base-texts directory
    for old_xml in xml_files:

        # Open and read the content of the .xml file
        with open(xml_files.get(old_xml), 'r', encoding="utf-8") as xml_file:
            content = xml_file.read()

        # Isolate the text content
        text_content = content[content.find("<body>"):content.find("</body>") + len("</body>")]
        tagset = []
        iend = 0
        for i in range(text_content.count("<")):
            istart = text_content.find("<", iend)
            iend = text_content.find(">", istart) + 1
            tagset.append(text_content[istart:iend])

        # Split the .xml file into substrings of tags and text between tags
        # This for-loop can take a particularly long time to run (e.g. on Priscian)
        textlist = []
        original_len = len(text_content)
        percent_divisor = 100/original_len
        percent_complete = 0
        if original_len >= 500000:
            print(f"Long for-loop in progress:\n    100% remaining.")
        for tag_no, found_tag in enumerate(tagset):
            find_pos = text_content.find(found_tag)

            # If the remaining text begins with a tag, add that tag to the list of substrings,
            # then remove that tag from the beginning of the remaining text
            if find_pos == 0:
                textlist.append(found_tag)
                text_content = text_content[len(found_tag):]

            # If tags remain, but the remaining text does not begin with a tag,
            # add all text until the next tag to the list of substrings,
            # then add the next tag to the list of substrings,
            # then remove both the tag and all text preceding it from the beginning of the remaining text
            else:
                pre_text = text_content[:find_pos]
                textlist.extend([pre_text, found_tag])
                text_content = text_content[len(pre_text) + len(found_tag):]

            # If text remains, but no tags remain, add the remaining text to the list of substrings
            if tag_no + 1 == len(tagset):
                if text_content:
                    textlist.append(text_content)

            if original_len >= 500000:
                percent_left = len(text_content)*percent_divisor
                exact_complete = 100-percent_left
                if exact_complete >= percent_complete + 0.1:
                    percent_complete += 0.1
                    print(f"    {round(percent_left, 2)}% remaining.")
                elif exact_complete >= 100:
                    print(f"    0% remaining.")

        # Iterate through the list of substrings, ignoring substrings containing tags
        ptlist = []
        for xml_split in textlist:
            if "<" not in xml_split:

                # If new-line characters occur in the remaining text substrings
                # (excluding substrings which contain only new-line and space characters),
                # separate these new-line characters from the rest of the substring
                if xml_split.strip() and "\n" in xml_split:
                    nl_count = xml_split.count("\n")
                    for nl_no in range(nl_count):

                        # If the first character in the substring is a new-line character,
                        # add it to the new substrings list and remove it from the current substring text
                        if xml_split[0] == "\n":
                            ptlist.append("\n")
                            xml_split = xml_split[1:]
                        # Otherwise find the index of the next new-line character in the current substring text,
                        # add the text up to that index to the new substring list, next add the new-line character too,
                        # then remove both from the current substring text
                        else:
                            next_nl = xml_split.find("\n")
                            ptlist.append(xml_split[:next_nl])
                            ptlist.append("\n")
                            xml_split = xml_split[next_nl + 1:]
                        # If no new-line characters remain in the current substring text, but text does remain,
                        # add that text to the new substrings list
                        if nl_no + 1 == nl_count and xml_split:
                            ptlist.append(xml_split)

                # Add substrings which contain no tags or text content, but do contain space or new-line characters
                # to the new substrings list without altering them
                else:
                    ptlist.append(xml_split)

            # Add substrings which contain tags to the new substrings list without altering them
            else:
                ptlist.append(xml_split)

        # Iterate through the updated list of substrings, ignoring substrings containing tags or only space characters
        pt_space_list = []
        for pt_item in ptlist:
            if pt_item.strip() and pt_item != pt_item.strip() and "<" not in pt_item:

                # If two consecutive space characters occur in any remaining text substrings,
                # separate these from the rest of the substring, then add each to a new list of substrings separately
                if "  " in pt_item and pt_item != "  ":
                    ds_count = pt_item.count("  ")
                    for ds_no in range(ds_count):

                        # If the first character in the current substring text are both spaces,
                        # add a double space substring to the new substrings list
                        # and remove them from the beginning of the current substring
                        if pt_item[0:2] == "  ":
                            pt_space_list.append("  ")
                            pt_item = pt_item[2:]
                        # Otherwise find the index of the next two consecutive space characters in the substring text,
                        # add the text up to that index to the new substring list,
                        # then add two consecutive space characters to the new substring list,
                        # then remove both the text and the two space characters from the beginning of the substring
                        else:
                            next_ds = pt_item.find("  ")
                            pt_space_list.append(pt_item[:next_ds])
                            pt_space_list.append("  ")
                            pt_item = pt_item[next_ds + 2:]
                        # If no double-spacing remains in the current substring text, but text does remain,
                        # add that text to the new substrings list
                        if ds_no + 1 == ds_count and pt_item:
                            pt_space_list.append(pt_item)

                # Add substrings which contain no double space characters to the new substrings list without changes
                else:
                    pt_space_list.append(pt_item)

            # Add substrings which contain tags to the new substrings list without altering them
            else:
                pt_space_list.append(pt_item)

        # Iterate through the updated list of substrings,
        # ignoring substrings containing tags or those which only contain space and new-line characters
        strip_list = []
        for pts_item in pt_space_list:
            if pts_item.strip() and "<" not in pts_item:

                # If any space or new-line characters remain at the beginning or end of any remaining text substrings,
                # separate these from the rest of the substring, then add each to a new list of substrings separately
                if pts_item != pts_item.strip():
                    ptstrip = pts_item.strip()
                    ptstart = pts_item.find(ptstrip)
                    ptend = ptstart + len(ptstrip)
                    strip_list.extend([pts_item[:ptstart], pts_item[ptstart:ptend], pts_item[ptend:]])

                # Otherwise, if the non-tag substring begins and ends with letter characters,
                # add it alone to the list of substrings without changing it
                else:
                    strip_list.append(pts_item)

            # Add substrings which contain tags to the new substrings list without altering them
            else:
                strip_list.append(pts_item)

        # Iterate through the stripped list of substrings
        # Look for substrings containing tags to find the section and ab keys
        # Look for substrings containing text and tokenise it
        tok_list = []
        segment_id = "null"
        for stripped_item in strip_list:

            # If the item is an xml tag
            if "<" in stripped_item:

                if 'type="section"' in stripped_item:
                    segment_id = "null"

                elif 'head' in stripped_item:
                    segment_id = "Header"

                elif 'ab xml:id="' in stripped_item:
                    key_start_index = stripped_item.find("ab xml:id=")
                    post_index_string = stripped_item[key_start_index:]
                    key_end_ex = key_start_index + post_index_string.find('"', post_index_string.find('"') + 1)
                    segment_id = stripped_item[key_start_index + 11:key_end_ex]

                tok_list.append(stripped_item)

            # If the item is not an xml tag, and not a header
            elif stripped_item.strip() and "<" not in stripped_item and segment_id not in ["null", "Header"]:

                # Tokenise text substrings, tagging each word and adding segment references to the tag
                toks_sublist = tze(stripped_item)

                # Get token indices in sentence, to preserve original spacing
                space_sublist = []
                reduce_sent = stripped_item[:]
                for tok_num, tok in enumerate(toks_sublist):
                    if tok_num == 0:
                        reduce_sent = reduce_sent[len(tok):]
                    else:
                        tok_ind = reduce_sent.find(tok)
                        if tok_ind == 0:
                            space_sublist.append("")
                        else:
                            space_sublist.append(reduce_sent[:tok_ind])
                        reduce_sent = reduce_sent[tok_ind + len(tok):]

                # Add token IDs and recombine the sentence
                toks_count = [num + 1 for num, _ in enumerate(toks_sublist)]
                toks_zip = zip(toks_sublist, [f'__{tok_id}' for tok_id in toks_count])
                toks_sublist = [
                    f'\n\t\t\t\t\t<pc xml:id="{segment_id}{tagged_tok[1]}">{tagged_tok[0]}</pc>' if all(
                        char in string.punctuation + "«»" for char in tagged_tok[0]
                    ) else f'\n\t\t\t\t\t<w xml:id="{segment_id}{tagged_tok[1]}">{tagged_tok[0]}</w>'
                    for tagged_tok in toks_zip
                ]
                for space_num, space_type in enumerate(space_sublist):
                    toks_sublist[space_num] = toks_sublist[space_num] + space_type
                toks = "".join(toks_sublist)
                tok_list.append(toks)

            else:
                tok_list.append(stripped_item)

        new_xml_body = "".join(tok_list)

        # Renumber tokens in glosses which contain problematic <title> tags
        if "<title>" in new_xml_body:

            reducing_xml_body = new_xml_body

            # Create a list of all glosses which contain problematic <title> tags
            problem_glosslist = []
            title_count = reducing_xml_body.count("<title>")
            for title_tag in range(title_count):
                title_index = reducing_xml_body.find("<title>")
                title_close_index = reducing_xml_body.find("</title>")
                problem_text = reducing_xml_body[title_index:title_close_index + len("</title>")]

                preceding = reducing_xml_body[:title_index]
                proceeding = reducing_xml_body[title_close_index + len("</title>"):]
                problem_pretext = preceding[preceding.rfind("<ab "):]
                problem_protext = proceeding[:proceeding.find("</ab>") + len("</ab>")]

                problem_gloss = problem_pretext + problem_text + problem_protext
                reducing_xml_body = reducing_xml_body[reducing_xml_body.find(problem_gloss) + len(problem_gloss):]
                problem_glosslist.append(problem_gloss)

            # Split each problematic gloss into tokens, replace the xml ID for that token, then recombine the list
            for problem_gloss in problem_glosslist:
                key_start_index = problem_gloss.find("ab xml:id=")
                post_index_string = problem_gloss[key_start_index:]
                key_end_ex = key_start_index + post_index_string.find('"', post_index_string.find('"') + 1)
                segment_id = problem_gloss[key_start_index + 11:key_end_ex] + "__"

                prob_gl_list = re.split(r"(</w>|</pc>)", problem_gloss)
                fixed_gloss_list = []
                prob_tok_num = 0
                for prob_tok in prob_gl_list:
                    if "<w " in prob_tok or "<pc " in prob_tok:
                        prob_tok_num += 1
                    if segment_id in prob_tok:
                        prob_id_index = prob_tok.find(segment_id)
                        old_gl_num = prob_tok[prob_id_index:]
                        old_gl_num = old_gl_num[:old_gl_num.find('"')]
                        new_gl_num = f"{segment_id}{prob_tok_num}"
                        prob_tok = new_gl_num.join(prob_tok.split(old_gl_num))
                    fixed_gloss_list.append(prob_tok)

                fixed_gloss = "".join(fixed_gloss_list)
                new_xml_body = fixed_gloss.join(new_xml_body.split(problem_gloss))

        new_xml_file = (
                content[:content.find("<body>")] + new_xml_body + content[content.find("</body>") + len("</body>"):]
        )

        # Create a new file path with "_tokenised" appended to the filename
        directory = os.path.join(texts_dir, old_xml)
        new_file_path = os.path.join(directory, 'basetext_tokenised.xml')  # Need better way to show base-texts tokenised (!!!)

        # Save the modified content to the new file path
        with open(new_file_path, 'w', encoding="utf-8") as new_file:
            new_file.write(new_xml_file)


if __name__ == "__main__":

    id_tokens()
