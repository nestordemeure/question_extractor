import os

def import_markdown_files(directory):
    """
    Take a folder path.
    Returns a list of (file_path, text) for all markdown files in the input folder.
    """
    files_data = []
    
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.md'):
                file_path = os.path.join(root, file_name)
                with open(file_path, "r", encoding="utf-8") as file:
                    file_content = file.read()
                files_data.append((file_path, file_content))
    return files_data

def find_highest_heading_level(lines):
    """
    Takes a string representation of a markdown file as input.
    Finds the highest level of heading and returns it as an integer.
    Returns None if the text contains no headings.
    """
    min_heading_level = None
    for line in lines:
        if line.startswith("#"):
            heading_level = len(line.split()[0])
            if (min_heading_level is None) or (heading_level < min_heading_level):
                min_heading_level = heading_level
    return min_heading_level

def split_markdown(text):
    """
    Takes a string representation of a markdown file as input.
    Finds the highest level of heading.
    Split into a list, one per heading of the given level.
    Return the list of strings.
    """
    lines = text.split('\n')
    # if the text starts with a (title) heading, trash it
    if (len(lines) > 0) and (lines[0].startswith('#')):
        lines = lines[1:]
    # finds highest heading level
    highest_heading_level = find_highest_heading_level(lines)
    if highest_heading_level is None:
        # there are no headings to be splitted at
        # FIXME if this is ever triggered, introduce an alternative splitting method
        print(f"Giving up on a piece of text that is too long for processing:\n```\n{text}\n```")
        return []
    headings_prefix = ("#" * highest_heading_level) + " "
    # split code at the found level
    sections = []
    current_section_title = ''
    current_section = []
    for line in lines:
        if line.startswith(headings_prefix):
            if len(current_section) > 0:
                current_section_body = '\n'.join(current_section)
                sections.append((current_section_title, current_section_body))
                current_section_title = line.strip()
                current_section = []
        current_section.append(line)

    if len(current_section) > 0:
        current_section_body = '\n'.join(current_section)
        sections.append((current_section_title, current_section_body))
    return sections