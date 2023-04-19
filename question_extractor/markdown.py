import os

def load_markdown_files_from_directory(directory):
    """
    Takes a folder path as input.
    Returns a list of tuples containing the file path and text for all markdown files in the input folder.
    
    Args:
        directory (str): The path to the folder containing markdown files.

    Returns:
        list of tuples: A list of tuples containing the file path (str) and the file content (str) for each markdown file.
    """
    markdown_files_data = []

    # Iterate through the files in the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            
            # Check if the file is a markdown file
            if file_name.endswith('.md'):
                file_path = os.path.join(root, file_name)
                
                # Read the file content
                with open(file_path, "r", encoding="utf-8") as file:
                    file_content = file.read()
                
                # Append the file path and content to the markdown_files_data list
                markdown_files_data.append((file_path, file_content))

    return markdown_files_data

def find_highest_markdown_heading_level(lines):
    """
    Takes a list of lines representing a markdown file as input.
    Finds the highest level of heading and returns it as an integer.
    Returns None if the text contains no headings.
    
    Args:
        lines (list of str): A list of lines in the markdown file.

    Returns:
        int or None: The highest heading level as an integer, or None if no headings are found.
    """
    highest_heading_level = None

    # Iterate through the lines in the markdown file
    for line in lines:
        
        # Check if the line starts with a heading
        if line.startswith("#"):
            
            # Calculate the heading level based on the number of '#' characters
            current_heading_level = len(line.split()[0])
            
            # Update the highest_heading_level if it is None or if the current_heading_level is higher
            if (highest_heading_level is None) or (current_heading_level < highest_heading_level):
                highest_heading_level = current_heading_level

    return highest_heading_level

def split_markdown(text):
    """
    Takes a string representation of a markdown file as input.
    Finds the highest level of heading and splits the text into sections accordingly.
    Returns a list of tuples, each containing the section title and section content.
    
    Args:
        text (str): The content of a markdown file as a single string.

    Returns:
        list of tuples: A list of tuples containing the section title (str) and section content (str).
    """
    lines = text.split('\n')

    # Remove the title heading (if present) from the text
    if (len(lines) > 0) and (lines[0].startswith('#')):
        lines = lines[1:]

    # Find the highest heading level
    highest_heading_level = find_highest_markdown_heading_level(lines)

    # If there are no headings, print a warning and return an empty list
    if highest_heading_level is None:
        # FIXME if this is ever triggered, introduce an alternative paragraph splitting method
        print(f"WARNING: Giving up on a piece of text that is too long for processing:\n```\n{text}\n```")
        return []

    # Construct the heading prefix for splitting
    headings_prefix = ("#" * highest_heading_level) + " "

    # Split the text at the highest heading level
    sections = []
    current_section_title = ''
    current_section = []

    for line in lines:
        # Check if the line starts with the highest heading level prefix
        if line.startswith(headings_prefix):
            # If the current_section is not empty, add it to the sections list
            if len(current_section) > 0:
                current_section_body = '\n'.join(current_section)
                sections.append((current_section_title, current_section_body))

                # Update the current_section_title and clear the current_section
                current_section_title = line.strip()
                current_section = []
        else:
            # Add the line to the current_section
            current_section.append(line)

    # Add the last section to the sections list (if not empty)
    if len(current_section) > 0:
        current_section_body = '\n'.join(current_section)
        sections.append((current_section_title, current_section_body))

    return sections