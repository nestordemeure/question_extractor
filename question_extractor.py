import os
import re
from pathlib import Path
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

#----------------------------------------------------------------------------------------
# PARAMETERS

input_folder = Path('./input')
output_folder = Path('./output')

#----------------------------------------------------------------------------------------
# IMPORT

def read_files(directory):
    """
    Take a folder path.
    Returns a list of (file_name, file_path, text) for all files in the input folder.
    """
    files_data = []
    
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
            
            files_data.append((file_name, file_path, file_content))
            print(f"Loaded '{file_path}'")

    return files_data

print("Loading inputs.")
inputs = read_files(input_folder)
print(f"Loaded {len(inputs)} inputs.")

#----------------------------------------------------------------------------------------
# MARKDOWN PROCESSING

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
        # TODO use an alternative splitting method
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

#----------------------------------------------------------------------------------------
# QUESTION EXTRACTION

# properties of the model used
model_type='gpt-3.5-turbo'
model_token_limit=4096
model_tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
model_tokens_per_name = -1  # if there's a name, the role is omitted

def messages_of_text(text):
    """
    Takes a piece of text and returns a list of messages designed to extracts questions from the text.
    """
    system_message = SystemMessage(content="You are an expert user extracting information to quizz people on documentation. You will be passed a page extracted from the documentation, write a numbered list of questions that can be answered based *solely* on the given text.")
    human_message = HumanMessage(content=text)
    return [system_message, human_message]

def compute_nb_tokens(text, messages):
    """
    Counts the number of tokens needed to encode the list of messages.
    Adapted from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    # settings of the tokeniser for the given model
    encoding = tiktoken.encoding_for_model(model_type)
    # count number of tokens for the text
    nb_tokens_text = len(encoding.encode(text))
    # count number of tokens for the messages
    nb_token_messages = 0
    for message in messages:
        nb_token_messages += model_tokens_per_message 
        nb_token_messages += len(encoding.encode(message.content))
        nb_token_messages += model_tokens_per_name
    nb_token_messages += 3  # every reply is primed with <|start|>assistant<|message|>
    return nb_tokens_text, nb_token_messages

def compute_nb_tokens_answer(nb_tokens_text, nb_token_messages):
    """
    Takes information on the size fo the text and messages.
    Returns None if there are not enough tokens left to answer.
    Otherwise returns the number of tokens that can be requested from the model.

    NOTE data gathered on test files:
    - nb_questions:127
    - total_question_size:1940 
    - average question size:15.28
    - nb_texts:17
    - total_texts_size:13736
    - ratio question/text:0.14
    """
    # parameters
    nb_tokens_single_question= 15.3
    nb_token_answers_per_token_text = 0.15 # tokens_answer / tokens_text
    nb_tokens_padding = nb_tokens_single_question / 2
    lower_bound_token_limit = model_token_limit - 16 # we avoid asking for the token limit
    nb_tokens_answer = lower_bound_token_limit - nb_token_messages
    # do we have enough tokens left to answer?
    can_return_single_question = nb_tokens_answer > (nb_tokens_single_question + nb_tokens_padding)
    can_return_all_questions = nb_tokens_answer > ((nb_token_answers_per_token_text * nb_tokens_text) + nb_tokens_padding)
    # TODO make sure we have enough tokens left to ask the question and get an answer
    if can_return_all_questions and can_return_single_question:
        # we have enough tokens left to provide an answer
        return nb_tokens_answer
    else:
        # we need a smaller text in order to answer
        return None

def questions_of_answer(answer):
    """
    takes a numebred list of question as a string
    returns them as a list of string
    the input might have prefixes / suffixes that are not questions or incomplete questions.
    """
    # a question is a line starting with a number followed by a dot and a space
    question_pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
    questions = question_pattern.findall(answer)
    # ignores the last questions if it does not end with punctuation
    if (len(questions) > 0) and (not re.search(r"[.!?]$", questions[-1])):
        print(f"Popping incomplete question: '{questions[-1]}'")
        questions.pop()
    return questions

def extract_questions(file_name, file_path, text):
    # messages that will be sent to the model
    text = text.strip()
    messages = messages_of_text(text)
    # make sur we have enough tokens left to get an answer from the model
    nb_tokens_text, nb_token_messages = compute_nb_tokens(text, messages)
    nb_tokens_answer = compute_nb_tokens_answer(nb_tokens_text, nb_token_messages)
    if nb_tokens_answer is None:
        # split text and call function recurcively
        print(f"Splitting '{file_path}' into smaller chunks.")
        outputs = []
        for sub_title,sub_text in split_markdown(text):
            sub_file_path = file_path + '/' + sub_title.replace('# ', '#').replace(' ', '-').lower()
            sub_outputs = extract_questions(file_name, sub_file_path, sub_text)
            outputs.extend(sub_outputs)
        return outputs
    # runs the model
    model = ChatOpenAI(temperature=0.0, max_tokens=nb_tokens_answer)
    answer = model(messages).content.strip()
    # extracts the questions
    questions = questions_of_answer(answer)
    # zip them with source information
    outputs = [(file_name, file_path, text, question.strip()) for question in questions]
    return outputs
    
#----------------------------------------------------------------------------------------
# PROCESS ALL INPUTS

# runs the model on all files
questions = []
for file_name, file_path, text in inputs:
    file_questions = extract_questions(file_name, file_path, text)
    questions.extend(file_questions)

# displays our outputs
for i,(file_name, file_path, text, question) in enumerate(questions):
    print(f"[{i}] {file_path}: {question}")

#----------------------------------------------------------------------------------------
# DATA GATHERING

encoding = tiktoken.encoding_for_model(model_type)

nb_questions = len(questions)
texts = set()
total_question_size = 0
for i, (file_name, file_path, text, question) in enumerate(questions):
    question = f"{i}. {question}"
    question_size = len(encoding.encode(question))
    total_question_size += question_size
    texts.add(text)

nb_texts = len(texts)
total_texts_size = 0
for text in texts:
    text_size = len(encoding.encode(text))
    total_texts_size += text_size

print(f"nb_questions:{nb_questions} total_question_size:{total_question_size} nb_texts:{nb_texts} total_texts_size:{total_texts_size}")
print(f"average question size:{total_question_size / nb_questions}")
print(f"ratio question/text:{total_question_size/total_texts_size}")