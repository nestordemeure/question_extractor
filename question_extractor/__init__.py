import re
import json
from langchain.chat_models import ChatOpenAI
from .markdown import import_markdown_files, split_markdown
from .token_counting import count_tokens_text, count_tokens_messages, count_tokens_left, has_tokens_left
from .prompts import generate_answering_messages, generate_extraction_messages

#---------------------------------------------------------------------------------------------
# COMPONENTS

def run_model(messages):
    """
    Runs the model, with as many token as possible, on the given messages
    """
    nb_tokens_messages = count_tokens_messages(messages)
    nb_tokens_available = count_tokens_left(nb_tokens_messages)
    model = ChatOpenAI(temperature=0.0, max_tokens=nb_tokens_available)
    output = model(messages).content.strip()
    return output

def questions_of_output(output):
    """
    takes a numebred list of question as a string
    returns them as a list of string
    the input might have prefixes / suffixes that are not questions or incomplete questions.
    """
    # a question is a line starting with a number followed by a dot and a space
    question_pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
    questions = question_pattern.findall(output)
    # ignores the last questions if it does not end with punctuation
    if (len(questions) > 0) and (not re.search(r"[.!?]$", questions[-1])):
        print(f"Popping incomplete question: '{questions[-1]}'")
        questions.pop()
    return questions

def extract_questions(file_path, text):
    # insures that we can run the model on the text
    text = text.strip()
    nb_tokens_text = count_tokens_text(text)
    if not has_tokens_left(nb_tokens_text):
        # split text and call function recurcively
        print(f"Splitting '{file_path}' into smaller chunks.")
        outputs = []
        for sub_title,sub_text in split_markdown(text):
            sub_file_path = file_path + '/' + sub_title.replace('# ', '#').replace(' ', '-').lower()
            sub_outputs = extract_questions(sub_file_path, sub_text)
            outputs.extend(sub_outputs)
        return outputs
    else:
        # run model
        messages = generate_extraction_messages(text)
        output = run_model(messages)
        questions = questions_of_output(output)
        # zip questions with source information
        outputs = [(file_path, text, question.strip()) for question in questions]
        return outputs

def answer_question(question, text):
    """
    Answers a question given a text containing the relevant information.
    """
    messages = generate_answering_messages(question, text)
    answer = run_model(messages)
    return answer

#---------------------------------------------------------------------------------------------
# MAIN

def question_extractor(input_folder, verbose=True):
    """
    Takes a path to a folder as input.
    Returns a list of {path, source, question, answer}
    """
    # load inputs
    files = import_markdown_files(input_folder)
    # runs the model on all files
    result = []
    for file_path, text in files:
        questions = extract_questions(file_path, text)
        for sub_file_path, sub_text, question in questions:
            answer = answer_question(question, sub_text)
            result.append({'path':sub_file_path, 'source':sub_text, 'question':question, 'answer':answer})
            if verbose: print(f"Q: {question}\nA: {answer}\n")
    return result
