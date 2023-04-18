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

def read_files(directory, verbose=False):
    """
    Take a folder path.
    Returns a list of (file_name, file_path, text) for all files in the input folder.
    """
    files_data = []
    
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            
            files_data.append((file_name, file_path, file_content))
            if verbose: print(f"Loaded '{file_path}'")

    return files_data

print("Loading inputs.")
inputs = read_files(input_folder, verbose=True)
print(f"Loaded {len(inputs)} inputs.")

#----------------------------------------------------------------------------------------
# QUESTION EXTRACTOR

"""
compute size of sum of all messages
if ok, then number of token left
if not okay then slice and retry

if tokens left are below what is needed to answer then we bissect

split policy:
if a document is too large to fit, we split at the highest heading level available
1,2,etc
if there is no heading inside, then we give up on it and report

to decie if we have enough tokens left
we get about x token of questions per token of input (page)
also lower bar should be number of token per question
also take into account the number of token per answer
"""

def messages_of_text(text):
    """
    Takes a piece of text and returns a list of messages.
    Extracts questions from the text.
    """
    system_message = SystemMessage(content="You are an expert user extracting information to quizz people on documentation. You will be passed a page extracted from the documentation, write a numbered list of questions that can be answered based *solely* on the given text.")
    human_message = HumanMessage(content=text)
    return [system_message, human_message]

def compute_nb_tokens(text, messages, model_type='gpt-3.5-turbo'):
    """
    Counts the number of tokens needed to encode the list of messages.
    Adapted from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    # settings of the tokeniser for the given model
    encoding = tiktoken.encoding_for_model(model_type)
    tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = -1  # if there's a name, the role is omitted
    # count number of tokens for the text
    nb_tokens_text = len(encoding.encode(text))
    # count number of tokens for the messages
    nb_token_messages = 0
    for message in messages:
        nb_token_messages += tokens_per_message 
        nb_token_messages += len(encoding.encode(message.content))
        nb_token_messages += tokens_per_name
    nb_token_messages += 3  # every reply is primed with <|start|>assistant<|message|>
    return nb_tokens_text, nb_token_messages

def compute_nb_tokens_answer(nb_tokens_text, nb_token_messages, model_token_limit=4096):
    # parameters
    nb_tokens_single_answer = 256 # TODO
    nb_token_answers_per_token_text = 0 # TODO tokens_answer / tokens_text
    # are we within parameters
    nb_tokens_answer = ((model_token_limit - nb_token_messages) * 9) // 10
    can_return_single_answer = nb_tokens_answer > nb_tokens_single_answer
    can_return_full_answer = nb_tokens_answer > (nb_token_answers_per_token_text * nb_tokens_text)
    # TODO make sure we have enough tokens to write an answer
    if can_return_full_answer and can_return_single_answer:
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
        print(f"Popping: '{questions[-1]}'")
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
        # TODO split text and call function recurcively
        print(f"Skipping '{file_path}'")
        return []
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