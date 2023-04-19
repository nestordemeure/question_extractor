import re
import asyncio
from langchain.chat_models import ChatOpenAI
from .markdown import import_markdown_files, split_markdown
from .token_counting import count_tokens_text, count_tokens_messages, count_tokens_left, has_tokens_left
from .prompts import generate_answering_messages, generate_extraction_messages

#---------------------------------------------------------------------------------------------
# QUESTION PROCESSING

def flatten_lists(nested_lists):
    """
    Flattens a list of lists
    """
    result = []
    for l in nested_lists:
        result.extend(l)
    return result

async def run_model(messages):
    """
    Runs the model, with as many token as possible, on the given messages
    """
    # we use as many tokens as possible
    nb_tokens_messages = count_tokens_messages(messages)
    nb_tokens_available = count_tokens_left(nb_tokens_messages)
    # temperature set to zero to minimise imagination
    model = ChatOpenAI(temperature=0.0, max_tokens=nb_tokens_available)
    # runs the model asyncrhonously
    # by default this is set to process a list of inputs
    output = await model.agenerate([messages])
    return output.generations[0][0].text.strip()

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
        print(f"WARNING: Popping incomplete question: '{questions[-1]}'")
        questions.pop()
    return questions

async def extract_questions(file_path, text):
    # insures that we can run the model on the text
    text = text.strip()
    nb_tokens_text = count_tokens_text(text)
    if not has_tokens_left(nb_tokens_text):
        # split text and call function recurcively
        print(f"WARNING: Splitting '{file_path}' into smaller chunks.")
        # builds all tasks and runs them
        tasks = []
        for sub_title,sub_text in split_markdown(text):
            sub_file_path = file_path + '/' + sub_title.replace('# ', '#').replace(' ', '-').lower()
            task = extract_questions(sub_file_path, sub_text)
            tasks.append(task)
        tasks_outputs = await asyncio.gather(*tasks)
        # flattens results
        return flatten_lists(tasks_outputs)
    else:
        # run model
        messages = generate_extraction_messages(text)
        output = await run_model(messages)
        questions = questions_of_output(output)
        # zip questions with source information
        outputs = [(file_path, text, question.strip()) for question in questions]
        return outputs

async def answer_question(question, source):
    """
    Answers a question given a text containing the relevant information.
    """
    messages = generate_answering_messages(question, source)
    answer = await run_model(messages)
    return answer

#---------------------------------------------------------------------------------------------
# FILE PROCESSING

async def process_file(file_path, text, progress_counter, verbose=True):
    """
    Extracts all questions from a file.
    Runs question answering on all questions concurently.
    Returns merged results.
    """
    # extracts the questions
    questions = await extract_questions(file_path, text)
    # build all answering tasks and runs them
    tasks = []
    for sub_file_path, sub_text, question in questions:
        task = answer_question(question, sub_text)
        tasks.append(task)
    tasks_outputs = await asyncio.gather(*tasks)
    # merge results
    result = []
    for (sub_file_path, sub_text, question), answer in zip(questions, tasks_outputs):
        result.append({'source':sub_file_path, 'question':question, 'answer':answer})
        #if verbose: print(f"Q: {question}\n\nA: {answer}\n")
    # display
    progress_counter['nb_files_done'] += 1 # no race condition as we are single threaded
    if verbose: 
        print(f"{progress_counter['nb_files_done']}/{progress_counter['nb_files']}: File '{file_path}' done!")
    return result

async def process_files(files, verbose=True):
    """
    Runs question extraction on all files concurently.
    Returns merged results
    """
    # used to display progress information
    nb_files = len(files)
    if verbose: print(f"Starting question extraction on {nb_files} files.")
    progress_counter = {'nb_files':nb_files, 'nb_files_done': 0}
    # build all file tasks and runs them
    tasks = []
    for file_path, text in files:
        task = process_file(file_path, text, progress_counter, verbose=verbose)
        tasks.append(task)
    tasks_outputs = await asyncio.gather(*tasks)
    # merges results
    return flatten_lists(tasks_outputs)

#---------------------------------------------------------------------------------------------
# MAIN

def question_extractor(input_folder, verbose=True):
    """
    Takes a path to a folder as input.
    Returns a list of {path, source, question, answer}
    """
    # load inputs
    if verbose: print(f"Loading files from '{input_folder}'.")
    files = import_markdown_files(input_folder)
    # runs question extraction tasks
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(process_files(files, verbose=verbose))
    if verbose: print(f"Done, {len(result)} question/answer pairs have been generated!")
    return result