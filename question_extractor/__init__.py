import re
import time
import asyncio
import openai
from langchain.chat_models import ChatOpenAI
from .markdown import load_markdown_files_from_directory, split_markdown
from .token_counting import count_tokens_text, count_tokens_messages, get_available_tokens, are_tokens_available_for_both_conversations
from .prompts import create_answering_conversation_messages, create_extraction_conversation_messages

#---------------------------------------------------------------------------------------------
# QUESTION PROCESSING

def flatten_nested_lists(nested_lists):
    """
    Takes a list of lists as input and returns a flattened list containing all elements.
    
    Args:
        nested_lists (list of lists): A list containing one or more sublists.

    Returns:
        list: A flattened list containing all elements from the input nested lists.
    """
    flattened_list = []

    # Iterate through the nested lists and add each element to the flattened_list
    for sublist in nested_lists:
        flattened_list.extend(sublist)

    return flattened_list


async def run_model(messages, max_retry=6):
    """
    Asynchronously runs the chat model with as many tokens as possible on the given messages.
    
    Args:
        messages (list): A list of input messages to be processed by the model.

    Returns:
        str: The model-generated output text after processing the input messages.
    """
    # Count the number of tokens in the input messages
    num_tokens_in_messages = count_tokens_messages(messages)

    # Calculate the number of tokens available for processing
    num_tokens_available = get_available_tokens(num_tokens_in_messages)

    # Create an instance of the ChatOpenAI model with minimum imagination (temperature set to 0)
    model = ChatOpenAI(temperature=0.0, max_tokens=num_tokens_available, max_retries=0)

    # Retries until we succeed
    generated_text = None
    for i in range(max_retry):
        try:
            # Asynchronously run the model on the input messages
            # by default it is set to process a list of inputs
            output = await model._agenerate(messages)
            # Extract and return the generated text from the model output
            generated_text = output.generations[0].text.strip()
            break
        except (asyncio.TimeoutError, openai.error.Timeout, openai.error.RateLimitError) as e:
            # Wait before retrying
            # the wait will purposefully impact *all* concurent tasks
            retry_delay = int(2**i)
            print(f"WARNING: Timeout, retrying in {retry_delay} seconds.")
            time.sleep(retry_delay)
    
    # Uses a dummy text in case of complete failure
    if generated_text is None:
        print(f"ERROR: Could not generate text for an input.")
        generated_text = 'ERROR'
    return generated_text


def extract_questions_from_output(output):
    """
    Takes a numbered list of questions as a string and returns them as a list of strings.
    The input might have prefixes/suffixes that are not questions or incomplete questions.

    Args:
        output (str): A string containing a numbered list of questions.

    Returns:
        list of str: A list of extracted questions as strings.
    """
    # Define a regex pattern to match questions (lines starting with a number followed by a dot and a space)
    question_pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)

    # Find all the questions matching the pattern in the input text
    questions = question_pattern.findall(output)

    # Check if the last question is incomplete (does not end with punctuation or a parenthesis)
    if (len(questions) > 0) and (not re.search(r"[.!?)]$", questions[-1].strip())):
        print(f"WARNING: Popping incomplete question: '{questions[-1]}'")
        questions.pop()

    return questions


async def extract_questions_from_text(file_path, text):
    """
    Asynchronously extracts questions from the given text.
    
    Args:
        file_path (str): The file path of the markdown file.
        text (str): The text content of the markdown file.

    Returns:
        list of tuple: A list of tuples, each containing the file path, text, and extracted question.
    """
    # Ensure the text can be processed by the model
    text = text.strip()
    num_tokens_text = count_tokens_text(text)

    if not are_tokens_available_for_both_conversations(num_tokens_text):
        # Split text and call function recursively
        print(f"WARNING: Splitting '{file_path}' into smaller chunks.")

        # Build tasks for each subsection of the text
        tasks = []
        for sub_title, sub_text in split_markdown(text):
            sub_file_path = file_path + '/' + sub_title.replace('# ', '#').replace(' ', '-').lower()
            task = extract_questions_from_text(sub_file_path, sub_text)
            tasks.append(task)

        # Asynchronously run tasks and gather outputs
        tasks_outputs = await asyncio.gather(*tasks)

        # Flatten and return the results
        return flatten_nested_lists(tasks_outputs)
    else:
        # Run the model to extract questions
        messages = create_extraction_conversation_messages(text)
        output = await run_model(messages)
        questions = extract_questions_from_output(output)

        # Associate questions with source information and return as a list of tuples
        outputs = [(file_path, text, question.strip()) for question in questions]
        return outputs


async def generate_answer(question, source):
    """
    Asynchronously generates an answer to a given question using the provided source text.
    
    Args:
        question (str): The question to be answered.
        source (str): The text containing relevant information for answering the question.

    Returns:
        str: The generated answer to the question.
    """
    # Create the input messages for the chat model
    messages = create_answering_conversation_messages(question, source)

    # Asynchronously run the chat model with the input messages
    answer = await run_model(messages)

    return answer

#---------------------------------------------------------------------------------------------
# FILE PROCESSING

async def process_file(file_path, text, progress_counter, verbose=True):
    """
    Asynchronously processes a file, extracting questions and generating answers concurrently.
    
    Args:
        file_path (str): The file path of the markdown file.
        text (str): The text content of the markdown file.
        progress_counter (dict): A dictionary containing progress information ('nb_files_done' and 'nb_files').
        verbose (bool): If True, print progress information. Default is True.

    Returns:
        list: A list of dictionaries containing source, question, and answer information.
    """
    # Extract questions from the text
    questions = await extract_questions_from_text(file_path, text)

    # Build and run answering tasks concurrently
    tasks = []
    for sub_file_path, sub_text, question in questions:
        task = generate_answer(question, sub_text)
        tasks.append(task)

    tasks_outputs = await asyncio.gather(*tasks)

    # Merge results into a list of dictionaries
    result = []
    for (sub_file_path, sub_text, question), answer in zip(questions, tasks_outputs):
        result.append({'source': sub_file_path, 'question': question, 'answer': answer})

    # Update progress and display information if verbose is True
    progress_counter['nb_files_done'] += 1  # No race condition as we are single-threaded
    if verbose:
        print(f"{progress_counter['nb_files_done']}/{progress_counter['nb_files']}: File '{file_path}' done!")

    return result


async def process_files(files, verbose=True):
    """
    Asynchronously processes a list of files, extracting questions and generating answers concurrently.
    
    Args:
        files (list): A list of tuples containing file paths and their respective text content.
        verbose (bool): If True, print progress information. Default is True.

    Returns:
        list: A merged list of dictionaries containing source, question, and answer information.
    """
    # Set up progress information for display
    nb_files = len(files)
    progress_counter = {'nb_files': nb_files, 'nb_files_done': 0}
    if verbose: print(f"Starting question extraction on {nb_files} files.")

    # Build and run tasks for each file concurrently
    tasks = []
    for file_path, text in files:
        task = process_file(file_path, text, progress_counter, verbose=verbose)
        tasks.append(task)

    tasks_outputs = await asyncio.gather(*tasks)

    # Merge results from all tasks
    return flatten_nested_lists(tasks_outputs)

#---------------------------------------------------------------------------------------------
# MAIN

def extract_questions_from_directory(input_folder, verbose=True):
    """
    Extracts questions and answers from all markdown files in the input folder.

    Args:
        input_folder (str): A path to a folder containing markdown files.
        verbose (bool): If True, print progress information. Default is True.

    Returns:
        list: A list of dictionaries containing path, source, question, and answer information.
    """
    # Load input files from the folder
    if verbose: print(f"Loading files from '{input_folder}'.")
    files = load_markdown_files_from_directory(input_folder)

    # Run question extraction tasks
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_files(files, verbose=verbose))

    if verbose: print(f"Done, {len(results)} question/answer pairs have been generated!")
    return results
