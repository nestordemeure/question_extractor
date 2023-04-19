import tiktoken
from .prompts import answering_system_prompt, extraction_system_prompt, generate_answering_messages, generate_extraction_messages

#----------------------------------------------------------------------------------------
# COUNTING

# properties of the model used
model_type='gpt-3.5-turbo'
model_token_limit=4096
model_tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
model_tokens_per_name = -1  # if there's a name, the role is omitted

# parameters
nb_padding_tokens = 16 # tokens added to make sure we do not go over the limit

# encoder used to turn text into token
encoding = tiktoken.encoding_for_model(model_type)

def count_tokens_text(text):
    """Returns the number of tokens used to encode a given text."""
    return len(encoding.encode(text))

def count_tokens_messages(messages):
    """
    Counts the number of tokens needed to encode the list of messages.
    Adapted from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    nb_token_messages = 0
    for message in messages:
        nb_token_messages += model_tokens_per_message 
        nb_token_messages += count_tokens_text(message.content)
        nb_token_messages += model_tokens_per_name
    nb_token_messages += 3  # every reply is primed with <|start|>assistant<|message|>
    return nb_token_messages

def count_tokens_left(nb_token_messages):
    """
    Takes information on the size of the messages.
    Returns the number of tokens that can be requested from the model.
    """
    lower_bound_token_limit = model_token_limit - nb_padding_tokens # we avoid asking for the token limit
    nb_tokens_left = lower_bound_token_limit - nb_token_messages
    return nb_tokens_left

#----------------------------------------------------------------------------------------
# PREDICTING

# data gathered running tests
average_question_size = 15.28
average_answer_size = 94.78
average_ratio_questions_text = 0.14 # sum(question)/sum(text)

# size of empty messages
nb_tokens_empty_extraction_messages = count_tokens_messages(generate_extraction_messages(text=''))
nb_tokens_empty_answering_messages = count_tokens_messages(generate_answering_messages(question='', text=''))

def predict_tokens_extraction(nb_tokens_text):
    """predicts the full size of the extraction conversation"""
    # at least one question
    # or a number of question proportional to the text length
    # plus half an average question worth of padding
    upperbound_output_size = max(average_question_size, nb_tokens_text * average_ratio_questions_text) + average_question_size/2
    # size of the prompt, text and output
    return nb_tokens_empty_extraction_messages + nb_tokens_text + upperbound_output_size

def predict_tokens_answering(nb_tokens_text):
    """predicts the full size of the answering conversation"""
    # one question plus half a question worth of padding
    upperbound_question_size = average_question_size * 1.5
    # one answer plus half an answer worth of padding
    upperbound_answer_size = average_answer_size * 1.5
    # size of the prompt, text, question and answer
    return nb_tokens_empty_answering_messages + nb_tokens_text + upperbound_question_size + upperbound_answer_size

#----------------------------------------------------------------------------------------
# CHECKING

def has_tokens_left(nb_tokens_text):
    """
    Returns true if there is enough tokens left to get an answer from the model for both extraction and answering
    """
    # extraction
    tokens_extraction = predict_tokens_extraction(nb_tokens_text)
    tokens_left_extraction = count_tokens_left(tokens_extraction)
    # answering
    tokens_anwering = predict_tokens_answering(nb_tokens_text)
    tokens_left_answering = count_tokens_left(tokens_anwering)
    # check 
    return (tokens_left_extraction > 0) and (tokens_left_answering > 0)
