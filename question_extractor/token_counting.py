import tiktoken
from .prompts import create_answering_conversation_messages, create_extraction_conversation_messages

#----------------------------------------------------------------------------------------
# COUNTING

# properties of the model used
model_type='gpt-3.5-turbo'
model_token_limit=4096
model_tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
model_tokens_per_name = -1  # if there's a name, the role is omitted

# parameters
padding_token_count = 16 # tokens added to make sure we do not go over the limit

# encoder used to turn text into token
encoding = tiktoken.encoding_for_model(model_type)


def count_tokens_text(text):
    """
    Counts the number of tokens used to encode a given text.
    
    Args:
        text (str): The input text to be tokenized.
        
    Returns:
        int: The number of tokens in the encoded text.
    """
    encoded_text = encoding.encode(text)
    return len(encoded_text)


def count_tokens_messages(messages):
    """
    Counts the number of tokens needed to encode a list of messages.
    
    Args:
        messages (list): A list of message objects to be tokenized.
        
    Returns:
        int: The total number of tokens required to encode the messages.
        
    Adapted from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    total_tokens = 0
    for message in messages:
        total_tokens += model_tokens_per_message
        total_tokens += count_tokens_text(message.content)
        total_tokens += model_tokens_per_name

    total_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return total_tokens


def get_available_tokens(messages_token_count):
    """
    Calculates the number of tokens that can be requested from the model.
    
    Args:
        messages_token_count (int): The total number of tokens used by the messages.
        
    Returns:
        int: The number of tokens available for the model request.
    """
    adjusted_token_limit = model_token_limit - padding_token_count  # Avoid requesting the exact token limit
    available_tokens = adjusted_token_limit - messages_token_count
    return available_tokens

#----------------------------------------------------------------------------------------
# PREDICTING

# data gathered running tests
average_question_size = 15.28
average_answer_size = 94.78
average_question_text_ratio = 0.14 # sum(question)/sum(text)

# size of empty messages
empty_extraction_messages_token_count = count_tokens_messages(create_extraction_conversation_messages(text=''))
empty_answering_messages_token_count = count_tokens_messages(create_answering_conversation_messages(question='', text=''))


def estimate_extraction_conversation_tokens(text_token_count):
    """
    Estimates the total number of tokens needed for the extraction conversation.
    
    Args:
        text_token_count (int): The total number of tokens in the input text.
        
    Returns:
        float: The estimated total number of tokens needed for the extraction conversation.
    """
    # Calculate the upper bound of output tokens based on the input text tokens
    # The upper bound is the maximum of an average question size or a number of questions
    # proportional to the text length, plus half an average question worth of padding
    upper_bound_output_size = max(
        average_question_size, 
        text_token_count * average_question_text_ratio
    ) + average_question_size / 2

    # The total estimated token count includes the extraction messages,
    # input text tokens, and the calculated upper bound output size
    estimated_token_count = (
        empty_extraction_messages_token_count + 
        text_token_count + 
        upper_bound_output_size
    )
    
    return estimated_token_count


def estimate_answering_conversation_tokens(text_token_count):
    """
    Estimates the total number of tokens needed for the answering conversation.
    
    Args:
        text_token_count (int): The total number of tokens in the input text.
        
    Returns:
        float: The estimated total number of tokens needed for the answering conversation.
    """
    # Calculate the upper bound of question tokens: one question plus half a question worth of padding
    upper_bound_question_size = average_question_size * 1.5
    
    # Calculate the upper bound of answer tokens: one answer plus half an answer worth of padding
    upper_bound_answer_size = average_answer_size * 1.5
    
    # The total estimated token count includes the answering messages tokens,
    # input text tokens, and the calculated upper bound question and answer sizes
    estimated_token_count = (
        empty_answering_messages_token_count + 
        text_token_count + 
        upper_bound_question_size + 
        upper_bound_answer_size
    )
    
    return estimated_token_count

#----------------------------------------------------------------------------------------
# CHECKING

def are_tokens_available_for_both_conversations(text_token_count):
    """
    Checks if there are enough tokens available to get an answer from the model for both extraction and answering.
    
    Args:
        text_token_count (int): The total number of tokens in the input text.
    
    Returns:
        bool: True if there are enough tokens available, False otherwise.
    """
    # Calculate tokens needed for extraction conversation
    tokens_needed_extraction = estimate_extraction_conversation_tokens(text_token_count)
    tokens_available_extraction = get_available_tokens(tokens_needed_extraction)
    
    # Calculate tokens needed for answering conversation
    tokens_needed_answering = estimate_answering_conversation_tokens(text_token_count)
    tokens_available_answering = get_available_tokens(tokens_needed_answering)
    
    # Check if there are enough tokens available for both extraction and answering
    return (tokens_available_extraction > 0) and (tokens_available_answering > 0)
