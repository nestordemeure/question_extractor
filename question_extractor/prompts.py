from langchain.schema import HumanMessage, SystemMessage

#----------------------------------------------------------------------------------------
# EXTRACTION

# prompt used to extract questions
extraction_system_prompt="You are an expert user extracting information to quizz people on documentation. You will be passed a page extracted from the documentation, write a numbered list of questions that can be answered based *solely* on the given text."

def generate_extraction_messages(text):
    """
    Takes a piece of text and returns a list of messages designed to extracts questions from the text.
    """
    system_message = SystemMessage(content="You are an expert user extracting information to quizz people on documentation. You will be passed a page extracted from the documentation, write a numbered list of questions that can be answered based *solely* on the given text.")
    human_message = HumanMessage(content=text)
    return [system_message, human_message]

#----------------------------------------------------------------------------------------
# ANSWERING

# prompt used to answer a question
answering_system_prompt="You are an expert user answering questions. You will be passed a page extracted from a documentation and a question. Generate a comprehensive and informative answer to the question based *solely* on the given text."

def generate_answering_messages(question, text):
    """
    Takes a piece of text and returns a list of messages designed to extracts questions from the text.
    """
    system_prompt="You are an expert user answering questions. You will be passed a page extracted from a documentation and a question. Generate a comprehensive and informative answer to the question based *solely* on the given text."
    system_message = SystemMessage(content=system_prompt)
    human_message_text = HumanMessage(content=text)
    human_message_question = HumanMessage(content=question)
    return [system_message, human_message_text, human_message_question]
