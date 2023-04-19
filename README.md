# Question Extractor

Large language models can be instruction tuned with a set of questions and answers.
However, to further fine-tune a model *on your own data*, you need a large number of questions and answers about your data.
Producing those questions and answers can be a lot of manual work.

This repository lets you use a non-fine-tuned language model and existing textual data to extract question/answer pairs automatically, eliminating all manual work.

## Installation

TODO

## Usage

TODO

## Inner-workings

The code loops on all files, for each file it extracts a list of questions using the following prompt followed by a chunk of text:

```
You are an expert user extracting information to quizz people on documentation. You will be passed a page extracted from the documentation, write a numbered list of questions that can be answered based *solely* on the given text.
```

It then loops on the questions, producing an answer by passing the following prompt followed by a chunk of text and a question:

```
You are an expert user answering questions. You will be passed a page extracted from a documentation and a question. Generate a comprehensive and informative answer to the question based *solely* on the given text.
```

Most of the actual logic of the code is dedicated to processing the files concurently (for speed) and insuring that text chunk passed to the model are small enough to leave enough tokens for answering.

If a text is too long to be sent to the model, it is split allong its highest markdown heading level (the process can be repeated recurcively if needed until we get down to single paragraphs).

# Potential improvements

- use straight OpenAI API instead of langchain to reduce dependencies