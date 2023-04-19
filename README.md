# Question Extractor

Large language models can be instruction tuned with a set of questions and answers.
However, to further fine-tune a model *on your own data*, you need a large number of questions and answers which can be a lot of manual work.

This repository lets you use a non-fine-tuned language model and existing textual data to extract question/answer pairs automatically, eliminating all manual work.

## Installation

TODO

## Usage

TODO

## Inner-workings

TODO

# Potential improvement

- parallelize the code with async?
- write basic documentation in readme
- update function documentation
- run on [full NERSC doc](https://gitlab.com/NERSC/nersc.gitlab.io/-/tree/main/docs)
- use straight OpenAI API instead of langchain to simplify installation