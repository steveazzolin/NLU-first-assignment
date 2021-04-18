# NLU-first-assignement
First assignment of the course Natural Language Understanding @ UNITN.

## Requirements
The repository was structured in a way to make it easy to run the code directly on Google Colab, so as to minimize the configuration on your local machine, so please refer to it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/steveazzolin/NLU-first-assignment/blob/main/code/NLU_first_assignment.ipynb)

However, in case is needed, I list here the explicit requirements:
- SpaCy 2.2.4
- Matplotlib
- Scikit-Learn
- Numpy
- Scipy
- My [fork](https://github.com/steveazzolin/nltk) of NLTK (for the extra point)
    - as a folder in the root
- [Pretrained](https://nlp.stanford.edu/projects/glove/) GloVe embeddings (for the extra point)
    - put *glove.6B.50d.txt* in *../data/glove.6B.50d.txt*


## Repo structure
- *data/*: empty directory since the data needed is downloaded directly by the Colab notebook
- *code/*: reference to the Colab notebook
- *report.pdf*: a pdf file with details about the implemented functions