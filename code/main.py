import spacy
from spacy import displacy

from collections import defaultdict
import matplotlib.pyplot as plt

import time
import sys
from pathlib import Path

spacy_nlp = spacy.load('en_core_web_sm')
sys.path.insert(1, 'nltk/') #to import the local version of NLTK

import nltk
from nltk.parse.transitionparser import TransitionParser, DependencyEvaluator
from nltk.corpus import dependency_treebank

nltk.download('dependency_treebank')


def plotDepGraph(spacy_doc):
  """
  function to export the dependency graph into SVG image
  """
  svg = displacy.render(spacy_doc, style="dep")
  output_path = Path("dep_graph.svg")
  output_path.open("w", encoding="utf-8").write(svg)


def es1(sentence:str, debug=False):
  """
  extract a path of dependency relations from the ROOT to a token
    - input is a sentence, you parse it and get a Doc object of spaCy
    - for each token the path will be a list of dependency relations, where first element is ROOT
  """
  spacy_doc = spacy_nlp(sentence)
  if debug: #plot the dependecy graph to inspect friendly the result
    plotDepGraph(spacy_doc)

  ret = []
  for sent in spacy_doc.sents:
      for token in sent:
        ancestors = " ".join([t.dep_ for t in token.ancestors][::-1])
        ret.append([t.dep_ for t in token.ancestors][::-1])
        ret[-1].extend([token.dep_])
        if debug: print("{}\t{:15s}\t{}".format(token.i, token.text, ancestors))
  return ret

def es2(sentence:str, debug=False):
  """
  extract subtree of a dependents given a token
    - input is a sentence, you parse it and get a Doc object of spaCy
    - for each token in Doc objects you extract a subtree of its dependents as a list (ordered w.r.t. sentence order)
  """
  spacy_doc = spacy_nlp(sentence)
  if debug: #plot the dependecy graph to inspect friendly the result
    plotDepGraph(spacy_doc)

  ret = []
  for sent in spacy_doc.sents:
      for token in sent:
        desc = " ".join([t.text for t in token.subtree])
        ret.append([t.text for t in token.subtree])
        if debug: print("{}\t{:15s}\t{}".format(token.i, token.text, desc))
  return ret

def es3(sentence:str, subtree:list, debug=False):
  """
  check if a given list of tokens (segment of a sentence) forms a subtree
    - you parse a sentence and get a Doc object of spaCy
    - providing as an input ordered list of words from a sentence, you output True/False based on the sequence forming a subtree or not
  """
  assert " ".join(subtree) in sentence
  spacy_doc = spacy_nlp(sentence)
  if debug: #plot the dependecy graph to inspect friendly the result
    plotDepGraph(spacy_doc)

  for sent in spacy_doc.sents:
      for token in sent:
        desc = [t.text for t in token.subtree] #or one might reuse the function defined by es2()
        if desc == subtree:
          return True
  return False
   

def es4(words:str, debug=False):
  """
  identify head of a span, given its tokens
    - input is a sequence of words (not necessarily a sentence)
    - output is the head of the span (single word)
  """
  spacy_doc = spacy_nlp(words)
  if debug: #plot the dependecy graph to inspect friendly the result
    plotDepGraph(spacy_doc)
  return list(spacy_doc.sents)[0].root


def es5(words:str, debug=False):
  """
  extract sentence subject, direct object and indirect object spans
    - input is a sentence, you parse it and get a Doc object of spaCy
    - output is lists of words that form a span (not a single word) for subject, direct object, and indirect object (if present of course, otherwise empty)
      - dict of lists, is better
  in particular, I extract tokens with the mentioned dependecy relations, but only the one forming a subtree (TODO ?????)
  """
  spacy_doc = spacy_nlp(words)
  if debug: #plot the dependecy graph to inspect friendly the result
    plotDepGraph(spacy_doc)

  ret = {"nsubj":[], "dobj":[], "iobj":[]}
  for sent in spacy_doc.sents:
      for token in sent:
        if token.dep_ in ["nsubj", "dobj", "iobj"]:
          #if [t.text for t in token.subtree] != [token.text]: #if it forms a span
          ret[token.dep_].append(" ".join([t.text for t in token.subtree]))
  return ret



def extra_point():
  tp = TransitionParser('arc-standard', use_glove=False, linear_svm=False)
  tp.train(dependency_treebank.parsed_sents()[:100], 'tp.model')
  parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')
  de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])
  las, uas = de.eval()
  print("\nLAS={} USA={} without GLOVE\n".format(round(las,2), round(uas,2)))

  tp = TransitionParser('arc-standard', use_glove=True, linear_svm=False)
  tp.train(dependency_treebank.parsed_sents()[:100], 'tp.model')
  parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')
  de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])
  las, uas = de.eval()
  print("\nLAS={} USA={} with GLOVE\n".format(round(las,2), round(uas,2)))


  results = []
  times = []
  train_data_to_test = [100, 300]# 100, 300, 500, 1000
  for i , linear in enumerate([True, False]): #enable/disable LinearSVC
    results.append([])
    times.append([])
    for train_data in train_data_to_test:    
      start = time.time()
      tp = TransitionParser('arc-standard', use_glove=False, linear_svm=linear)
      tp.train(dependency_treebank.parsed_sents()[:train_data], 'tp.model')

      times[i].append(time.time()-start)

      parses = tp.parse(dependency_treebank.parsed_sents()[-20:], 'tp.model')
      de = DependencyEvaluator(parses, dependency_treebank.parsed_sents()[-20:])
      las, _ = de.eval()
      results[i].append(las)
      
  ax1 = plt.subplot(211)
  ax1.plot(train_data_to_test, results[0], label="LinearSVC")
  ax1.plot(train_data_to_test, results[1], label="SVC")
  ax1.set(xlabel='n° training samples', ylabel='LAS', title='LAS scores')
  plt.legend()

  ax2 = plt.subplot(212, sharex=ax1)
  ax2.plot(train_data_to_test, times[0], label="LinearSVC")
  ax2.plot(train_data_to_test, times[1], label="SVC")
  ax2.set(xlabel='n° training samples', ylabel='Training time\n (s)', title='Training times')

  plt.legend()
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
    example = "I saw the man with a telescope."


    print(example, "\n")

    print("Point 1: \n", es1(example))
    print("\nPoint 2: \n", es2(example))
    print("\nPoint 3: \n", es3(example, ["man", "with", "a", "telescope"], False))
    print("\nPoint 4: \n", es4(example))
    print("\nPoint 5: \n", es5(example, True))

    print("\nExtra point: \n", )
    extra_point()
