# Spam classifier using TfIdf Vectorizer and Naive Bayes

## Jupyter Notebook file
the Notebook file contains the code for building the model.

Different Naive Bayes models have been compared namely:
- GaussianNB
- BernoulliNB
- ComplementNB
- MultinomialNB

Based on the precision, and f1_score, the BernoulliNB was selected as the best model

The vectorizer and the model are stored as binary files

## CLI
the cli.py file asks for the input text and classifies the output as "spam" or "ham"(not spam)

It uses the vectorizer and the model from the binary files

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 cli.py
```