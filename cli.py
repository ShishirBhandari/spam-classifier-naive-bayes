import pickle
from utils.processing import clean_text

def load_bin(filename: str):
    return pickle.load(open(filename, "rb"))


def classify(model, vectorizer, input_text):
    cleaned_input = clean_text(input_text)
    cleaned_input_vec = vectorizer.transform([cleaned_input])

    prediction = model.predict(cleaned_input_vec)
    return prediction[0]


def main():
    spam_classifier_model = load_bin("spam_classification_model.pkl")
    vectorizer = load_bin("vectorizer.pkl")

    while True:
        print("-" * 50)
        print("Enter a text to classify (enter 'q' to quit):")
        input_text = input("input: ")
        if input_text.lower() == 'q':
            break
        res = classify(spam_classifier_model, vectorizer, input_text)
        print("output: ", res)
        print("_" * 50)


if __name__ == "__main__":
    main()
