import re
import nltk
from collections import Counter, defaultdict

class SmartNextWordPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = defaultdict(Counter)
        self.load_data()
        self.build_model()

    def load_data(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        self.words = text.split()

        print("Total words:", len(self.words))
        print("Vocabulary:", len(set(self.words)))

    def build_model(self):
        for a, b in zip(self.words, self.words[1:]):
            self.model[a][b] += 1
        print("Model created.")

    def predict(self, sentence, top_k=5):
        words = re.sub(r"[^a-zA-Z\s]", "", sentence.lower()).split()
        if not words:
            return []

        last_word = words[-1]
        predictions = self.model.get(last_word)

        if not predictions:
            return []

        total = sum(predictions.values())

        return [
            (word, count / total)
            for word, count in predictions.most_common(top_k)
        ]

    def display(self, sentence):
        results = self.predict(sentence)

        print("\nInput:", sentence)

        if not results:
            print("No prediction found.")
            return

        print("Suggestions:")
        for i, (word, probability) in enumerate(results, 1):
            print(f"{i}. {word} ({probability:.2%})")


def main():
    predictor = SmartNextWordPredictor("smart-word.txt")

    print("\nSMART NEXT-WORD PREDICTOR")
    print("Type 'exit' to stop.")

    while True:
        sentence = input("\nEnter text: ").strip()

        if sentence.lower() == "exit":
            break

        predictor.display(sentence)


if __name__ == "__main__":
    main()
