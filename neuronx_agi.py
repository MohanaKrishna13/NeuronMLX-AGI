import numpy as np
import tensorflow as tf
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ====== GPU/CPU Strategy ======
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f"✅ Strategy initialized: {gpus}")
except:
    device = 'cpu'
    print("⚠️ Default strategy applied.")

print(f"Device set to use {device}")

# ====== Perception Module ======
class Perception:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def summarize(self, text):
        if any(char.isdigit() for char in text) or len(text.split()) < 5:
            return text
        return self.summarizer(text, max_length=50, min_length=5, do_sample=False)[0]['summary_text']

# ====== Reasoning ======
class Reasoning:
    def plan(self, text):
        return f"Plan → Analyze input: '{text}' and apply reasoning"

# ====== Trainer ======
class Trainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.trained = False

    def train(self, texts, labels):
        vectors = self.vectorizer.fit_transform(texts)
        self.model.fit(vectors, labels)
        self.trained = True

    def predict(self, text):
        if not self.trained:
            return "❌ Classifier not trained."
        vector = self.vectorizer.transform([text])
        return self.model.predict(vector)[0]

# ====== NeuronX AGI System ======
class NeuronXAGI:
    def __init__(self, num_cells=10000):
        self.num_cells = num_cells
        self.perception = Perception()
        self.reasoning = Reasoning()
        self.trainer = Trainer()

    def train_brain(self, texts, labels):
        self.trainer.train(texts, labels)
        print("✅ AGI Brain Trained with Dataset")

    def think(self, input_text):
        print("\n==========================")
        print(f"🧪 Input: {input_text}")

        summary = self.perception.summarize(input_text)
        plan = self.reasoning.plan(summary)
        prediction = self.trainer.predict(summary)

        print(f"📘 Summary: {summary}")
        print(f"🧭 Reasoning: {plan}")
        print(f"🎯 Prediction: {prediction}")

        # Simulate Neuron Firing
        inputs = tf.random.normal([self.num_cells, 100])
        weights = tf.random.normal([self.num_cells, 100])
        bias = tf.random.normal([self.num_cells, 1])
        dot = tf.reduce_sum(inputs * weights, axis=1, keepdims=True) + bias
        outputs = tf.math.tanh(dot)

        print(f"🧠 {self.num_cells:,} neurons fired! Sample: {outputs[:5].numpy().squeeze().tolist()}")

# ====== Training Data ======
train_texts = [
    "force causes motion", "glucose releases energy",
    "Newton's third law", "photosynthesis in chloroplasts",
    "What is 123 × 456?", "Area of circle with radius 7",
    "Solve 56 + 44", "Square root of 144", "Calculate 999 × 88",
    "Explain Newton's law of motion"
]

train_labels = [
    "Physics", "Biology",
    "Physics", "Biology",
    "Math", "Math", "Math", "Math", "Math",
    "Physics"
]

# ====== Run AGI Prototype ======
agi = NeuronXAGI(num_cells=10000)
agi.train_brain(train_texts, train_labels)

# ====== Test Inputs ======
test_inputs = [
    "What is the powerhouse of the cell?",
    "Calculate the area of a circle with radius 7.",
    "Explain Newton's Third Law of Motion.",
    "Who discovered gravity?",
    "What is 123 × 456?"
]

for text in test_inputs:
    agi.think(text)
