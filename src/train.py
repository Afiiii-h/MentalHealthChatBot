from preprocessing import preprocess_text, augment_texts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Step 1: Create dataset
texts = ["I am happy", "I feel anxious", "I am sad", "I feel great"]
labels = ["happy", "anxiety", "sad", "happy"]

# Step 2: Preprocess and augment
texts = [preprocess_text(t) for t in texts]
texts, labels = augment_texts(texts, labels)

# Step 3: Build pipeline (vectorizer + model)
model = make_pipeline(CountVectorizer(), LogisticRegression())

# Step 4: Train model
model.fit(texts, labels)

# Step 5: Test model
test_input = "I am feeling very anxious today"
print("Prediction:", model.predict([preprocess_text(test_input)])[0])