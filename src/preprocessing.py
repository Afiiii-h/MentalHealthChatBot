import string

def preprocess_text(s: str) -> str:
    """
    Clean and normalize text: lowercase, remove punctuation, trim spaces.
    """
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def augment_texts(texts, labels):
    """
    Simple data augmentation: expands 'i am' -> 'i'm' etc.
    Helps the model learn variations.
    """
    extra_texts, extra_labels = [], []
    for t, l in zip(texts, labels):
        if "i am " in t:
            extra_texts.append(t.replace("i am ", "i'm "))
            extra_labels.append(l)
        if "i feel " in t:
            extra_texts.append(t.replace("i feel ", "i'm feeling "))
            extra_labels.append(l)
    return texts + extra_texts, labels + extra_labels