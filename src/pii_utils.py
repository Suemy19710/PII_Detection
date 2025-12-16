
import pandas as pd
import numpy as np
import time
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold   
from sklearn.feature_extraction.text import TfidfVectorizer 
from gensim.models import Word2Vec 
from typing import Tuple
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# 1. Text Cleaning
# nltk.download('stopwords')
# stop = set(stopwords.wors('english'))
def clean_text(text):
    text = str(text).lower()                    # lowercase all words
    text = re.sub(r'[^a-z\s]', '', text)        # remove punctuation, numbers, special chars
    words = [w for w in text.split() if w not in stop]  #remove stopwords
    return " ".join(words) 

def clean_text_light(text):
    return str(text).lower()    

# 2. Load Dataset
def load_dataset(path):
    df = pd.read_csv(path)  
    df.columns = ['Text', 'PII_types']
    df['has_PII'] = df['PII_types'].apply(lambda x: 0 if pd.isna(x) or x.strip() == '' else 1)
    df['clean_text'] = df['Text'].apply(clean_text)
    df['clean_text_light'] = df['Text'].apply(clean_text_light)
    return df

# 3. Stratified Train-Test Split    
def stratified_split(df, text_col="Text", label_col="has_PII", test_size=0.2, random_state=42):
    X = df[text_col].astype(str)
    y = df[label_col].astype(int)

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

# 3. Text Cleaning
nltk.download('stopwords')
stop = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()                    # lowercase all words
    text = re.sub(r'[^a-z\s]', '', text)        # remove punctuation, numbers, special chars
    words = [w for w in text.split() if w not in stop]  #remove stopwords
    return " ".join(words) 

def clean_text_light(text):
    return str(text).lower()    

# 4. Embedding Generation   
def get_embeddings(
    method: str, 
    X_train,    
    X_test, 
    max_features: int = 10000,   
) -> Tuple[np.ndarray, np.ndarray]:
    method = method.lower()
    if method == "tfidf":
        tfidf_model = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
        X_train_tfidf = tfidf_model.fit_transform(X_train).toarray()
        X_test_tfidf = tfidf_model.transform(X_test).toarray()
        return X_train_tfidf, X_test_tfidf
    elif method == "word2vec":
        # Tokenize the training data (convert strings to lists of words)
        tokenized_train = [doc.split() for doc in X_train]
        tokenized_test = [doc.split() for doc in X_test]
        
        # Train Word2Vec on the tokenized training data
        w2v_model = Word2Vec(
            sentences=tokenized_train, 
            vector_size=100, 
            window=5, 
            min_count=1, 
            workers=4
        )
        
        def get_sentence_vector(tokenized_doc):
            doc = [word for word in tokenized_doc if word in w2v_model.wv.key_to_index]
            return np.mean(w2v_model.wv[doc], axis=0) if doc else np.zeros(w2v_model.vector_size)

        X_train_w2v = np.array([get_sentence_vector(doc) for doc in tokenized_train])
        X_test_w2v = np.array([get_sentence_vector(doc) for doc in tokenized_test])
        
        return X_train_w2v, X_test_w2v
        
    elif method == "bert":
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        X_train_bert = bert_model.encode(X_train.tolist(), convert_to_numpy=True)
        X_test_bert = bert_model.encode(X_test.tolist(), convert_to_numpy=True)
        return X_train_bert, X_test_bert
    elif method == "glove": 
        glove_model = api.load("glove-wiki-gigaword-100")
        
        def get_glove_vector(doc):
            words = doc.split()
            word_vectors = [glove_model[word] for word in words if word in glove_model]
            return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(glove_model.vector_size)

        X_train_glove = np.array([get_glove_vector(doc) for doc in X_train])
        X_test_glove = np.array([get_glove_vector(doc) for doc in X_test])
        
        return X_train_glove, X_test_glove
    else:
        raise ValueError(f"Embedding method '{method}' not recognized. Choose from 'tfidf', 'word2vec', 'bert', or 'glove'.")


# 5. Model training + Evaluation    
def train_and_eval (models: dict, X_train, y_train, X_test, y_test):
    results = []
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time
        })
        print(classification_report(y_test, y_pred))
    return pd.DataFrame(results, index=models.keys())  

# 6. PII Type Evaluation
def pii_type_evaluation(df):
    rows=[]
    for idx, row in df.iterrows():
        pii_dict = eval(row["PII_types"]) if isinstance(row["PII_types"], str) else row["PII_types"]
        for pii_type, values in pii_dict.items():
            if len(values) > 0:
                rows.append({
                    "index": idx,
                    "PII_types": pii_type,
                    "has_PII": 1
                })
            else:
                rows.append({
                    "index": idx,
                    "PII_types": pii_type,
                    "has_PII": 0
                })
    return pd.DataFrame(rows)

def per_pii_type_report(df_true, df_pred):  
    return classification_report(
        df_true["has_PII"],
        df_pred["has_PII"],
        target_names=df_true["PII_types"].unique().tolist()
    )

# 7. Timing Utility 
def time_pipeline(embedding_method, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    X_train_emb, X_test_emb, _ = get_embeddings(
        method=embedding_method,
        X_train=X_train,
        X_test=X_test
    )
    embedding_time = time.time() - start_time

    start_time = time.time()
    model.fit(X_train_emb, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test_emb)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'embedding_time': embedding_time,
        'training_time': training_time,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


# 8. Full Experiment Pipeline
def experiment_pipeline(embedding_method, classifier, df, text_col="clean_text", label_col="has_PII"):
    # 1. Split
    X_train, X_test, y_train, y_test = stratified_split(
        df,
        text_col=text_col,
        label_col=label_col
    )

    # 2. Embedding (timed)
    t0 = time.time()
    X_train_emb, X_test_emb, emb_model = get_embeddings(
        method=embedding_method,
        X_train=X_train,
        X_test=X_test
    )
    emb_time = time.time() - t0

    # 3. Train model (timed)
    t1 = time.time()
    classifier.fit(X_train_emb, y_train)
    train_time = time.time() - t1

    # 4. Evaluate
    y_pred = classifier.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "embedding": embedding_method,
        "text_col": text_col,
        "classifier": type(classifier).__name__,
        "accuracy": acc,
        "f1_weighted": f1,
        "embedding_time": emb_time,
        "train_time": train_time,
        "model_obj": classifier,
        "emb_model": emb_model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }
