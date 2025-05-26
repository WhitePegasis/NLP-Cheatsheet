# NLP Cheat Sheet for Interviews

## 1. Core NLP Concepts
### a. Tokenization
- **Explanation**: Splitting text into smaller units (words, sentences, or subwords).
- **Example**: "I love NLP" → ["I", "love", "NLP"]
- **Use Case**: Preprocessing text for search engines or chatbots to analyze individual words.
- **Code Snippet**:
  ```python
  import nltk
  from nltk.tokenize import word_tokenize
  text = "I love NLP"
  tokens = word_tokenize(text)  # Output: ['I', 'love', 'NLP']
  ```

### b. Stop Words Removal
- **Explanation**: Removing common words (e.g., "the", "is") that carry little meaning.
- **Example**: "The cat is cute" → ["cat", "cute"]
- **Use Case**: Improving text classification (e.g., spam detection) by focusing on meaningful words.
- **Code Snippet**:
  ```python
  from nltk.corpus import stopwords
  stop_words = set(stopwords.words('english'))
  words = ["the", "cat", "is", "cute"]
  filtered = [w for w in words if w not in stop_words]  # Output: ['cat', 'cute']
  ```

### c. Stemming & Lemmatization
- **Explanation**: Reducing words to their root form (Stemming: crude; Lemmatization: precise).
- **Example**: Stemming: "jumping" → "jump"; Lemmatization: "ran" → "run"
- **Use Case**: Search engines (Stemming for speed) or sentiment analysis (Lemmatization for accuracy).
- **Code Snippet**:
  ```python
  from nltk.stem import PorterStemmer, WordNetLemmatizer
  stemmer = PorterStemmer()
  lemmatizer = WordNetLemmatizer()
  print(stemmer.stem("jumping"))  # Output: 'jump'
  print(lemmatizer.lemmatize("ran", pos='v'))  # Output: 'run'
  ```

### d. Part-of-Speech (POS) Tagging
- **Explanation**: Assigning grammatical tags (noun, verb, etc.) to words.
- **Example**: "She runs fast" → [("She", "PRP"), ("runs", "VBZ"), ("fast", "RB")]
- **Use Case**: Grammar checkers or text-to-speech systems to understand sentence structure.
- **Code Snippet**:
  ```python
  import nltk
  tokens = ["She", "runs", "fast"]
  pos_tags = nltk.pos_tag(tokens)  # Output: [('She', 'PRP'), ('runs', 'VBZ'), ('fast', 'RB')]
  ```

### e. Named Entity Recognition (NER)
- **Explanation**: Identifying entities like person names, organizations, or locations in text.
- **Example**: "Elon Musk founded xAI" → [("Elon Musk", "PERSON"), ("xAI", "ORG")]
- **Use Case**: Extracting key info from news articles (e.g., people, companies) for knowledge bases.
- **Code Snippet**:
  ```python
  import spacy
  nlp = spacy.load("en_core_web_sm")
  doc = nlp("Elon Musk founded xAI")
  entities = [(ent.text, ent.label_) for ent in doc.ents]  # Output: [('Elon Musk', 'PERSON'), ('xAI', 'ORG')]
  ```

## 2. Text Representation
### a. Bag of Words (BoW)
- **Explanation**: Representing text as a sparse vector of word frequencies, ignoring order.
- **Example**: "I like NLP" → { "I": 1, "like": 1, "NLP": 1 }
- **Use Case**: Simple document classification (e.g., spam vs. not spam) where word order isn’t critical.
- **Code Snippet**:
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  texts = ["I like NLP", "NLP is fun"]
  vectorizer = CountVectorizer()
  bow = vectorizer.fit_transform(texts)  # Output: Sparse matrix
  ```

### b. TF-IDF (Term Frequency-Inverse Document Frequency)
- **Explanation**: Weighing words by importance (frequent in a document but rare across documents).
- **Example**: "NLP is great" → "NLP" gets higher weight if rare in corpus.
- **Use Case**: Keyword extraction for search engines or document ranking in information retrieval.
- **Code Snippet**:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  texts = ["NLP is great", "I love coding"]
  vectorizer = TfidfVectorizer()
  tfidf = vectorizer.fit_transform(texts)  # Output: Sparse matrix with TF-IDF scores
  ```

### c. Word Embeddings (Word2Vec, GloVe)
- **Explanation**: Dense vector representations capturing semantic meaning of words.
- **Example**: "king" - "man" + "woman" ≈ "queen"
- **Use Case**: Semantic search or recommendation systems (e.g., similar product descriptions).
- **Code Snippet**:
  ```python
  from gensim.models import Word2Vec
  sentences = [["I", "love", "NLP"], ["NLP", "is", "fun"]]
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
  vector = model.wv["NLP"]  # Output: 100-dim vector
  ```

### d. Transformers (BERT, GPT)
- **Explanation**: Contextual embeddings capturing word meaning based on context.
- **Example**: "Bank" in "river bank" vs. "money bank" has different embeddings.
- **Use Case**: Question answering systems (e.g., chatbots) needing context-aware responses.
- **Code Snippet**:
  ```python
  from transformers import BertTokenizer, BertModel
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
  inputs = tokenizer("I love NLP", return_tensors="pt")
  outputs = model(**inputs)  # Output: Contextual embeddings
  ```

## 3. Key NLP Algorithms
### a. Naive Bayes (Text Classification)
- **Explanation**: Probabilistic classifier assuming feature independence, great for sentiment analysis.
- **Example**: "I love this" → Positive (based on word probabilities).
- **Use Case**: Sentiment analysis of product reviews or spam email filtering.
- **Code Snippet**:
  ```python
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.feature_extraction.text import CountVectorizer
  X = ["I love this", "I hate this"]
  y = ["positive", "negative"]
  vectorizer = CountVectorizer()
  X_vec = vectorizer.fit_transform(X)
  model = MultinomialNB().fit(X_vec, y)
  ```

### b. RNN (Recurrent Neural Networks)
- **Explanation**: Processes sequential data by maintaining a "memory" of previous inputs.
- **Example**: "I love to" → Predicts next word ("eat", "run", etc.) based on sequence.
- **Use Case**: Early NLP tasks like language modeling or speech recognition.
- **Difference from CNN/ANN**: Unlike CNNs (spatial data like images) or ANNs (feedforward, no memory), RNNs handle sequences with temporal dependencies.

### c. LSTM (Long Short-Term Memory)
- **Explanation**: RNN variant for capturing long-term dependencies in text, avoiding vanishing gradients.
- **Example**: "I forgot to mention earlier, but I love NLP" → Captures "love" despite distance.
- **Use Case**: Text generation (e.g., predictive typing) or sentiment analysis with long sentences.
- **Code Snippet**:
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import LSTM, Dense, Embedding
  model = Sequential()
  model.add(Embedding(1000, 32))
  model.add(LSTM(64))
  model.add(Dense(1, activation='sigmoid'))  # For binary classification
  ```

### d. Attention Mechanism (Transformers)
- **Explanation**: Focuses on relevant parts of input for tasks like translation or summarization.
- **Example**: In "The cat, which is cute, sleeps," attention focuses on "cat" for "sleeps."
- **Use Case**: Machine translation (e.g., Google Translate) or text summarization tools.
- **Code Snippet**: (See BERT example above; attention is built-in.)

### e. Why Transformers Over LSTM?
- **Explanation**: Transformers use self-attention to process all words simultaneously, avoiding sequential bottlenecks of LSTMs. They capture long-range dependencies better and scale with parallel computation.
- **Advantages**: Faster training (parallelism), better performance on large datasets, no gradient vanishing issues.
- **Trade-off**: Higher memory usage and complexity vs. LSTM’s simplicity for small datasets.
- **Use Case**: BERT for contextual understanding vs. LSTM for lightweight sequence tasks.

### f. Difference Between RNN, CNN, and ANN
- **RNN**: Designed for sequential data (e.g., text, time series); has memory via loops.
- **CNN**: Best for spatial data (e.g., images); uses convolution to detect local patterns (can be adapted for NLP).
- **ANN**: Feedforward network, no memory or sequence handling; used for static inputs (e.g., tabular data).
- **Example**: RNN for text generation, CNN for image classification, ANN for simple regression.

### g. Can CNN or ANN Be Used in NLP?
- **CNN in NLP**: Yes, for tasks like text classification (e.g., sentiment analysis). CNNs detect local patterns (n-grams) in text embeddings.
  - **Example**: "I love this" → CNN extracts "love this" as a positive feature.
  - **Code Snippet**:
    ```python
    from tensorflow.keras.layers import Conv1D
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(max_len, embedding_dim)))
    ```
- **ANN in NLP**: Rarely used alone; possible for simple tasks (e.g., BoW-based classification) but lacks sequence/context understanding.
  - **Use Case**: Classifying short, preprocessed text like "spam" vs. "not spam" with BoW.

## 4. Common NLP Tasks
### a. Sentiment Analysis
- **Explanation**: Determining emotion or opinion (positive, negative, neutral) in text.
- **Example**: "Great product!" → Positive
- **Use Case**: Analyzing customer feedback or social media sentiment (e.g., brand monitoring).
- **Pipeline**: 
  1. Preprocess (tokenize, remove stop words, lemmatize).
  2. Vectorize (e.g., TF-IDF or embeddings).
  3. Train model (e.g., Naive Bayes, LSTM, or BERT) on labeled data (positive/negative).
  4. Evaluate (accuracy, F1-score).
- **Code Snippet**:
  ```python
  from transformers import pipeline
  classifier = pipeline("sentiment-analysis")
  result = classifier("Great product!")  # Output: [{'label': 'POSITIVE', 'score': 0.999}]
  ```

### b. Text Generation
- **Explanation**: Generating coherent text (e.g., chatbots, story writing).
- **Example**: Input: "Once upon a time" → Output: "there was a brave knight."
- **Use Case**: Chatbots (e.g., customer support) or creative writing assistants.
- **LSTM/Transformer Output**: 
  - **One Word or Sentence?**: Can predict one word at a time (e.g., next-word prediction) or entire sequences (e.g., autoregressive generation in GPT).
  - **Input/Output Format**: Input as tokenized integers (e.g., [101, 234, 567]); Output as logits over vocabulary or decoded text.
- **Code Snippet**:
  ```python
  from transformers import pipeline
  generator = pipeline("text-generation", model="gpt2")
  output = generator("Once upon a time", max_length=20)  # Generates continuation
  ```

### c. Machine Translation
- **Explanation**: Translating text from one language to another.
- **Example**: "Hola" → "Hello"
- **Use Case**: Real-time translation apps (e.g., travel apps) or multilingual customer support.
- **Code Snippet**:
  ```python
  from transformers import pipeline
  translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
  result = translator("Hello")  # Output: "Bonjour"
  ```

## 5. Libraries to Know
- **NLTK**: Basic NLP tasks (tokenization, stemming, POS tagging) – Use for educational projects or lightweight apps.
- **spaCy**: Fast, industrial-strength NLP (NER, dependency parsing) – Use for production-grade pipelines.
- **scikit-learn**: Traditional ML for text (BoW, TF-IDF, Naive Bayes) – Use for quick prototyping.
- **Transformers (Hugging Face)**: State-of-the-art models (BERT, GPT) – Use for advanced tasks like chatbots.
- **Gensim**: Topic modeling and word embeddings (Word2Vec) – Use for semantic analysis or clustering.

## 6. Additional Interview Questions & Answers
### a. How Does Input/Output Look in NLP Models?
- **Input**: Text → Tokenized (strings to integers via vocabulary, e.g., "I love" → [101, 234]) → Padded/Embedded (fixed-length vectors).
- **Output**: Depends on task:
  - Classification: Probability scores (e.g., [0.9, 0.1] for positive/negative).
  - Generation: Logits over vocabulary (e.g., [0.1, 0.5, …] → "next") or decoded text.
- **Example**: LSTM input: [[101, 234, 0], …]; Output: [0.7] (sentiment score).

### b. What’s the Difference Between Supervised and Unsupervised NLP?
- **Supervised**: Labeled data (e.g., sentiment analysis with positive/negative tags).
- **Unsupervised**: No labels (e.g., topic modeling with LDA or clustering similar documents).
- **Use Case**: Supervised for classification; Unsupervised for exploratory analysis.

### c. How Do You Handle Imbalanced Datasets in NLP?
- **Techniques**: Oversampling (e.g., SMOTE), undersampling, class weights in loss function, or data augmentation (paraphrasing).
- **Example**: Sentiment analysis with 90% positive, 10% negative → Use class weights to penalize majority class.
- **Code Snippet**:
  ```python
  from sklearn.utils.class_weight import compute_class_weight
  class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
  model.fit(X_train, y_train, class_weight=dict(enumerate(class_weights)))
  ```

### d. What’s the Role of Pretraining in Transformers?
- **Explanation**: Pretraining (e.g., BERT on masked language modeling) learns general language patterns; fine-tuning adapts to specific tasks (e.g., classification).
- **Benefit**: Reduces training time/data needs for downstream tasks.
- **Use Case**: Pretrained BERT → Fine-tuned for NER.

---

## Quick Tips for Interviews
1. **Explain Concepts**: Define terms with use cases (e.g., "NER for extracting names from resumes").
2. **Code Basics**: Practice snippets and mention libraries (e.g., `spaCy` for NER).
3. **Know Trade-offs**: E.g., BoW (fast, simple) vs. BERT (slow, contextual); LSTM vs. Transformer.
4. **Applications**: Tie techniques to scenarios (e.g., LSTM for time-series text data, CNN for n-gram detection).
5. **Stay Updated**: Mention recent trends (e.g., LLMs like GPT-4, efficient fine-tuning with LoRA).
