from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import string
import nltk
from nltk.util import ngrams
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import re

app = Flask(__name__)

dataset = pd.read_csv("bbc_news.csv")
df = dataset[['title', 'description']]
df.drop_duplicates(inplace=True)

raw_data = df.copy()

punctuations = string.punctuation
numbers = '0123456789'
raw_data['removed_puncs_title'] = raw_data['title'].apply(lambda x: ''.join(char for char in x if char not in punctuations))
raw_data['removed_puncs_desc'] = raw_data['description'].apply(lambda x: ''.join(char for char in x if char not in punctuations))
raw_data = raw_data.drop(columns=['title', 'description'])

raw_data['removed_numbers_title'] = raw_data['removed_puncs_title'].apply(lambda x: ''.join(char for char in x if char not in numbers))
raw_data['removed_numbers_desc'] = raw_data['removed_puncs_desc'].apply(lambda x: ''.join(char for char in x if char not in numbers))
raw_data = raw_data.drop(columns=['removed_puncs_title', 'removed_puncs_desc'])

raw_data['lower_title'] = raw_data['removed_numbers_title'].apply(lambda x: ''.join(char.lower() for char in x))
raw_data['lower_desc'] = raw_data['removed_numbers_desc'].apply(lambda x: ''.join(char.lower() for char in x))
raw_data = raw_data.drop(columns=['removed_numbers_title', 'removed_numbers_desc'])

sentences = [item for sublist in zip(raw_data['lower_title'], raw_data['lower_desc']) for item in sublist]

nltk.download('punkt')

# Tokenisasi dan pembuatan n-gram (hingga 3-gram untuk contoh ini)
tokens = [word_tokenize(sentence.lower()) for sentence in sentences]

# Membuat bigram dan trigram
bigrams = [bigram for sentence in tokens for bigram in ngrams(sentence, 2)]
trigrams = [trigram for sentence in tokens for trigram in ngrams(sentence, 3)]

# Hitung frekuensi bigram dan trigram
bigram_freq = FreqDist(bigrams)
trigram_freq = FreqDist(trigrams)

# Hitung frekuensi total untuk unigram (kata)
unigram_freq = FreqDist([word for sentence in tokens for word in sentence])

# Menghitung probabilitas bigram
def bigram_probability(word1, word2):
    bigram = (word1, word2)
    return bigram_freq[bigram] / unigram_freq[word1] if unigram_freq[word1] > 0 else 0

# Menghitung probabilitas trigram
def trigram_probability(word1, word2, word3):
    trigram = (word1, word2, word3)
    bigram = (word1, word2)
    return trigram_freq[trigram] / bigram_freq[bigram] if bigram_freq[bigram] > 0 else 0

# Fungsi untuk memberikan saran kata berdasarkan input
def suggest_next_word(input_text, num_suggestions=3):
    words = word_tokenize(input_text.lower())
    num_words = len(words)
    
    suggestions = defaultdict(float)
    
    if num_words == 1:
        last_word = words[-1]
        for next_word in unigram_freq:
            prob = bigram_probability(last_word, next_word)
            if prob > 0:
                suggestions[next_word] = prob
    
    elif num_words == 2:
        last_two_words = tuple(words[-2:])
        for next_word in unigram_freq:
            prob = trigram_probability(last_two_words[0], last_two_words[1], next_word)
            if prob > 0:
                suggestions[next_word] = prob
    
    sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
    return sorted_suggestions[:num_suggestions]

@app.route('/suggest', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_word = request.form['input_word']
        suggestions = suggest_next_word(input_word)
        return render_template('home.html', suggestions=suggestions, input_word=input_word)
    return render_template('home.html')

@app.route('/output')
def output():
    input_word = request.args.get('input_word', '')
    chosen_word = request.args.get('chosen_word', '')
    combined_result = f"{input_word} {chosen_word}"

    # Memastikan DataFrame tidak kosong dan kondisi boolean cocok
    if dataset.empty:
        filtered_articles = pd.DataFrame(columns=dataset.columns)
    else:
        # Menggunakan regex dengan word boundaries untuk memastikan hanya frasa penuh yang dicocokkan
        title_condition = dataset['title'].str.contains(rf'\b{re.escape(combined_result)}\b', case=False, na=False)
        description_condition = dataset['description'].str.contains(rf'\b{re.escape(combined_result)}\b', case=False, na=False)

        if title_condition.any() or description_condition.any():
            filtered_articles = dataset[title_condition | description_condition]
        else:
            filtered_articles = pd.DataFrame(columns=dataset.columns)

    # Jika tidak ada artikel yang ditemukan, kirimkan daftar kosong ke template
    if filtered_articles.empty:
        articles = []
    else:
        # Highlight atau bold hanya frasa kombinasi
        def highlight_combined_result(text, combined_result):
            # Menggunakan regex untuk mencocokkan frasa penuh, case-insensitive
            pattern = re.compile(rf'(\b{re.escape(combined_result)}\b)', re.IGNORECASE)
            highlighted_text = pattern.sub(r"<b>\1</b>", text)
            return highlighted_text

        filtered_articles['title'] = filtered_articles['title'].apply(lambda x: highlight_combined_result(x, combined_result))
        filtered_articles['description'] = filtered_articles['description'].apply(lambda x: highlight_combined_result(x, combined_result))

        articles = filtered_articles.to_dict(orient='records')

    return render_template('output.html', input_word=input_word, chosen_word=chosen_word, combined_result=combined_result, articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
