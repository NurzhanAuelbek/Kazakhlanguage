from gensim.models import Word2Vec
from nltk.corpus import stopwords
import bs4 as bs
import urllib.request
import re
import nltk
from numpy.core.fromnumeric import size

scrapped_data = urllib.request.urlopen(
    'https://infourok.ru/oushilardi-zheke-mlimetteri-aparatti-zhyesin-ru-248398.html')
article = scrapped_data .read()
parsed_article = bs.BeautifulSoup(article, 'lxml')
paragraphs = parsed_article.find_all('p')
article_text = ""
for p in paragraphs:
    article_text += p.text

# Мәтінді тазалау
processed_article = article_text.lower()
processed_article = re.sub(
    '[^а-яА-ЯәіңғүұқөӘІҢҒҮҰҚёЁһҺ]', ' ', processed_article)
processed_article = re.sub(r'\s+', ' ', processed_article)

# Деректер жиынтығын дайындау
nltk.download('stopwords')
all_sentences = nltk.sent_tokenize(processed_article)
all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
print(all_words)

# Тоқтату Сөздерін Жою
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i]
                    if w not in stopwords.words('kazakh')]


word2vec = Word2Vec(all_words, min_count=3)

vocabulary = word2vec.wv.index_to_key
print(vocabulary)

sim_words = word2vec.wv.most_similar('программалар')
print(sim_words)
