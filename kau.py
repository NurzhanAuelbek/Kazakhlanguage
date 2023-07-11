import gzip
import gensim
import logging

# журнал параметрлерін орнату
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_file = "cc.kk.300.vec.gz"

with gzip.open('cc.kk.300.vec.gz', 'rb') as f:
    for i, line in enumerate(f):
        print(line)
        break


def read_input(input_file):
#     Бұл әдіс кіріс файлын gzip форматында оқиды

    logging.info("reading file {0}...this may take a while".format(input_file))

    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f):

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # алдын ала өңдеуді орындау және шолудың әр мәтіні үшін сөздер тізімін қайтару
            yield gensim.utils.simple_preprocess (line)

# тізімге таңбаланған шолуларды оқиды
# шолудың әр элементі бірқатар сөздерге айналдырады
# Сонымен, бұл тізімдер тізіміне айналады
documents = list (read_input (data_file))
logging.info ("Done reading data file")


# len(documents)

# documents[0]

model = gensim.models.Word2Vec (documents, window=5, min_count=1, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)


#path = get_tmpfile("word2vec_rev.model")
model.save("word2vec.model")

model_new = gensim.models.Word2Vec.load("word2vec.model")

w1 = "программалар"
print(model_new.wv.most_similar(positive=w1))

