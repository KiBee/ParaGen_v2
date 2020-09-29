import time
from random import randint

import pandas as pd
import pymorphy2
import sqlalchemy
from deeppavlov import configs, build_model

ner_model = build_model(configs.classifiers.paraphraser_rubert, download=True, )

news_lite_engine = 'sqlite:///../Paraphrase_gen/rus_news.db'
news_lite_engine = sqlalchemy.create_engine(news_lite_engine)
lite_engine = 'sqlite:///../Paraphrase_gen/dataset.db'
lite_engine = sqlalchemy.create_engine(lite_engine)


# запрос на сравнение двух словосочетаний на парафразирование
def get_paraans_array(texts1, texts2):
    result = pd.DataFrame(columns=["text1", "text2", "result"])

    res = ner_model(texts1, texts2)
    result.text1 = texts1
    result.text2 = texts2
    result.result = list(res)

    return result


def pos(sentence, morth=pymorphy2.MorphAnalyzer()):
    functors_pos = {'INTJ', 'PRCL', 'CONJ', 'PREP', 'NUMR'}
    try:
        if morth.parse(sentence.split()[-1])[0].tag.POS in functors_pos:
            return False
        else:
            return True
    except:
        print(sentence)


def crazy_v2(epoch=10):
    result = pd.DataFrame(columns=["text1", "text2", "result"])

    lens = 0
    sums = 0

    for i in range(epoch):
        eptime = time.time()
        ls = cutter_v2(lite_rand_sent())
        buf = get_paraans_array(ls[:(len(ls) // 2)], ls[(len(ls) // 2):])
        result = result.append(buf)

        buf.to_sql('ds', con=lite_engine, if_exists='append', index=False)

        sums += buf.result.sum()
        lens += len(buf)
        tms = int(time.time() - start_time)

        print(
            'Ep:', i + 1,
            '\tStr:', str(lens) + '(' + str(len(buf)) + ')',
            '\tRes:', str(sums) + '(' + str(buf.result.sum()) + ')',
            '\tEp_Sec:', str(time.time() - eptime)[:5],
            '\tTime:', str(tms // 3600) + 'h', str(int(tms / 60 % 60)) + 'm', str(int(tms % 60)) + 's'

        )

    return result.reset_index(drop=True)


# Разбиение предложения вида:

def cutter_v2(sentences):
    result = []

    for sentence in sentences:
        buf = sentence.split()

        if len(buf) > 5:
            left_border = randint(0, len(buf) - 5)
            right_border = randint(left_border + 2, left_border + 5)

            buf = " ".join(buf[left_border: right_border])
            if pos(buf):
                result.append(buf)

        elif pos(sentence):
            result.append(sentence)

    if len(result) % 2 == 0:
        return result
    else:
        return result[:-1]


# def rand_sent(comp=200):
#     req = f'''SELECT *
#                 FROM (SELECT ROUND(RAND() * (SELECT MAX(id)
#                                              FROM sentence)) random_num,
#                              @num := @num + 1
#                       FROM (SELECT @num := 0) AS a,
#                            sentence
#                       LIMIT {comp * 2}) AS b,
#                      sentence AS t
#                 WHERE b.random_num = t.id;'''
#
#     lol = pd.read_sql(sql=req, con=engine)
#     lol = lol.drop(columns=['random_num', '@num := @num + 1', 'id'])
#     result = lol.text.to_list()
#
#     return result

def lite_rand_sent(comp=200):
    result = []
    for _ in range(comp * 2):
        rnd = randint(0, 96465321)
        req = f'''
                SELECT *
                FROM sentence
                WHERE id={rnd}
                    '''

        lol = pd.read_sql(sql=req, con=news_lite_engine)
        lol = lol.drop(columns=['id'])
        result.append(lol.text.to_list()[0])

    return result


start_time = time.time()
a = crazy_v2(500)
print(a)
