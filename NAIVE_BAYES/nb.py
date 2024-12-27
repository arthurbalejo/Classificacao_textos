#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes NB

# ##### é um classificador probabilístico que utiliza o Teorema de Bayes para prever a probabilidade de uma instância pertencer a uma determinada classe. Ele é chamado de “naive” (ingênuo) porque assume que todas as características (ou atributos) são independentes entre si, o que raramente é verdade na prática, mas simplifica muito os cálculos.

# In[175]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
dados_reviews = pd.read_csv('dados_reviews_tratados.csv', sep = ',')
dados_reviews['content'] = dados_reviews['content'].fillna('')
dados_reviews = dados_reviews[~dados_reviews['sentiment'].isin(['surprise', 'fear'])]


# In[176]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB

# Vectorização com Bag-of-Words
vectorizer_bow = CountVectorizer()
BoW_matriz = vectorizer_bow.fit_transform(dados_reviews['content'])
palavras_bow = vectorizer_bow.get_feature_names_out()
BOW_dataframe = pd.DataFrame(BoW_matriz.toarray(), columns=palavras_bow)

# Vectorização com TF-IDF
vectorizer_tfidf = TfidfVectorizer()
tfidf_matrix = vectorizer_tfidf.fit_transform(dados_reviews['content'])
palavras_tfidf = vectorizer.get_feature_names_out()
TFIDF_dataframe = pd.DataFrame(tfidf_matrix.toarray(), columns=palavras_tfidf)

# Definir as categorias
y = dados_reviews['sentiment']
y_polaridade = dados_reviews['sentiment_polarity']

# Configurar a avaliação cruzada
cv = StratifiedKFold(n_splits=5)


naive_bayes = MultinomialNB()


# # Validação Cruzada Sentimento BoW

# In[177]:


scores = cross_val_score(naive_bayes, BoW_matriz, y, cv=cv, scoring='accuracy')
print("Validação cruzada para Sentiment:", scores)
print("Média dos Scores:", scores.mean())

# Obter previsões de validação cruzada
predictions_BoW = cross_val_predict(naive_bayes, BoW_matriz, y, cv=cv)
print("Relatório de Classificação para Sentiment:")
print(classification_report(y, predictions_BoW, zero_division=0))

# matriz confusão
conf_matrix = confusion_matrix(y, predictions_BoW, labels=y.unique())
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=y.unique())
disp.plot(cmap='Blues')
print("Matriz de Confusão para Sentiment:")
print(conf_matrix)


# # Validação Cruzada Sentimento TF-IDF

# In[178]:


scores = cross_val_score(naive_bayes, tfidf_matrix, y, cv=cv, scoring='accuracy')
print("Validação cruzada para Sentiment:", scores)
print("Média dos Scores:", scores.mean())

# Obter previsões de validação cruzada
predictions = cross_val_predict(naive_bayes, tfidf_matrix, y, cv=cv)
print("Relatório de Classificação para Sentiment:")
print(classification_report(y, predictions, zero_division=0))

# matriz confusão
conf_matrix = confusion_matrix(y, predictions, labels=y.unique())
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=y.unique())
disp.plot(cmap='Blues')
print("Matriz de Confusão para Sentiment:")
print(conf_matrix)


# # Validação Cruzada Polaridade BoW

# In[179]:


# Avaliação cruzada para Sentiment Polarity
scores_polaridade = cross_val_score(naive_bayes, BoW_matriz, y_polaridade, cv=cv, scoring='accuracy')
print("Validação cruzada para Sentiment Polarity:", scores_polaridade)
print("Média dos Scores:", scores_polaridade.mean())

# Obter previsões de validação cruzada para polaridade
predictions_polaridade_BoW = cross_val_predict(naive_bayes, BoW_matriz, y_polaridade, cv=cv)
print("Relatório de Classificação para Sentiment Polarity:")
print(classification_report(y_polaridade, predictions_polaridade_BoW, zero_division=0))

# Gerar e mostrar a matriz de confusão para Sentiment Polarity
conf_matrix_polaridade = confusion_matrix(y_polaridade, predictions_polaridade_BoW, labels=y_polaridade.unique())
disp_polaridade = ConfusionMatrixDisplay(conf_matrix_polaridade, display_labels=y_polaridade.unique())
disp_polaridade.plot(cmap='Blues')
print("Matriz de Confusão para Sentiment Polarity:")
print(conf_matrix_polaridade)


# # Validação Cruzada Polaridade TF-IDF

# In[180]:


# Avaliação cruzada para Sentiment Polarity
scores_polaridade = cross_val_score(naive_bayes, tfidf_matrix, y_polaridade, cv=cv, scoring='accuracy')
print("Validação cruzada para Sentiment Polarity:", scores_polaridade)
print("Média dos Scores:", scores_polaridade.mean())

# Obter previsões de validação cruzada para polaridade
predictions_polaridade_tfidf = cross_val_predict(naive_bayes, tfidf_matrix, y_polaridade, cv=cv)
print("Relatório de Classificação para Sentiment Polarity:")
print(classification_report(y_polaridade, predictions_polaridade_tfidf, zero_division=0))

# Gerar e mostrar a matriz de confusão para Sentiment Polarity
conf_matrix_polaridade = confusion_matrix(y_polaridade, predictions_polaridade_tfidf, labels=y_polaridade.unique())
disp_polaridade = ConfusionMatrixDisplay(conf_matrix_polaridade, display_labels=y_polaridade.unique())
disp_polaridade.plot(cmap='Blues')
print("Matriz de Confusão para Sentiment Polarity:")
print(conf_matrix_polaridade)


# # Teste com avalições da Google Play

# In[181]:


naive_bayes_BoW = MultinomialNB()
naive_bayes_polaridade_BoW = MultinomialNB()
naive_bayes_BoW.fit(BOW_dataframe, y)
naive_bayes_polaridade_BoW.fit(BOW_dataframe, y_polaridade)
naive_bayes_tfidf = MultinomialNB()
naive_bayes_polaridade_tfidf = MultinomialNB()
naive_bayes_tfidf.fit(TFIDF_dataframe, y)
naive_bayes_polaridade_tfidf.fit(TFIDF_dataframe, y_polaridade)


# In[182]:


import pandas as pd
teste_emocoes = pd.read_csv('teste_tratado.csv', sep = ',')
teste_emocoes.head(1)


# In[183]:


avaliacao_BoW = vectorizer_bow.transform(teste_emocoes['content'])
avaliacao_tfidf = vectorizer_tfidf.transform(teste_emocoes['content'])


# # Teste Naive Bayes com BoW sentimento

# In[184]:


emocao_predita = naive_bayes_BoW.predict(avaliacao_BoW)
print('emocao predita:')
print(emocao_predita)
print('emocao real:')
print(list(teste_emocoes['sentiment']))


# # Teste Naive Bayes com TF-IDF sentimento

# In[185]:


emocao_predita = naive_bayes_tfidf.predict(avaliacao_tfidf)
print('emocao predita:')
print(emocao_predita)
print('emocao real:')
print(list(teste_emocoes['sentiment']))


# # Teste Naive Bayes com BoW polaridade

# In[186]:


emocao_predita = naive_bayes_polaridade_BoW.predict(avaliacao_BoW)
print('emocao predita:')
print(emocao_predita)
print('emocao real:')
print(list(teste_emocoes['sentiment_polarity']))


# # Teste Naive Bayes com TF-IDF polaridade

# In[187]:


emocao_predita = naive_bayes_polaridade_tfidf.predict(avaliacao_tfidf)
print('emocao predita:')
print(emocao_predita)
print('emocao real:')
print(list(teste_emocoes['sentiment_polarity']))


# In[ ]:





# In[ ]:




