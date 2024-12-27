#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
dados_reviews = pd.read_csv('apps_reviews_validacao.csv', sep=',')
dados_reviews.head(2)


# In[15]:


dados_reviews.describe()


# In[16]:


dados_reviews.loc[0,'content']


# # pre-processamento
# ## removendo caracteres nao latinos
# 

# In[17]:


import regex # trabalhar com expressões regulares
dados_reviews['content'] = dados_reviews['content'].apply(lambda x: regex.sub(r'[^\p{Latin}]', u' ', str(x)))


# In[18]:


dados_reviews.loc[0,'content']


# ## tirando maiusculas

# colocando textos minusculo

# In[19]:


dados_reviews['content'] = dados_reviews['content'].apply(lambda x: str(x).lower())


# In[7]:


dados_reviews.loc[0,'content']


# ## removendo stop words

# In[8]:


'''import nltk #caso não funcione, adicione a linha nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(texto):
    stops_list = stopwords.words("portuguese")
    word_tokens = word_tokenize(texto)
    texto_sem_stops = [w for w in word_tokens if w not in stops_list]
    return " ".join(texto_sem_stops)

#dados_reviews['content'] = dados_reviews['content'].apply(remove_stopwords)'''


# In[9]:


'''import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Baixar os recursos necessários
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(texto):
    stops_list = set(stopwords.words("portuguese"))
    word_tokens = word_tokenize(texto.lower())  # Converter para minúsculas
    word_tokens = [word for word in word_tokens if word.isalnum()]  # Remover pontuação
    texto_sem_stops = [w for w in word_tokens if w not in stops_list]
    return " ".join(texto_sem_stops)

# Exemplo de uso
# dados_reviews['content'] = dados_reviews['content'].apply(remove_stopwords)'''


# In[22]:


import spacy

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load('pt_core_news_sm')

def remove_stopwords(text):
    """
    Remove stopwords de um texto usando spaCy.
    
    Args:
    text (str): Texto de entrada.
    
    Returns:
    str: Texto sem stopwords.
    """
    # Processar o texto
    doc = nlp(text)
    
    # Filtrar tokens que não são stopwords e juntar o texto de volta
    tokens_without_stopwords = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens_without_stopwords)

# Exemplo de uso da função
sample_text = "This is an example sentence demonstrating the removal of stopwords."
cleaned_text = remove_stopwords(sample_text)
print("Texto original:", sample_text)
print("Texto sem stopwords:", cleaned_text)
dados_reviews['content'] = dados_reviews['content'].apply(remove_stopwords)


# In[23]:


dados_reviews.loc[0,'content']


# ## lematização

# In[12]:


'''import pandas as pd
import spacy

# Carregar o modelo de linguagem do spaCy para português
nlp = spacy.load('pt_core_news_sm')

# Função para lematizar um texto
def lemmatize_text(texto):
    doc = nlp(texto)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)


# Aplicar lematização à coluna 'content'
dados_reviews['content'] = dados_reviews['content'].apply(lemmatize_text)'''


# In[24]:


import pandas as pd
import spacy

# Carregar o modelo de linguagem do spaCy para português
nlp = spacy.load('pt_core_news_sm')

# Função para lematizar um texto
def lemmatize_text(texto):
    # Verificar se o texto é nulo ou não é uma string
    if not isinstance(texto, str):
        return ""
    doc = nlp(texto)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

# Aplicar lematização à coluna 'content'
dados_reviews['content'] = dados_reviews['content'].apply(lemmatize_text)


# In[25]:


dados_reviews.loc[0,'content']


# In[11]:


'app shopee razoável   apesar fácil utilizar   apresenta bugs     sugestão palavra errada campo pesquisa acarreta busca errada   sugestões produtos coisas interesse   dificuldade acesso jogos travamento durante alguma jogada   falhas carregar certos anexos avaliação produtos     problemas contornar   ocorrem frequência'


# In[26]:


dados_reviews['content'] = dados_reviews['content'].fillna('')
dados_reviews.to_csv('dados_reviews_tratados.csv', index=False)


# In[27]:


dados_reviews[['sentiment','sentiment_polarity']].describe()


# In[28]:


dados_reviews['sentiment'].value_counts()


# ## tratando dados usados para teste

# In[29]:


teste_emocoes = pd.read_csv('teste.csv', sep = ',', encoding = 'latin-1')
teste_emocoes


# instanciando metodos que fara a representacao do texto usando o modelo bag of word

# In[33]:


teste_emocoes['content'] = teste_emocoes['content'].apply(lambda x: regex.sub(r'[^\p{Latin}]', u' ', str(x)))
teste_emocoes['content'] =teste_emocoes['content'].apply(lambda x: str(x).lower())
teste_emocoes['content'] = teste_emocoes['content'].apply(remove_stopwords)
teste_emocoes['content'] = teste_emocoes['content'].apply(lemmatize_text)
teste_emocoes['content']


# In[34]:


teste_emocoes.to_csv('teste_tratado.csv', index=False)


# In[ ]:




