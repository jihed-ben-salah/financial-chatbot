import os
import argparse
import PyPDF2
from textblob import TextBlob
import stopwordsiso as stopwords
from nltk.corpus import stopwords as sp
from transformers import AutoTokenizer, pipeline, AutoModelWithLMHead
from textblob import TextBlob
import nltk
import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import ngrams
import spacy
import datefinder
from langdetect.lang_detect_exception import LangDetectException
import pdfplumber
import gensim
import pandas as pd 
import matplotlib as mpl
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', category=UserWarning)
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import re
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.corpora import Dictionary
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from spellchecker import SpellChecker
import fitz
from unidecode import unidecode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from gensim.models import Word2Vec
import ast
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import torch



def common_words(text):
    tokenized_words = nltk.word_tokenize(text)
    words = []
    freq = []
    word_counts = Counter(tokenized_words)
    most_common = word_counts.most_common(30)
    with open('./EDA_output/most common words.txt', "a",encoding='utf-8') as file:
        for item in most_common:
            file.write(item[0]+':'+str(item[1])+'\n')
            words.append(item[0])
            freq.append(item[1])
        fig = plt.figure(1)
        sns.barplot(x=freq, y=words)
        plt.title('Top 30 Most Frequently Occurring Words')
        fig.savefig("./EDA_output/most_frequent_words.png", dpi=900)
        plt.show()

def chunking(text):
    text_spllitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex = False,
    )
    chunks=text_spllitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, conversation):
    response = conversation({'question': user_question})

    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            print(f"User: {message.content}")
        else:
            print(f"Bot: {message.content}")


def sentence_based_chunking(text, max_chunk_words=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_word_count = 0
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        sent_word_count = len(words)
        if current_chunk_word_count + sent_word_count <= max_chunk_words:
            current_chunk.append(sent)
            current_chunk_word_count += sent_word_count
        else:
            chunks.append(current_chunk)
            current_chunk = [sent]
            current_chunk_word_count = sent_word_count
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def en_fr_translate(text,model,tokenizer):
    inputs = tokenizer(text, return_tensors="pt",truncation=True,max_length=512)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    print(translated_text)
    print()
    return translated_text

def get_vectorstore(text_chrunk):
    embeddings=HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore=FAISS.from_texts(texts=text_chrunk,embedding=embeddings)
    return vectorstore


def extract_dates_from_pdf(pdf_path):
    dates = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            matches = datefinder.find_dates(text)
            for match in matches:
                dates.append(match)
    return dates

# using PyPDF2
def extract_text_from_pdf_1(pdf_path,file_name,trans_model,trans_tokenizer,dic_pdf={},start_page=0,end_page=-1):
    initial_text=""
    processed_text=""
    with open(pdf_path, 'rb') as file:     
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        for page_num in range(start_page, total_pages):          
            page = pdf_reader.pages[page_num] 
            clean_text,tokens=preprocess_text(page.extract_text())
            dic_pdf['text'].append(regular_expressions(page.extract_text()))
            dic_pdf['file'].append(file_name)
            dic_pdf['clean_text'].append(clean_text)
            dic_pdf['page_number'].append(page_num+1)
            dic_pdf['translated_text']=en_fr_translate(regular_expressions(page.extract_text()),trans_model,trans_tokenizer)
            # Calculating subjectivity score
            blob = TextBlob(clean_text)
            dic_pdf['subjectivity_score'].append(blob.sentiment.subjectivity)
            initial_text+=page.extract_text()+'\n'
            processed_text+=clean_text+' '
        return processed_text,dic_pdf,initial_text
    
#PyMuPDF 
def extract_text_from_pdf_2(file_name,pdf_path="",decoded_content=""):
    if pdf_path=="":
        doc = fitz.open(stream=decoded_content, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)
    
    block_dict = {}
    page_num = 1
    for page in doc: 
        file_dict = page.get_text('dict') 
        block = file_dict['blocks']
        block_dict[page_num] = block 
        page_num += 1 
    rows = []
    for page_num, blocks in block_dict.items():
        page_text=""
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    for span in line['spans']:
                        """font_size = span['size']
                        span_font = span['font']
                        is_upper = False
                        is_bold = False
                        if "bold" in span_font.lower():
                            is_bold = True
                        if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                            is_upper = True"""
                        
                        text = unidecode(span['text'])
                        #translated_text=en_fr_translate(text,trans_model,trans_tokenizer)
                        page_text+=text+'\n'
        clean_text,tokens=preprocess_text(page_text) 
        blob = TextBlob(page_text)
        subjectivity_score=blob.sentiment.subjectivity     
        date,bank,title= file_name.split('_')          
        rows.append((file_name,page_num,page_text,clean_text,tokens,subjectivity_score,date,bank,title))
        span_df = pd.DataFrame(rows, columns=['file_name','page_num','text','clean_text','tokens','subjectivity_score','date','bank','title'])         
    return span_df

def get_page_text(pdf_path):
    page_text=""
    doc = fitz.open(pdf_path)
    for page in doc:
        output = page.get_text("blocks")
        previous_block_id = 0 # Set a variable to mark the block id
        for block in output:
            if block[6] == 0: # We only take the text
                if previous_block_id != block[5]:
                    # Compare the block number
                    page_text+="\n"
                page_text+=block[4]
    
    return page_text
        

def find_similar_words(text,word_comp):
    words = nltk.word_tokenize(text)
    similar_pairs = []
    for i, sentence1 in enumerate(words):
        similarity_score = fuzz.ratio(sentence1, word_comp)
        threshold = 80
        if similarity_score >= threshold:
            similar_pairs.append(sentence1)
    return similar_pairs


def lemmatize_stemming(text) :
  stemmer = SnowballStemmer('english')
  return stemmer.stem(WordNetLemmatizer().lemmatize(text))

def ngram(n,text):
    tokens = nltk.word_tokenize(text)
    n_gram = list(ngrams(tokens, n))
    n_gram_counts = Counter(n_gram)
    words=[]
    count=[]
    most_frequent_grams = n_gram_counts.most_common(30)

    for i in range(30):
        words.append(most_frequent_grams[i][0])
        count.append(most_frequent_grams[i][1])
    
    ngram_freq=pd.DataFrame({"ngram":words,"frequency":count})
    ngram_freq.to_csv('./data/'+str(n)+'-gram.csv',index=False)
    fig = plt.figure(1,figsize=(20, 6))
    sns.barplot(x=ngram_freq['frequency'], y=ngram_freq['ngram'])
    if n==2:
        type='bi'
    elif n==3:
        type='tri'

    plt.title('Top 30 Most Frequently Occurring '+type+'-gram')
    fig.savefig("./EDA_output/most_frequent_"+str(n)+"gram.png", dpi=900,format='png')
    plt.show()
    path_='./EDA_output/most common '+str(n)+'-gram.txt'
    if os.path.exists(path_)==False:
        with open(path_, 'a',encoding='utf-8') as file:
            for gram, count in most_frequent_grams:
                file.write(str(gram) + ' ' + str(count) + '\n')
    else:
        print('The file already exists')

def preprocess_text(text):
    text_reg= regular_expressions(text)
    #removing numbers
    text_clean = re.sub(r'\d+', '', text_reg)
    tokens = word_tokenize(text_clean)
    # Removing punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    # Removing stop words
    try:
        stop_words = set(stopwords.stopwords(detect(text_clean)))
    except LangDetectException as e:
        # Handle the exception gracefully
        print("Error occurred during language detection: ", e)
        stop_words = set("en") 
    repetitive_words=["morgan","and/or",'last','source','without','us','j.p.','end',
                            'would','purposes','goldman','morgan','kkr','www','may','generally',
                            'private','see','bank','person','used','subject','kingdom','within',
                            'year','data','please','date','given','author','outlook','sachs','bnp']
    
    tokenized_corpus_lem = [lemmatize_stemming(token.lower()) for token in tokens if (token.lower() not in stop_words) and (token.lower() \
                                                                                                        not in repetitive_words) and (len(token)<15)]
    tokenized_corpus= [token.lower() for token in tokens if (token.lower() not in stop_words) and (token.lower() not in repetitive_words) and (len(token)<15)]
    preprocessed_text_lem = ' '.join(tokenized_corpus_lem)
    preprocessed_text = ' '.join(tokenized_corpus)
    return preprocessed_text_lem,tokenized_corpus
    
def read_files_in_directory(directory_path):
    all_files = []
    for filename in os.listdir(directory_path):
        all_files.append(filename)
    return all_files

def regular_expressions(text):
    # remove links
    text_link = re.sub(r"http\S+", " ", text)
    #remove special characters
    text_spc =re.sub(r'\W+', ' ', text_link)
    # remove all single characters
    text_clean = re.sub(r'\s+[a-zA-Z]\s+', ' ', text_spc)
    #remove repeted letters like 'aaa' 'cc'
    text_clean= re.sub(r'\b(\w)\1+\b',' ',text_clean)
    return text_clean

def search_NER(text,f_name):
    name,extention=f_name.split('.')
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    # Iterate over each named entity
    for entity in doc.ents:
        with open('./EDA_output/'+name+'_NER.txt', 'a',encoding='utf-8') as file:
            file.write(entity.text + ' : ' + entity.label_ + '\n')
    #spacy.displacy.serve(doc,style="ent")
    
def summarization_t5(text,model,tokenizer,max_len):
    print(text)   
    inputs=tokenizer.encode('summarize this financial text: '+text,return_tensors='pt',truncation=True)
    outputs=model.generate(inputs,max_length=max_len,min_length=50,length_penalty=5.,num_beams=2)
    summary=tokenizer.decode(outputs[0])
    return summary

def summarization_hf(text,summarizer):
    summary=summarizer(text,max_length=130,min_length=30,do_sample=False)
    return summary[0]['summary_text']


def tfidf_vect(corpus):
    v = TfidfVectorizer()
    tfidf_features=v.fit_transform(corpus)    
    df = pd.DataFrame(tfidf_features.toarray(), columns=v.get_feature_names_out())
    df = df.transpose().reset_index()
    return df

def topic_modeling(n_topics,text):
    processed_docs = text.tolist()
    tokenized_docs = [word_tokenize(doc.lower()) for doc in processed_docs]
    stop_words = set(sp.words('english'))
    filtered_docs = [[word for word in doc if word.lower() not in stop_words] for doc in tokenized_docs]
    dictionary = Dictionary(filtered_docs)
    dictionary.filter_extremes(no_below=1, keep_n=300)
    bow_corpus = [dictionary.doc2bow(doc) for doc in filtered_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics =n_topics, id2word = dictionary, passes = 12)
    return lda_model,bow_corpus,dictionary 


def unique_sentences(text):
    sentences = text.split('.')
    uniq_sentences = list(set([sentence.strip() for sentence in sentences]))
    cleaned_text = '. '.join(filter(None, uniq_sentences))
    return cleaned_text

def word_cloud(corpus):
    mpl.rcParams['font.size']=14              
    mpl.rcParams['savefig.dpi']=100             
    mpl.rcParams['figure.subplot.bottom']=.1 
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
                                background_color='white',
                                stopwords=stopwords,
                                max_words=1000,
                                max_font_size=60, 
                                random_state=42
                                ).generate(str(corpus))

    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    fig.savefig("./EDA_output/word_cloud.png", dpi=900)




def main(function_name,input_directory="./pdf",texts_directory="./extracted_data",output_directory="./extracted_data"):
    
    if function_name=='txt_pymupdf':
        files_names= read_files_in_directory('./pdf')
        data=pd.DataFrame(columns=['file_name','page_num','text','clean_text','tokens','subjectivity_score','date','bank','title'])
        for f in files_names:  
            name,extention=f.split('.')
            pdf_path=input_directory+'/'+f   
            data_f=extract_text_from_pdf_2(name,pdf_path)   
            data=pd.concat([data,data_f], ignore_index=True)
            print(name)
            print()
        
        if os.path.exists('./data/data.csv')==False:
            data.to_csv('./data/data.csv', index=False)
        else:
            print('The file already exists')
        #documents=dic_pdf['text']
        """ 
        
        
        """
        """for t in final_data['text']:
            print(t)
            search_NER(t)
        """

        
        #dates=extract_dates_from_pdf('./data/01 01 2023_Goldman Sachs_Caution Heavy Fog.pdf')


        
    if function_name=='extract_text':
        corpus=""
        dic_pdf={
        'file':[],
        'page_number':[],
        'text':[],
        'clean_text':[],
        'text':[],
        'subjectivity_score':[]
        }
        files_names= read_files_in_directory('./pdf')
        translation_model_name = "Helsinki-NLP/opus-mt-en-fr"
        trans_model = MarianMTModel.from_pretrained(translation_model_name)
        trans_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
        final_data=pd.DataFrame(dic_pdf)
        for f in files_names:  
            name,extention=f.split('.')
            pdf_path=input_directory+'/'+f
            clean_text,dic_pdf,initial_text= extract_text_from_pdf_1(pdf_path,f,trans_model,trans_tokenizer,dic_pdf)  
            dataframe=pd.DataFrame(dic_pdf)
            final_data=pd.concat([final_data,dataframe],axis=0)
            corpus+=clean_text 
        final_data[['date', 'bank', 'title']] = final_data['file'].str.split('_', expand=True)
        final_data.drop('file',axis=1,inplace=True)
        final_data.to_csv('./data/data_pypdf2.csv', index=False)
        if os.path.exists('./EDA_output/corpus.txt')==False:
            with open('./EDA_output/corpus.txt', 'a',encoding='utf-8') as file:
                file.write(corpus)
        else:
            print('The file already exists')

    if function_name=='get_corpus':
        data=pd.read_csv('./data/data.csv')
        corpus=data['clean_text'].str.cat(sep=' ')
        if os.path.exists('./EDA_output/corpus.txt')==False:
            with open('./EDA_output/corpus.txt', 'a',encoding='utf-8') as file:
                file.write(corpus)
        else:
            print('The file already exists')
        
    if function_name=='text_analysis':    
        with open('./EDA_output/corpus.txt', 'r',encoding='utf-8') as file:
            corpus=file.read()
        word_cloud(corpus)
        common_words(corpus)
        ngram(2,corpus)
        ngram(3,corpus)

    if function_name=='NER':
        txt_files= read_files_in_directory('./extracted_data')
        for f in txt_files:
            file_path='./extracted_data/'+f
            with open(file_path, 'r',encoding='utf-8') as file:
                file_contents = file.read()
                search_NER(file_contents,f)

                
    if function_name=='similarity':
        with open('./extracted_data/01 01 2023_Goldman Sachs_Caution Heavy Fog.txt', 'r',encoding='utf-8') as file:
            file_contents = file.read()
        data=pd.read_csv('classified_texts.csv')
        text=data['text'][6]
        word_comp='allocate'
        print(text)
        s=find_similar_words(text,word_comp)
        print(s)



    if function_name=="data_extraction":
        files_names= read_files_in_directory('./pdf')
        for f in files_names:      
            name,extention=f.split('.')
            pdf_path='./pdf/'+f
            page_text=get_page_text(pdf_path)   
            output_path='./extracted_data/'+name+'.txt'
            if os.path.exists(output_path)==False:   
                with open(output_path, "a",encoding='utf-8') as file:
                    file.write(page_text)
            else:
                print('The file already exists')
            

    if function_name=='summary_t5':
        tokenizer=AutoTokenizer.from_pretrained('t5-base')
        model=AutoModelWithLMHead.from_pretrained('t5-base',return_dict=True)
        files_names= read_files_in_directory('./extracted_data')
        max_len=750
        output_dir='summary_max_len_'+str(max_len)
        for f in files_names:  
            print ()
            name,extention=f.split('.')
            print(name)
            path=texts_directory+'/'+f
            with open(path, 'r',encoding='utf-8') as file:
                file_contents = file.read()
                summary=summarization_hf(file_contents,model)
                with open('./EDA_output/'+ output_dir+'/'+name+'_summary.txt', "a",encoding='utf-8') as file:
                    file.write(summary)

    if function_name=='summary_hf':
        tokenizer=AutoTokenizer.from_pretrained('t5-base')
        model=AutoModelWithLMHead.from_pretrained('t5-base',return_dict=True)
        summarizer=pipeline('summarization')
        files_names= read_files_in_directory('./extracted_data')
        max_len=1500
        separator=" "
        for f in files_names:  
            name,extention=f.split('.')
            path=texts_directory+'/'+f
            output_path='./EDA_output/summary_hf/'+name+'_summary.txt'
            with open(path, 'r',encoding='utf-8') as file:
                if os.path.exists(output_path)==False: 
                    print(name)
                    print ()
                    summary=""
                    file_contents = file.read()
                    chunks=chunking(file_contents)
                    for i in range(len(chunks)):
                        text=separator.join(chunks[i])
                        print("\nlen: ",len(text))
                        summary+=' '+summarization_t5(text,model,tokenizer,max_len)+'\n'
                    with open('./EDA_output/summary_hf/'+name+'_summary.txt', "a",encoding='utf-8') as file_s:
                        file_s.write(summary)
                else:
                    print('The file already exists')
    
    if function_name=='chunk':
        data=pd.read_csv('./data/data.csv')
        clean_data= data[data['text'].str.len() > 200]
        clean_data.dropna(inplace=True)
        clean_data.reset_index(drop=True, inplace=True)
        chunked_texts=[chunking(clean_data['text'][i]) for i in range(len(clean_data))]
        clean_data['chunked_text']=chunked_texts
        clean_data.to_csv('./data/data_chunk.csv',index=False)
        #vectorstore=get_vectorstore(chunked_texts)

    if function_name=='get_doc_data':
        data_financial=pd.read_csv('./data/financial_data.csv')
        data_financial['text']=data_financial.groupby('file_name')['text'].transform(lambda x: ' '.join(x))
        data_financial['tokens']=data_financial.groupby('file_name')['tokens'].transform(lambda x: ' '.join(x))
        data_financial['clean_text']=data_financial.groupby('file_name')['clean_text'].transform(lambda x: ' '.join(x))
        data_financial.drop_duplicates(subset=['file_name', 'text', 'clean_text','tokens'], inplace=True)
        data_financial.reset_index(drop=True, inplace=True)
        tfidf_vect(data_financial['tokens'])
        #test=[ for i in range(len(data_financial)))]
        data_financial.to_csv('./data/pdfs.csv',index=False)

    if function_name=='embedding':
        data_financial=pd.read_csv('./data/financial_data.csv')
        tokenized_text=[ast.literal_eval(data_financial['tokens'][i]) for i in range(len(data_financial))]
        model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=5, sg=1) 
        model.save('word2vec_model.model')
        #loading the model
        #model = Word2Vec.load('../word2vec_model.model')
        vocab = list(model.wv.index_to_key)
        word_embeddings = [model.wv[word] for word in vocab]
        embedding_df = pd.DataFrame(word_embeddings, index=vocab)
        embedding_df.to_csv('./dash-tsne/data/financial.csv', header=True)



 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove lines with only style from an HTML file.')
    parser.add_argument('function', type=str, help='Name of the function to use')
    #parser.add_argument('input_directory', type=str, help='Directory of the output files')
    #parser.add_argument('output_directory', type=str, help='Directory of the output files')
    args = parser.parse_args()
    main(args.function)
