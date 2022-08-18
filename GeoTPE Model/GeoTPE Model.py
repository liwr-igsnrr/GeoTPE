from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input, TimeDistributed, Embedding,Attention
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import pandas as pd
from itertools import groupby
from tensorflow import concat
import tensorflow as tf
import numpy as np
from CRF_TF2 import CRF
from nltk import pos_tag
from transformers import BertTokenizer,BertConfig
from transformers import TFBertModel
import os
from transformers import logging
import string
from tensorflow.keras.initializers import RandomUniform
import itertools
from sklearn.preprocessing import StandardScaler
from math import pow,floor
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau

#Transformers    version 4.7.0
#Tensorflow      version 2.3.0
#Python          version 3.8

                   
def file_read(input_url):
    data_read=pd.read_csv(input_url)
    words=list(data_read['paper'])
    lalbels=list(data_read['annotation'])
    pos=list(data_read['pos_tag'])
    return words,lalbels,pos

#transform label into id
def idx_switch(sentence,label_tag,label_id):
    idx=sentence.copy()
    for i in range(len(sentence)):
        if sentence[i] in label_tag:
           idx[i]=label_id[label_tag.index(sentence[i])]
        else:
            idx[i]=len(label_tag)
    return idx

#sentence segmention
def sentence_segmention(words_sequences,label_sequences,pos_sequences):
    Index_fullstop = [i for i, x in enumerate(words_sequences) if x == '.']
    # print(Index_fullstop)

    words_sentences_nest=[]

    labels_sentences_nest=[]

    pos_sentences_nest=[]

    
    char_sentences_nest=[]

    
    first_sentence_word=words_sequences[:Index_fullstop[0]+1]
    words_sentences_nest.append(first_sentence_word)
   

    first_sentence_label=label_sequences[:Index_fullstop[0]+1]
    labels_sentences_nest.append(first_sentence_label)
   

    first_sentence_pos = pos_sequences[:Index_fullstop[0] + 1]
    pos_sentences_nest.append(first_sentence_pos)
    

    first_sentence_word_copy=first_sentence_word.copy()
    for i in range(len(first_sentence_word)):
        first_sentence_word_char=list(first_sentence_word[i])
        first_sentence_word_copy[i]=first_sentence_word_char

    char_sentences_nest.append(first_sentence_word_copy)
    

    for j in range(len(Index_fullstop)):
        if j%100==0:
            print(j)

        if j!=len(Index_fullstop)-1:
           #word
           word_sentence=words_sequences[Index_fullstop[j]+1:Index_fullstop[j+1]+1]
           words_sentences_nest.append(word_sentence)
           

           #part of speech
           pos_sentence = pos_sequences[Index_fullstop[j] + 1:Index_fullstop[j + 1] + 1]
           pos_sentences_nest.append(pos_sentence)
           # print('pos_sentence:', len(pos_sentence), pos_sentence)

           #label
           label_sentence = label_sequences[Index_fullstop[j] + 1:Index_fullstop[j + 1] + 1]
           labels_sentences_nest.append(label_sentence)
           # print('label_sentence:', len(label_sentence), label_sentence)

           #char
           word_sentence_copy = word_sentence.copy()
           for k in range(len(word_sentence)):
               word_char = list(str(word_sentence[k]))
               word_sentence_copy[k] = word_char
           char_sentences_nest.append(word_sentence_copy)
           

    return words_sentences_nest,labels_sentences_nest,pos_sentences_nest,char_sentences_nest


def create_inputs(words_sentences_nest,pos_sentences_nest,char_sentences_nest,labels_sentences_nest):

    maxlenth_sentence=300
    maxlenth_word=64

    input_idx=[]
    token_type_idx=[]
    attention_mask_idx=[]
    pos_idx=[]
    label_idx=[]
    char_idx=[]
    sentences_lenth=[]
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    
    pos_tag=["ADP","DET","NOUN","ADJ","PUNCT","VERB","AUX","PART","ADV","PRON","PROPN","NUM","CCONJ","X","SYM","SCONJ","INTJ"]
    pos_tag_id=list(range(len(pos_tag)))
    

    
    label_tag=['O','B-topic','I-topic']
    label_tag_id=list(range(len(label_tag)))
    

   
    char_tag=list(string.printable)
    char_tag_id=list(range(len(char_tag)))
   



    for i in range(len(words_sentences_nest)):
        if i%100==0:
            print(i)
        sentence_lenth=len(words_sentences_nest[i])
        sentences_lenth.append(sentence_lenth)
        
        input_id=tokenizer.convert_tokens_to_ids(words_sentences_nest[i])
        input_id_padding=input_id+[0]*(maxlenth_sentence-len(input_id))
        input_idx.append(input_id_padding)
        # print("input_id:", len(input_id_padding), input_id_padding)

        
        token_type_id=[0] * len(input_id_padding)
        token_type_idx.append(token_type_id)
        

        
        attention_mask_id = [1] * len(input_id)+ ([0] * (maxlenth_sentence-len(input_id)))
        attention_mask_idx.append(attention_mask_id)
        

        #pos_idx
        pos_id=idx_switch(pos_sentences_nest[i],pos_tag,pos_tag_id)+([0] * (maxlenth_sentence-len(pos_sentences_nest[i])))
        pos_idx.append(pos_id)
        # print("pos_id:",len(pos_id),pos_id)

        
        label_id = idx_switch(labels_sentences_nest[i], label_tag, label_tag_id) + ([0] * (maxlenth_sentence - len(labels_sentences_nest[i])))
        label_idx.append(label_id)
        # print("label_id:", len(label_id), label_id)


        #char_idx
        char_sentences_id=char_sentences_nest[i].copy()
        for j in range(len(char_sentences_nest[i])):
            if j%100==0:
                print(j)
            char_id=idx_switch(char_sentences_nest[i][j],char_tag,char_tag_id)+([0] * (maxlenth_word-len(char_sentences_nest[i][j])))
            char_sentences_id[j]=char_id

        #padding
        char_id_padding=char_sentences_id+[[0] * maxlenth_word]*(maxlenth_sentence-len(char_sentences_id))

       
        char_idx.append(char_id_padding)

    wp_sentence=pd.DataFrame({"句子长度":sentences_lenth})
    wp_sentence.to_excel("句子长度.xlsx",index=False)
    return input_idx,token_type_idx,attention_mask_idx,pos_idx,char_idx,label_idx,sentences_lenth

def create_model():


    HIDDEN_SIZE = 64
    MAX_LEN = 300
    CLASS_NUMS = 3
    POS_SIZE=18
    word_maxlen=64
    char_maxlen=101

    # Parameter Setting
    configuration = BertConfig.from_pretrained('bert-base-cased', output_attentions=True, output_hidden_states=True,
                                               use_cache=True, return_dict=True, position_embedding_type='relative_key_query')
    encoding = TFBertModel.from_pretrained('bert-base-cased',config=configuration)
    print(encoding.config)

    
    input_ids = Input(shape=(MAX_LEN,), dtype='int32', name="input_ids")
    token_type_ids = Input(shape=(MAX_LEN,), dtype='int32', name="token_type_ids")
    attention_mask = Input(shape=(MAX_LEN,), dtype='int32', name="attention_mask")

    
    Bert = encoding(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    print(len(Bert))

    # the last hidden layer
    Bert_Last_Hidden = Bert[0]
    print('the last layer:', Bert_Last_Hidden)

    
    First_token = Bert[1]
    print('First_token:', First_token)
    print("...................................................")
    
    # all hidden layers
    Hidden_layer = Bert[2]
    
    for i in range(len(Hidden_layer)):
        if i == 0:
            print("Embedding layer:", Hidden_layer[i])
        else:
            print("layer", i, Hidden_layer[i])
    print("...................................................")
    
    Hidden_layer_12 = Hidden_layer[12]
    Hidden_layer_11 = Hidden_layer[11]
    Hidden_layer_10 = Hidden_layer[10]
    Hidden_layer_9 = Hidden_layer[9]

    print(Hidden_layer_12)
    print(Hidden_layer_11)
    print(Hidden_layer_10)
    print(Hidden_layer_9)

    # concatenate the last four hidden layers
    Bert_Last_Four_Hidden = concat([Hidden_layer_9, Hidden_layer_10, Hidden_layer_11, Hidden_layer_12], axis=-1)
    print(Bert_Last_Four_Hidden)

    
    #pos_Embedding
    input_pos = Input(shape=(None,), dtype='int32', name="input_pos")
    pos_embedding = Embedding(input_dim=POS_SIZE, output_dim=POS_SIZE, dtype='float32', name='POS_Embedding')(input_pos)

    #char_embedding
    input_char = Input(shape=(None, word_maxlen), dtype='float32')
    
    
    char_embedding = TimeDistributed(Embedding(input_dim=char_maxlen, output_dim=64,embeddings_initializer=
                                     RandomUniform(minval=-0.5, maxval=0.5)))(input_char)

    char_lstm = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False, return_state=False)),name='char_LSTM')(char_embedding)

    #concatenate Bert_Embedding,POS_Embedding, and char_Embedding
    Concatenated_Embedding = concat([Bert_Last_Four_Hidden, char_lstm,pos_embedding], axis=-1, name='Concatenated_Embedding')

    # BiLSTM layer
    Bilstm_layer = Bidirectional(LSTM(HIDDEN_SIZE,return_sequences=True,dropout=0.50))(Concatenated_Embedding)

   

    Fully_connected_layer = TimeDistributed(Dense(3))(Bilstm_layer)

    # CRF layer
    crf = CRF(CLASS_NUMS, name='crf_layer')
    outputs = crf(Fully_connected_layer)

    
    model = Model(inputs=[input_ids, token_type_ids, attention_mask, input_char,input_pos], outputs=outputs)

    model.summary()
    model.compile(loss=crf.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=crf.accuracy)
    return model


if __name__ == '__main__':

    
    input_url_train = "G:\\Project\\Topic Phrases Extraction\\topic_phrase_data\\final result\\train_data.csv"
    train_data=file_read(input_url_train)
    words_sequences=train_data[0]
    labels_sequences=train_data[1]
    pos_sequences=train_data[2]
    

    
    data_sequence_nest=sentence_segmention(words_sequences,labels_sequences,pos_sequences)
    words_sentences_nest=data_sequence_nest[0]
    labels_sentences_nest=data_sequence_nest[1]
    pos_sentences_nest=data_sequence_nest[2]
    char_sentences_nest=data_sequence_nest[3]

    
    train_data_id=create_inputs(words_sentences_nest,pos_sentences_nest,char_sentences_nest,labels_sentences_nest)

    
    input_idx=train_data_id[0]
    token_type_idx=train_data_id[1]
    attention_mask_idx=train_data_id[2]

    
    pos_idx=train_data_id[3]
    char_idx=train_data_id[4]

    
    print("train_lenth:",len(train_data_id[5]),train_data_id[5])
    label_idx=to_categorical(train_data_id[5],3)

    
    #creat model
    model=create_model()
    
    #training
    model.fit([np.array(input_idx), np.array(token_type_idx), np.array(attention_mask_idx),np.array(char_idx),np.array(pos_idx)], label_idx, epochs=30, verbose=1, batch_size=32)

    print("complete。。。。。。。。。。。。。。。。。。。。。。。")

    
    input_url_test="G:\\Project\\Topic Phrases Extraction\\topic_phrase_data\\final result\\test_data_and_predict_result.xlsx"
    test_data_read=pd.read_excel(input_url_test)
    words_sequences_test = list(test_data_read['paper'])
    labels_sequences_test = list(test_data_read['annotation'])
    pos_sequences_test = list(test_data_read['pos_tag'])

    
    data_sequence_nest_test=sentence_segmention(words_sequences_test,labels_sequences_test,pos_sequences_test)
    words_sentences_nest_test=data_sequence_nest_test[0]
    labels_sentences_nest_test=data_sequence_nest_test[1]
    pos_sentences_nest_test=data_sequence_nest_test[2]
    char_sentences_nest_test=data_sequence_nest_test[3]

    
    test_data_id = create_inputs(words_sentences_nest_test, pos_sentences_nest_test, char_sentences_nest_test, labels_sentences_nest_test)

    
    input_idx_test = test_data_id[0]
    token_type_idx_test = test_data_id[1]
    attention_mask_idx_test = test_data_id[2]

    
    pos_idx_test = test_data_id[3]
    char_idx_test = test_data_id[4]


    #model prediction
    predict_lables_one_hot=model.predict([np.array(input_idx_test),np.array(token_type_idx_test),np.array(attention_mask_idx_test), np.array(char_idx_test),np.array(pos_idx_test)],verbose=1)

    
    sentence_lenth_test=test_data_id[6]
    predict_result_idx=[]
    predict_result_labels=[]
    for i in range(len(predict_lables_one_hot)):
        predict_label_id=np.argmax(predict_lables_one_hot[i],axis=1)
        predict_id=list(predict_label_id[0:sentence_lenth_test[i]])
        predict_id_copy = predict_id.copy()
        for j in range(len(predict_id)):
            if predict_id[j]==0:
                predict_id_copy[j]="O"
            elif predict_id[j]==1:
                predict_id_copy[j] = "B-topic"
            else:
                predict_id_copy[j] = "I-topic"
        predict_result_labels.append(predict_id_copy)
        predict_result_idx.append(predict_id)

    
    print("complete 。。。。。。。。。。。。。。。。。。。。。。。。。。")

    #save the prediction results
    Output_url = "G:\\Project\\Topic Phrases Extraction\\topic_phrase_data\\final result\\predict_result.csv"
    result_DataFrame = pd.DataFrame({"predict_results": list(itertools.chain.from_iterable(predict_result_labels))})
    result_DataFrame.to_csv(Output_url, index=False)


































