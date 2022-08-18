import re
import pandas as pd
import spacy
import en_core_web_sm


def DataAnnotation(wiki_articel_path, TopicPhrases_path, Output_path):
    

    wiki_articel_Read = pd.read_excel(wiki_articel_path)

    
    Lead_Section = list(wiki_articel_Read['lead section'])
    

    TopicPhrases_Read = pd.read_excel(TopicPhrases_path, sheet_name='Toponym')
    TopicPhrasesList = list(TopicPhrases_Read['TopicPhrases_lower'])

    
    Annotation = []
    TextList_Origin = []
    TextList = []
    pos_tag = []
    

    # load spacy model
    nlp = spacy.load('en_core_web_sm')
    for text in Lead_Section:
        if Lead_Section.index(text) % 100 == 0:
            print(Lead_Section.index(text))
        doc = nlp(text)

        for token in doc:
            TextList_Origin.append(str(token))
            pos_tag.append(token.pos_)
    for word in TextList_Origin:
        word_lower = word.lower()
        TextList.append(word_lower)

    print(len(TextList))
    print(len(TextList_Origin))
    print(len(pos_tag))
    print(len(pos_tag_id))

    for i in range(len(TextList)):
        Annotation.append("O")

    str_con = " "
    for i in range(0, len(TextList)):
        if i % 1000 == 0:
            print(i)
        has_match = False 
        current_word = TextList[i]
        begin_index = i
        end_index = begin_index
        matching = True
        short_match = False
        while matching:
            matching_label = matching_sub_list(current_word, TopicPhrasesList)
            if end_index < len(TextList) - 1:
                if matching_label[0] and not (matching_label[1]):
                    has_match = True
                    matching = False

                elif not matching_label[0] and matching_label[1]:
                    end_index = end_index + 1
                    next_word = TextList[end_index]
                    current_word = current_word + " " + next_word

                elif matching_label[0] and matching_label[1]:
                    short_match = True
                    short_match_phrase = current_word
                    end_index = end_index + 1
                    next_word = TextList[end_index]
                    current_word = current_word + " " + next_word

                else:
                    if short_match:
                        has_match = True
                        current_word = short_match_phrase
                    matching = False
            else:
                if matching_label[0]:
                    has_match = True
                    matching = False
                else:
                    matching = False

            if has_match:
                for i in range(begin_index, begin_index + len(current_word.split(" "))):
                    if Annotation[i] == "B-topic" or Annotation[i] == "I-topic":
                        continue
                    else:
                        if i == begin_index:
                            Annotation[i] = "B-topic"
                        else:
                            Annotation[i] = "I-topic"

    print(Annotation)
    wp = pd.DataFrame(
        {"Wiki": TextList_Origin, "pos_tag": pos_tag, "annotation": Annotation})
    wp.to_csv(Output_url, index=False)


def matching_sub_list(matching_str, str_list):
    label_whole = False
    label_part = False

    if matching_str in str_list:
        label_whole = True

    for item in str_list:
        if str(item).startswith(matching_str) & (item != matching_str):
            label_part = True

    return label_whole, label_part

if __name__ == '__main__':
    wiki_article_path = "D:\\SCI Paper\\GeoTPE\\Data\\Wikipedia_article.xlsx"
    TopicPhrases_path = "D:\\SCI Paper\\GeoTPE\\Data\\Wikipedia_TopicPhrases.xlsx"
    Output_path = "D:\\SCI Paper\\GeoTPE\\Data\\Data\\Wikipedia_train_data.csv"
    DataAnnotation(wiki_article_url, TopicPhrases_path, Output_path)

    
