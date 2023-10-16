import pandas as pd 
from flair.data import Sentence
from flair.models import SequenceTagger


def validation(test_data,model):
    '''
    Applies the model in the variable model to the test data in the variable test_data.
    Returns the result in the expected format in the file submission.csv.
    '''
    df = pd.read_csv(test_data)
    dfTest = df.copy()
    lSent = []
    lTag = ["aucun"]
    lPred = []

    tagger = SequenceTagger.load(model)
    for index,row in dfTest.iterrows():
    #Add each word from a sentence to a list until you reach a dot.
    #When a dot is reached, we transform this list into a flair phrase, 
    #which we apply to the model.
    #We store the tags of each sentence in the variable lPred.
        word = row["Token"][1:-1]
        lSent.append(word)
        if row["Id"] == 26172 or row["Id"] == "26172":
            word = "."
        if word == ".":
            lTag *= len(lSent)
            print("Index ",index," "," ".join(lSent))
            sentence = Sentence()
            for token in lSent:
                sentence.add_token(token)
            tagger.predict(sentence)
            print(sentence.get_spans('ner'))
            for entity in sentence.get_spans('ner'):
                tag = entity.labels[0].value
                w = entity.position_string
                if "-" in tag:
                    tag = " ".join(tag.split("-"))
                lTag[int(w)-1] = tag
            lPred += lTag
            lSent = []
            lTag = ["aucun"]

    #We retrieve the IDs and associate each ID with its tag obtained by the model.
    df2 = dfTest[["Id"]].copy()
    df2["Label"] = lPred
    df2.to_csv("submission.csv",index=False)

test_data="test.csv"
model = 'best-model.pt'
validation(test_data,model)