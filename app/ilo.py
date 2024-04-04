import itertools
from datasets import Dataset
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json


TOKENIZER = AutoTokenizer.from_pretrained("raruidol/ArgumentMining-EN-ILO-AIF-RoBERTa_L")
MODEL = AutoModelForSequenceClassification.from_pretrained("raruidol/ArgumentMining-EN-ILO-AIF-RoBERTa_L")


def preprocess_data(filexaif):
    idents_p = []
    idents_l = []
    idents_comb = []
    proploc = {}
    data = {'text': [], 'text2': []}

    for node in filexaif['nodes']:
        if node['type'] == 'I':
            proploc[node['nodeID']] = node['text']
            idents_p.append(node['nodeID'])
        elif node['type'] == 'L':
            proploc[node['nodeID']] = node['text']
            idents_l.append(node['nodeID'])

    for loc in idents_l:
        for prop in idents_p:
            idents_comb.append((loc, prop))
            data['text'].append(proploc[loc])
            data['text2'].append(proploc[prop])

    final_data = Dataset.from_dict(data)

    return final_data, idents_comb, proploc


def tokenize_sequence(samples):
    return TOKENIZER(samples["text"], samples["text2"], padding="max_length", truncation=True)


def make_predictions(trainer, tknz_data):
    predicted_logprobs = trainer.predict(tknz_data)
    predicted_labels = np.argmax(predicted_logprobs.predictions, axis=-1)

    return predicted_labels


def output_xaif(idents, labels, fileaif):
    newnodeId = 90000
    newedgeId = 80000
    for i in range(len(labels)):
        lb = labels[i]

        # MAP = {'None': 0, 'Asserting': 1, 'Pure Questioning': 2, 'Rhetorical Questioning': 3,
        # 'Assertive Questioning': 4, 'Agreeing': 5}

        if lb == 0:
            continue

        elif lb == 1:
            # Add the Asserting node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Asserting", "type": "YA", "timestamp": "", "scheme": "Asserting", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 2:
            # Add the Pure Questioning node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Pure Questioning", "type": "YA", "timestamp": "", "scheme": "Pure Questioning", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 3:
            # Add the Rethorical Questioning node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Rethorical Questioning", "type": "YA", "timestamp": "",
                                     "scheme": "Rethorical Questioning", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 4:
            # Add the Assertive Questioning node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Assertive Questioning", "type": "YA", "timestamp": "",
                                     "scheme": "Assertive Questioning", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 5:
            # Add the Agreeing node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Agreeing", "type": "YA", "timestamp": "", "scheme": "Agreeing", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 999:
            # Add the Challenging node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Challenging", "type": "YA", "timestamp": "",
                                     "scheme": "Challenging", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 7:
            # Add the Default Illocuting node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Default Illocuting", "type": "YA", "timestamp": "", "scheme": "Default Illocuting", "schemeID": "0"})

            # Add the edges from ident[0] to MA and from MA to ident[1]
            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 8:
            # Add the Disagreeing node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Disagreeing", "type": "YA", "timestamp": "", "scheme": "Disagreeing", "schemeID": "0"})

            # Add the edges from ident[0] to MA and from MA to ident[1]
            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 9:
            # Add the Arguing node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Arguing", "type": "YA", "timestamp": "", "scheme": "Arguing", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

        elif lb == 10:
            # Add the Restating node
            fileaif["nodes"].append({"nodeID": newnodeId, "text": "Restating", "type": "YA", "timestamp": "", "scheme": "Restating", "schemeID": "0"})

            sc = idents[i][0]
            ds = idents[i][1]
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": sc, "toID": newnodeId})
            newedgeId += 1
            fileaif["edges"].append({"edgeID": newedgeId, "fromID": newnodeId, "toID": ds})
            newedgeId += 1
            newnodeId += 1

    return fileaif


def illocution_identification(xaif):

    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    dataset, ids, props = preprocess_data(xaif)

    # Tokenize the Dataset.
    tokenized_data = dataset.map(tokenize_sequence, batched=True)

    # Instantiate HF Trainer for predicting.
    trainer = Trainer(MODEL)

    # Predict the list of labels for all the pairs of "L/I" nodes.
    labels = make_predictions(trainer, tokenized_data)

    # Prepare the xAIF output file.
    out_xaif = output_xaif(ids, labels, xaif)

    return out_xaif


if __name__ == "__main__":
    ff = open('', 'r')
    content = json.load(ff)
    print(content)
    out = illocution_identification(content)
    with open("", "w") as outfile:
        json.dump(out, outfile, indent=4)
