from feature_extraction_rgb import feature_extract
#from classification import classify
import pandas as pd
from joblib import load

#dataset = pd.read_csv("../dataset/train.csv")
#dataset = dataset[dataset["classname"].isin(["face_with_mask",
#                                             "face_no_mask"])]

def test(dataset, im):
    data = feature_extract(dataset, im)
    print(dataset)
    print("_____________________")
    print(data)
    print(data.columns)
    model = load("svc_clf.joblib")
    X_test = data.drop(["pic_ref"], axis =1)
    print("_____________________")
    print(data["rate_r"])
    print(X_test)
    prediction = model.predict(X_test)
    return prediction

    #classify(data)
