import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_bc_pd(file='breast-cancer.data'):
    np.random.seed(123123)
    # Read stuff
    df = pd.read_csv(file, header=None,
                     names=['target',
                            'age',
                            'menopause',
                            'tumor-size',
                            'inv-nodes',
                            'node-caps',
                            'deg-malig',
                            'breast',
                            'breast-quad',
                            'irradiat'])

    # Encode to number (numerical)
    for col in df.columns:
        le = preprocessing.LabelEncoder().fit(df[col])
        df[col] = le.transform(df[col])

    # Split data and target
    df_data = df.drop(['target'], axis=1)
    y = df.target
    df_train, df_test, y_train, y_test = train_test_split(df_data, y.values, test_size=0.2)
    return df_train, df_test, y_train, y_test


def load_titanic_pd(train_file, test_file, as_category=True):
    # get header
    with open(train_file) as f:
        train_header_names = f.readline().strip().split(",")
    with open(test_file) as f:
        test_header_names = f.readline().strip().split(",")

    df_train = pd.read_csv(train_file, header=0, names=train_header_names)
    df_test = pd.read_csv(test_file, header=0, names=test_header_names)

    df_train = remove_useless_feature(df_train)
    df_test = remove_useless_feature(df_test)

    df_train = df_train.fillna(value="<NULL>")
    df_test = df_test.fillna(value="<NULL>")

    df_train = numerical_df(df_train)
    df_test = numerical_df(df_test)

    df_train = fill_NULL_df(df_train)
    df_test = fill_NULL_df(df_test)

    if as_category:
        df_train = re_numerical_df(df_train)
        df_test = re_numerical_df(df_test)

    print(df_train)
    y_train = df_train.Survived
    df_train = df_train.drop(["Survived"], axis=1)
    return df_train, df_test, y_train, None


def numerical_df(df):
    # Encode to number (numerical)
    nan_bool_map = df.isna()
    for col in df.columns:
        if col in ["Sex", "Embarked"]:
            numer_map = {}
            for i in range(len(df[col])):
                if nan_bool_map[col][i]:  # skip NULL
                    continue
                if df[col][i] not in numer_map.keys():
                    numer_map[df[col][i]] = len(numer_map)
                df[col][i] = numer_map[df[col][i]]
    return df

def remove_useless_feature(df):
    # They are not so related to the survived possibility
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    return df


def get_sample_p(df_col, skip="<NULL>"):
    frequency = {}
    valid_num = 0
    for i in range(len(df_col)):
        if df_col[i] == skip:
            continue
        if df_col[i] not in frequency.keys():
            frequency[df_col[i]] = 1
        else:
            frequency[df_col[i]] += 1
        valid_num += 1

    pre = 0
    for k in frequency.keys():
        frequency[k] /= valid_num + pre
        pre = frequency[k]

    return frequency

def fill_NULL_df(df):
    for col in df.columns:
        if col == "Sex":
            frequency = get_sample_p(df[col])
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    rand = np.random.rand()
                    if rand < frequency[0]:
                        df[col][i] = 0
                    else:
                        df[col][i] = 1
        elif col == "Embarked":
            frequency = get_sample_p(df[col])
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    rand = np.random.rand()
                    if rand < frequency[0]:
                        df[col][i] = 0
                    elif rand < frequency[1]:
                        df[col][i] = 1
                    else:
                        df[col][i] = 2
        elif col == "Fare":
            value = np.mean(df[col][df[col] != "<NULL>"])
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    df[col][i] = value
        else:
            value = np.round(np.mean(df[col][df[col] != "<NULL>"]))
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    df[col][i] = value
    return df


def re_numerical_df(df, feature_list=["Fare", "Age"]):
    for col in feature_list:
        if col == "Fare":
            for i in range(len(df[col])):
                bound = [-1, 10, 20, 30, 1e7]
                for k in range(len(bound)- 1):
                    if df[col][i] >= bound[k] and df[col][i] < bound[k + 1]:
                        df[col][i] = k
                        break
            df[col] = df[col].astype("int")
        elif col == "Age":
            for i in range(len(df[col])):
                bound = [-1, 10, 20, 30, 40, 50, 60, 70, 1e7]
                for k in range(len(bound) - 1):
                    if df[col][i] >= bound[k] and df[col][i] < bound[k + 1]:
                        df[col][i] = k
                        break
        else:
            raise NameError("Only support [Fare, Age]")
    return df

if __name__ == "__main__":
    train_file = "train.csv"
    test_file = "test.csv"

    df_train, y_train, df_test = load_titanic_pd(train_file, test_file)
