from sklearn import preprocessing
import pandas as pd

"""
- Label Encoding
- One Hot Encoding
- Binary Encoding
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        '''
        df: pandas dataframe
        categorical_features: list of column names, e.g. ['ord_1', 'nom_0',...]
        handle_na: True/False
        '''
        self.df = df
        self.categorical_features = categorical_features
        self.handle_na = handle_na
        self.label_encoders = {}
        self.binary_encoders = {}
        self.enc_type = encoding_type
        self.ohe = None

        if self.handle_na:
            for c in self.categorical_features:
                self.df.loc[:, c] = self.df.loc[:, c].fillna(self.df[c].mode()[0])
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.categorical_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
            if self.output_df[c].dtype == 'float64':
                self.output_df[c] = self.output_df[c].astype(int)
        return self.output_df
    
    def _label_binarization(self):
        new_cols = []
        for c in self.categorical_features:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            # val is a numpy array
            val = lbl.transform(self.df[c].values)
            bin_df = pd.DataFrame(val, columns=[f"{c}__bin_{j}" for j in range(val.shape[1])], index=self.df.index)
            new_cols.append(bin_df)
            self.output_df = self.output_df.drop(c, axis=1)
            self.binary_encoders[c] = lbl
        self.output_df = pd.concat([self.output_df] + new_cols, axis=1)
        return self.output_df
    
    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.categorical_features].values)
        self.ohe = ohe
        return ohe.transform(self.df[self.categorical_features].values)
    
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.categorical_features:
                dataframe.loc[:, c] = dataframe.loc[:, c].fillna(dataframe[c].mode()[0])
        
        if self.enc_type == 'label':
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        
        elif self.enc_type == "binary":
            new_cols = []
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                bin_df = pd.DataFrame(val, columns=[f"{c}__bin_{j}" for j in range(val.shape[1])], index=dataframe.index)
                new_cols.append(bin_df)
                dataframe = dataframe.drop(c, axis=1)
            dataframe = pd.concat([dataframe] + new_cols, axis=1)
            return dataframe
        else:
            raise Exception("Encoding type not understood")
        

if __name__ == "__main__":
    from sklearn import linear_model
    df_train = pd.read_csv("input/train_cat.csv")
    df_test = pd.read_csv("input/test_cat.csv")
    sample = pd.read_csv("input/sample_submission_cat.csv")

    train_len = len(df_train)

    df_test["target"] = -1
    df = pd.concat([df_train, df_test])

    cols = [c for c in df_train.columns if c not in ["id", "target"]]
    cat_feats = CategoricalFeatures(df,
                                    categorical_features=cols,
                                    encoding_type="ohe",
                                    handle_na=True)
    df_transformed = cat_feats.fit_transform()
    
    X = df_transformed[:train_len]
    X_test = df_transformed[train_len:]

    clf = linear_model.LogisticRegression()
    clf.fit(X, df_train.target.values)
    preds = clf.predict_proba(X_test)[:, 1]
    
    sample["target"] = preds
    sample.to_csv("input/sample_submission_cat.csv", index=False)
