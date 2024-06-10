from sklearn import preprocessing
import pandas as pd

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        '''
        df: pandas dataframe
        categorical_features: list of column names, e.g. ['ord_1', 'nom_0',...]
        handle_na: True/False
        '''
        self.df = df
        self.output_df = self.df.copy(deep=True)
        self.categorical_features = categorical_features
        self.handle_na = handle_na
        self.label_encoders = {}
        self.binary_encoders = {}
        self.enc_type = encoding_type

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
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                new_cols.append(pd.DataFrame(val[:, j], columns=[new_col_name], index=self.df.index))
            self.binary_encoders[c] = lbl
        self.output_df = pd.concat([self.output_df] + new_cols, axis=1)
        return self.output_df
    
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.categorical_features:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-999999")
        

if __name__ == "__main__":
    df = pd.read_csv("input/train_cat.csv")
    cols = [c for c in df.columns if c not in ["id", "target"]]
    cat_feats = CategoricalFeatures(df,
                                    categorical_features=cols,
                                    encoding_type="binary",
                                    handle_na=True)
    output_df = cat_feats.fit_transform()
    print(output_df.head())
