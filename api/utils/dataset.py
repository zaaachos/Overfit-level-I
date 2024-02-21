import numpy as np
import pandas as pd

class ObesityDataset:
    def __init__(self, train_data:pd.DataFrame, test_data:pd.DataFrame, seed:int=42):
        self.train_data = train_data
        self.test_data = test_data
        self.seed = seed
        np.random.seed(self.seed)
    
    def build_data(self, validation_size:float=0.2):
        # firstly, we need to drop the ID column (it's not needed)
        self.train_data.drop('id', axis=1, inplace=True)
        
        # get X_features and Targets
        target = self.train_data.NObeyesdad
        x_features = self.train_data.drop('NObeyesdad', axis=1)
        
        # Label Encode the target variable
        # we will also return this, for decoding the test predictions!
        le = LabelEncoder()
        
        # split first and then apply preprocessing and feature engineering steps
        # to avoid Data Leakage train-test
        if validation_size>0:
            x_train, x_valid, y_train, y_valid = self.make_splits(x_features, target, validation_size)

            # first fit in y_train, and the transform the y_valid
            y_train = le.fit_transform(y_train).astype(np.uint8)
            y_valid = le.transform(y_valid).astype(np.uint8)

            # apply feature engineering
            x_train = self.feature_engineering(x_train)
            x_valid = self.feature_engineering(x_valid)

            # Standarize unbalanced data
#             x_train_scaled, scaler = self.standarize_data(x_train)
#             x_valid_scaled = scaler.transform(x_valid)

        else:
            
            x_valid, y_valid = None, None

            # first fit in y_train, and the transform the y_valid
            y_train = le.fit_transform(target).astype(np.uint8)

            # apply feature engineering
            x_train = self.feature_engineering(x_features)

            # Standarize unbalanced data
#             x_train_scaled, scaler = self.standarize_data(x_train)
        
        # apply all aforementioned steps to test data
        test_ids = self.test_data.id
        test_features = self.test_data.drop('id', axis=1)

        x_test = self.feature_engineering(test_features)
#         x_test = scaler.transform(x_test)
        
        print("\n------------------------------------------------------------------------")
        print(f"Train samples: {len(x_train)} | Train targets: {len(y_train)}")
        print(f"Validation samples: {len(x_valid) if x_valid is not None else 0} | Validation targets: {len(y_valid) if x_valid is not None else 0}")
        print(f"Test samples: {len(x_test)}")
        print("\n------------------------------------------------------------------------")
        
        return x_train, y_train, x_valid, y_valid, x_test, test_ids, le
        
    
    def make_splits(self, x_train:pd.DataFrame, y_train:pd.Series, test_size:float=0.2):
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=test_size,
                                                                random_state=self.seed)
        return x_train, x_valid, y_train, y_valid