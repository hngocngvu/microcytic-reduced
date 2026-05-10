from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Feature selection: Correlation filter → (MI + LR + RF song song) → Combine (merge từ 3 cái step trước) → Validate (cross val)

class FeatureSelection():
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(X,y, test_size=0.2, random_state=42)
        self.features= self.X_train.columns


    def correlation_filter(self):
        df_train = pd.concat([self.X_train, self.y_train], axis=1)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            df_train.corr(),
            cmap="coolwarm",   # màu
            square=True,
            cbar=True
        )

        plt.title("Correlation Matrix Heatmap")
        plt.savefig("corr_heatmap.png") 
        plt.show()

    def get_features_rf(self, n_estimators= 200, min_samples_leaf=3, max_depth=5):
        rf= RandomForestClassifier(n_estimators=n_estimators, random_state= 42, min_samples_leaf= min_samples_leaf, max_depth=max_depth)
        rf.fit(self.X_train, self.y_train)
        importance= pd.DataFrame({
            'feature': self.features,
            'rf': rf.feature_importances_
        }).sort_values('rf', ascending= False)

        importance= importance.reset_index(drop=True)

        return importance
    

    def get_features_lr(self, C=0.1, penalty='l1', solver='liblinear'):
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)

        # Train model
        lr = LogisticRegression(C=C, penalty=penalty, solver=solver, random_state=42)
        lr.fit(X_scaled, self.y_train)

        # Handle multi-class
        coef = np.abs(lr.coef_)
        importance_values = coef.mean(axis=0)  # average across classes

        importance = pd.DataFrame({
            'feature': self.features,
            'lr': importance_values
        }).sort_values('lr', ascending=False)

        # Reset index
        importance = importance.reset_index(drop=True)

        return importance

    
    def combine_features(self):
        lr_df = self.get_features_lr()
        rf_df = self.get_features_rf()

        merged = lr_df.merge(rf_df, on='feature', how='outer').fillna(0)

        merged['lr'] = merged['lr'].abs()

        for col in ['lr', 'rf']:
            max_val = merged[col].max()
            if max_val != 0:
                merged[col] = merged[col] / max_val
            else:
                merged[col] = 0

        merged['score'] = merged['lr'] + merged['rf']

        return merged.sort_values('score', ascending=False).reset_index(drop=True)
      
    def evaluate_k(self, ranked_features, k_range=range(5, 20)):
        results = []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for k in k_range:
            selected_features = ranked_features['feature'].head(k)

            X_subset = self.X_train[selected_features]

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            scores = cross_val_score(
                rf,
                X_subset,
                self.y_train,
                cv=skf,
                scoring='f1_macro' 
            )

            results.append({
                'k': k,
                'score': scores.mean(),
                'std': scores.std()
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False).reset_index(drop=True)

        # lấy index của best k 
        best_k = int(results_df.loc[results_df['score'].idxmax(), 'k'])

        return results_df, best_k
    
    def plot_k_performance(self, results_df):
        plt.figure()
        plt.plot(results_df['k'], results_df['score'], marker='o')
        plt.xlabel("Number of Features (k)")
        plt.ylabel("CV Accuracy")
        plt.title("Feature Selection Performance")
        plt.grid()
        plt.show()


