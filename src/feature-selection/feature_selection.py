from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


class FeatureSelection():

    def __init__(self, X, y, model):
        msss= MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(msss.split(X, y))

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]


        self.X_train, self.X_test, self.y_train, self.y_test= X_train, X_test, y_train, y_test
        self.features= list(self.X_train.columns)
        self.model= model


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
      
    
    def get_features_mutual_info(self):
        mi_scores = np.zeros(len(self.features))

        for col in self.y_train.columns:
            mi = mutual_info_classif(
                self.X_train, self.y_train[col],
                random_state=42, n_neighbors=5
            )
            mi_scores += mi

        mi_scores /= len(self.y_train.columns)

        mi_df = pd.DataFrame({
            "feature": self.features,
            "importance": mi_scores
        })

        mi_df = (
            mi_df
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return mi_df

    def get_features_permutation(self, n_repeats= 30, scoring= "f1_macro"):
        model= clone(self.model)
        model.fit(self.X_train, self.y_train)

        result= permutation_importance(model, self.X_test, self.y_test, n_repeats= n_repeats,
                                       scoring= scoring, random_state= 42, n_jobs= -1)
        
        importance_df= pd.DataFrame({
            "feature": self.features,
            "importance": result.importances_mean,
            "std": result.importances_std
        })

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        return importance_df
    
    def plot_permutation_importance(self, importance_df):
        df= importance_df.copy()
        plt.figure(figsize=(8, 6))

        plt.barh(
            df["feature"],
            df["importance"]
        )

        plt.gca().invert_yaxis()

        plt.xlabel("Permutation Importance")
        plt.ylabel("Feature")
        plt.title("Feature Ranking")

        plt.tight_layout()
        plt.show()


    def plot_elbow_method(self, importance_df):
        df= importance_df.copy()

        df["importance"] = (
            df["importance"]
            .clip(lower=0)
        )

        total = df["importance"].sum()

        if total == 0:
            print("No positive importance values.")
            return

        cumulative = (
            df["importance"]
            .cumsum()
            / total
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, len(df)+1),
            cumulative,
            marker="o"
        )

        plt.xlabel("Number of Features")
        plt.ylabel("Cumulative Importance")
        plt.title("Elbow Analysis")

        plt.grid(True)
        plt.show()



    def evaluate_k(self, ranked_features, k_range=range(5, 20)):

        model= clone(self.model)
        results = []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for k in k_range:
            selected_features = ranked_features['feature'].head(k).tolist()

            X_subset = self.X_train[selected_features]

            scores = cross_val_score(
                model,
                X_subset,
                self.y_train,
                cv=skf,
                scoring='f1_macro',
                n_jobs= -1
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


