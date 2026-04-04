from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

class FeatureSelection():
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(X,y, test_size=0.2, random_state=42)
        self.features= self.X_train.columns


    def get_features(self, n_estimators= 200, min_samples_leaf=3, max_depth=5):
        rf= RandomForestClassifier(n_estimators=n_estimators, random_state= 42, min_samples_leaf= min_samples_leaf, max_depth=max_depth)
        rf.fit(self.X_train, self.y_train)
        importance= pd.DataFrame({
            'feature': self.features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending= False)

        importance= importance.reset_index(drop=True)

        plt.figure(figsize=(8,5))
        plt.plot(importance.index + 1, importance['importance'], marker= 'o')
        plt.xlabel("Feature Rank")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importance Ranking (Random Forest)")
        plt.savefig("feature-importance.png", dpi= 300)

        return importance
    
