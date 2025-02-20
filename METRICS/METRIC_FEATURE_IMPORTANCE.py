import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.interpolate import make_interp_spline

# MAIN ------------------------------------------------------------------------

# This script features functions and helpers to aid in feature importance analysis

# Produces feature importances of a given dataset based on a feature/labelset using RandomForestClassifier
# Produces aggregated feature importances by feature and timestep to inspect both the overall and timestep importance
def displayFeatureImportance(features, labels, dataframe):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(features, labels)

    # Extract feature importances
    importances = rf.feature_importances_
    featureNames = dataframe.drop("SedimentaryLabel", axis=1).columns

    # Create a new dataframe with the feature importances
    featureImportances = pd.DataFrame({
        "Feature": featureNames,
        "Importance": importances
    })

    # Aggregated Feature Importance (by each feature)
    featureImportances["BaseFeature"] = featureImportances["Feature"].str.split('_T').str[0]
    baseImportances = featureImportances.groupby("BaseFeature")["Importance"].sum().reset_index()
    baseImportances = baseImportances.sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.barh(baseImportances["BaseFeature"], baseImportances["Importance"])
    plt.xlabel("Importance")
    plt.title("Aggregated Feature Importances (All Timesteps)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Aggregated Feature Importance (by timestep)
    featureImportances["Timestep"] = featureImportances["Feature"].str.split('_T').str[1].astype(float)
    timestepImportances = featureImportances.groupby("Timestep")["Importance"].sum().reset_index()
    timestepImportances = timestepImportances.sort_values(by="Timestep", ascending=True)

    # Plot timestep importance
    x = timestepImportances["Timestep"]
    y = timestepImportances["Importance"]
    # Scale the Timesteps to 300 for a smoother plot
    xScaled = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    # Smooth the Y values on the increased timesteps
    ySmooth = spl(xScaled)

    plt.figure(figsize=(12, 6))
    plt.plot(xScaled, ySmooth, linestyle='-')
    plt.xlabel("Timestep")
    plt.ylabel("Total Feature Importance")
    plt.title("Feature Importance by Timestep")
    plt.grid()
    plt.tight_layout()
    plt.show()