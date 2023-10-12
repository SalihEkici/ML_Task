import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import streamlit as st

data = pd.read_excel("./Dry_Bean_Dataset.xlsx")
df = pd.DataFrame(data)
print(df.isnull().values.any())
# Print the first few rows of the DataFrame to check if it is 
print(df.head(10))


features = df.columns.to_list()
features.remove("Class")
X = df[features].values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=8000)
lr_model.fit(X_train,y_train)
lr_y_pred = lr_model.predict(X_test)

g_model = GaussianNB(var_smoothing=1e-9)
g_model.fit(X_train,y_train)
g_y_pred = g_model.predict(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
rf_y_pred = rf_model.predict(X_test)

# Add models to an array to loop over them while creating the plots
models = [lr_model, g_model, rf_model]
model_names = ["Logistic Regression", "Gaussian Naive Bayes", "Random Forest"]

# Add the predictions to a list so we can loop over them while plotting the heatmaps
predictions = [lr_y_pred, g_y_pred, rf_y_pred]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for i, model in enumerate(models):
    #select the subplot to use in the sns heatmap
    ax = axes[i]
    #get the predictions of the selected model
    y_pred = predictions[i]
    #get the name of the model to show on the plot
    model_name = model_names[i]
    #get the accuracy of the model to show in the title of the plot
    model_accuracy = round(accuracy_score(y_test,y_pred),2)

    # Calculate the confusion matrix using the selected models predictions
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix as a heatmap using the selected ax and the selected models confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)

    # Set axis labels and title
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f"{model_name} accuracy {model_accuracy}")

    # Add axis labels for the class labels
    tick_labels = np.unique(y_test)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

plt.tight_layout()
plt.show()
