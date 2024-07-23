#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset
df = pd.read_excel("D:/BankDataset.xlsx", engine='openpyxl')


# In[3]:


# Display the first few rows of the dataset
print("First few rows of the dataset:")
display(df.head())


# In[4]:


# Display column names to confirm 'y' is present
print("\nColumn names in the dataset:")
print(df.columns)


# In[5]:


# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())


# In[6]:


# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])


# In[7]:


# Confirm 'y' is in the DataFrame columns after encoding
if 'y' not in df.columns:
    print("\nError: 'y' column not found in the DataFrame after encoding")
else:
    # Split the data into training and test sets
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)  


# In[9]:


# Make predictions on the test set
y_pred = clf.predict(X_test)


# In[10]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
    


# In[11]:


# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['No', 'Yes'], filled=True, rounded=True)
plt.show()


# In[ ]:




