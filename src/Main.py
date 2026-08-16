from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from joblib import load


# Setting up directories
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

RESULTS_DIR='results'
os.makedirs(RESULTS_DIR,exist_ok=True)


accuracy = []
precision = []
recall = []
fscore = []

labels=['Moderate Reduction', 'No Reduction', 'Low Reduction','High Reduction']


def load_dataset():
    """Load the dataset from a CSV file selected via file dialog."""
    filepath = filedialog.askopenfilename(
        initialdir=".",
        title="Select CSV File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if filepath:
        return pd.read_csv(filepath)
    else:
        print("No file selected.")
        return None

def preprocess_data(df, is_train=True,label_encoders=None):
    """Preprocess the dataset."""

    # Convert NaN to string "nan" for object columns
    df[df.select_dtypes(include='object').columns] = \
        df.select_dtypes(include='object').fillna("nan")

    if is_train:
        label_encoders = {}

        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    else:
        if label_encoders is None:
            raise ValueError("label_encoders must be provided for test/inference.")

        for col in df.select_dtypes(include='object').columns:

            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str)

                # -------- SAFE TRANSFORM (NO CRASHES) ----------
                df[col] = df[col].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                # ----------------------------------------------
            else:
                raise ValueError(f"Missing encoder for column: {col}")

    # Fill numeric NaNs
    df = df.fillna(df.mean(numeric_only=True))

    if is_train:
        X = df.drop(columns=['Carbon Emission Reduction Category'], axis=1)
        y = df['Carbon Emission Reduction Category']
        return X, y, label_encoders
    else:
        return df


def perform_eda(X, y):
    
    plt.figure(figsize=(18, 12))
    
    # 1. Countplot of Classification Target
    plt.subplot(1, 3, 1)
    sns.countplot(x=y)
    plt.title("Countplot of Carbon Emission Reduction Category")
    plt.xticks(rotation=45)

    # 2. Correlation Heatmap
    plt.subplot(1, 3, 2)
    corr_df = pd.concat([X, y], axis=1)
    corr = corr_df.select_dtypes(include='number').corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')

    # 3. Scatter: Temperature vs Classification Category
    plt.subplot(1, 3, 3)
    if 'Temperature (°C)' in X.columns:
        plt.scatter(X['Temperature (°C)'], y, alpha=0.5)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Reduction Category')
        plt.title('Temperature vs Reduction Category')
    else:
        plt.text(0.5, 0.5, "No 'Temperature (°C)' in X", ha='center', va='center')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/eda_plots.png')
    plt.show()

def train_test_split_data(X, y,test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,  test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# Dataframes to store results
metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
class_report_df = pd.DataFrame()
class_performance_dfs = {}  # Dictionary to store dataframes for each class

if not os.path.exists('results'):
    os.makedirs('results')
    

def Calculate_Metrics(algorithm, predict, y_test, y_proba=None):
    global metrics_df, class_report_df, class_performance_dfs
    
    categories = labels
    
    # Calculate overall metrics
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    metrics_entry = pd.DataFrame({
        'Algorithm': [algorithm],
        'Accuracy': [a],
        'Precision': [p],
        'Recall': [r],
        'F1-Score': [f]
    })
    metrics_df = pd.concat([metrics_df, metrics_entry], ignore_index=True)
    
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    
    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure() 
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        xticklabels=categories,
        yticklabels=categories,
        cmap="viridis",
        fmt="g",
        cbar=False
    )

    ax.set_ylim([0, len(categories)])

    # --- FIX LABEL OVERLAP ---
    plt.xticks(rotation=35, ha='right')   # Slight angle, no overlap
    plt.yticks(rotation=0)                # Straight, readable
    plt.tight_layout()                    # Auto-fix cutting & spacing
    # -----------------------------------

    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    plt.savefig(
        f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
    
    # ROC Curve (Only if probability scores are available)
    if y_proba is not None:
        y_test_bin = label_binarize(y_test, classes=np.arange(len(categories)))
        n_classes = y_test_bin.shape[1]

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curves
        plt.figure(figsize=(8, 6))
        colors = cycle(['blue', 'red', 'green', 'purple'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Class {categories[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.title(f"{algorithm} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_ROC.png")
        plt.show()


def train_logistic_regression(X_train, y_train, X_test, y_test):
    model_path = os.path.join(MODEL_DIR, 'logistic_regression.joblib')
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        dump(model, model_path)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)   # for ROC
    Calculate_Metrics("LRC Model", y_pred, y_test, y_proba)

def train_XGB_Classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate XGB Classifier model."""
    model_path = os.path.join(MODEL_DIR, 'XGB_Classifier.joblib')
    
    if os.path.exists(model_path):
        xgb = load(model_path)
    else:
        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)
        dump(xgb, model_path)
    
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)
    Calculate_Metrics("XGB Classifier",y_pred,y_test, y_proba)

def train_ExtraTreesClassifier(X_train, y_train, X_test, y_test):
    """Train and evaluate ExtraTreesClassifier model."""
    model_path = os.path.join(MODEL_DIR, 'ExtraTreesClassifier.joblib')
    
    if os.path.exists(model_path):
        etc = load(model_path)
    else:
        etc = ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42)
        etc.fit(X_train, y_train)
        dump(etc, model_path)
    
    y_pred = etc.predict(X_test)
    y_proba = etc.predict_proba(X_test) 
    Calculate_Metrics("ETC Model",y_pred,y_test, y_proba)


def train_NN_DNDT(X_train, y_train, X_test, y_test):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    nn_path = os.path.join(MODEL_DIR, "GraphNN.h5")
    dndt_path = os.path.join(MODEL_DIR, "DNDT.joblib")

    # ------------------------------
    # Scaling
    # ------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ------------------------------
    # Neural Network Feature Extractor
    # ------------------------------
    from Graph_Polynomial_Neural_Network import GraphPolynomialNN

    # ------------------------------
    # DNDT Classifier (ETC inside)
    # ------------------------------
    from DNDT_Classifier import DNDTClassifier
    DNDT = DNDTClassifier(dndt_path,X_train_scaled,X_test_scaled)

    if not os.path.exists(dndt_path):
        GNN = GraphPolynomialNN(input_dim=X_train.shape[1], model_path=nn_path)
        GNN.train(X_train_scaled, y_train)
        X_train_feat = GNN.extract_features(X_train_scaled)
        X_test_feat  = GNN.extract_features(X_test_scaled)
        DNDT.train(X_train, y_train)


    y_pred = DNDT.predict(X_test)
    y_proba = DNDT.predict_proba(X_test)

    # ------------------------------
    # Final Metrics
    # ------------------------------
    Calculate_Metrics("Proposed Neuro-Tree Fusion Model", y_pred, y_test, y_proba)

def predict_with_hybrid_model(test_df, label_encoders):

    # Paths
    etc_model_path = os.path.join(MODEL_DIR, 'DNDT.joblib')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(test_df)


    # Load ETC model
    etc = load(etc_model_path)

    # Predict
    predictions = etc.predict(test_df)

    # Decode predictions to original labels
    label_decoder = label_encoders['Carbon Emission Reduction Category']
    decoded_predictions = label_decoder.inverse_transform(predictions)

    # Combine with original test data
    prediction_df = test_df.copy()
    prediction_df['Predicted Category'] = decoded_predictions

    # Save results
    prediction_df.to_csv(os.path.join(RESULTS_DIR, 'hybrid_model_predictions.csv'), index=False)

    return prediction_df
   


def Upload_Dataset():
    global df
    text.delete('1.0', END)
    df = load_dataset()
    text.insert(END, "Dataset loaded successfully.\n\n")
    text.insert(END,str(df)+"\n\n")

def Preprocess_Dataset():
    global df, X, y, label_encoders
    text.delete('1.0', END)    
    X,y,label_encoders= preprocess_data(df, is_train=True)
    perform_eda(X, y)
    text.insert(END,str(X)+"\n\n")
    text.insert(END, "Preprocessing successfully completed.\n\n")


def Train_Test_Splitting():
    global X, y, X_train, X_test, y_test, y_train
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split_data(X,y)
    
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Dataset Train and Test Split Completed" + "\n")
    text.insert(END, "Total records found in dataset to train: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total records found in dataset to test: " + str(X_test.shape[0]) + "\n")    
   
def existing_classifier1():
    global X_train, X_test, y_test, y_train
    text.delete('1.0', END)
    results = {}
    results ['LRC Model'] = train_logistic_regression(X_train, y_train, X_test, y_test)

def existing_classifier2():
    text.delete('1.0', END)

    global X_train, X_test, y_test, y_train
    results = {}
    results ['XGB Classifier'] = train_XGB_Classifier(X_train, y_train, X_test, y_test)

def existing_classifier3():
    text.delete('1.0', END)

    global X_train, X_test, y_test, y_train
    results = {}
    results ['ETC Model'] = train_ExtraTreesClassifier(X_train, y_train, X_test, y_test)

def proposed_classifier3():
    text.delete('1.0', END)
    global X_train, X_test, y_test, y_train
    train_NN_DNDT(X_train, y_train, X_test, y_test)
      
def Prediction():
    global test_data, df1, prediction_df, label_encoders
    
    text.delete('1.0', END)
    test_data = load_dataset()
    
    df1 = preprocess_data(test_data, is_train=False,label_encoders=label_encoders)

    # Call function to make predictions
    prediction_df = predict_with_hybrid_model(df1, label_encoders)   
    text.insert(END, f'Predicted Outcomes for each row:\n')
    test = prediction_df

    for index, row in test.iterrows():
        outcome = row.iloc[-1]   # last column = predicted outcome
        text.insert(
            END,
            f'Row {index + 1}: {row.to_dict()} - Predicted Outcome: {outcome}\n\n'
        )


import tkinter as tk
from tkinter import messagebox
import redis
import hashlib

# Connect to Redis
def connect_redis():
    return redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Hash password before storing in Redis for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Signup functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before storing
                hashed_password = hash_password(password)

                # Store the user in Redis with multiple field-value pairs
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    messagebox.showerror("Error", "User already exists!")
                else:
                    # Using multiple field-value pairs in hset
                    conn.hset(user_key, "username", username)
                    conn.hset(user_key, "password", hashed_password)
                    conn.hset(user_key, "role", role)
                    messagebox.showinfo("Success", f"{role} Signup Successful!")
                    signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    # Create the signup window
    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    # Username field
    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)
    
    # Password field
    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    # Signup button
    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before checking
                hashed_password = hash_password(password)

                # Check if the user exists in Redis
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    stored_password = conn.hget(user_key, "password")
                    stored_role = conn.hget(user_key, "role")

                    if stored_password == hashed_password and stored_role == role:
                        messagebox.showinfo("Success", f"{role} Login Successful!")
                        login_window.destroy()
                        if role == "Admin":
                            show_admin_buttons()
                        elif role == "User":
                            show_user_buttons()
                    else:
                        messagebox.showerror("Error", "Invalid Credentials!")
                else:
                    messagebox.showerror("Error", "User not found!")
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

def show_admin_buttons():
    clear_buttons()
    tk.Button(main, text="Upload Electricity Dataset", command=Upload_Dataset, font=font1).place(x=100, y=160)
    tk.Button(main, text="Data Preprocessing", command=Preprocess_Dataset, font=font1).place(x=350, y=160)
    tk.Button(main, text="Data Splitting", command=Train_Test_Splitting, font=font1).place(x=580, y=160)
    tk.Button(main, text="Build & Train LRC Model", command=existing_classifier1, font=font1).place(x=760, y=160)
    tk.Button(main, text="Build & Train XGB Model", command=existing_classifier2, font=font1).place(x=100, y=210)
    tk.Button(main, text="Build & Train ETC Model", command=existing_classifier3, font=font1).place(x=380, y=210)
    tk.Button(main, text="Build & Train Neuro-Tree Fusion Model", command=proposed_classifier3, font=font1).place(x=700, y=210)
   
def show_user_buttons():
    clear_buttons()
    tk.Button(main, text="Prediction on Test Data", command=Prediction, font=font1).place(x=650, y=200)

# Clear buttons before adding new ones
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

bg_image = Image.open("background.jpg")  
bg_image = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image)

# === ADD THIS FOR BACKGROUND ===
bg_label = tk.Label(main, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.image = bg_photo      # Prevent garbage collection
# =================================


# Title
font = ('times', 18, 'bold')
title_text = "GreenGrid AI: Classifying Carbon Reduction from Electricity Data using Neural and Ensemble Learning"
title = tk.Label(main, text=title_text, bg='powder blue', fg='black', 
                 font=font, wraplength=screen_width - 200, justify='center')
title.place(relx=0.5, y=20, anchor="n")


# Create text widget and scrollbar
font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=300)
text.config(font=font1)



# Admin and User Buttons
font1 = ('times', 14, 'bold')

tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=25, height=1, bg='salmon').place(x=50, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=25, height=1, bg='salmon').place(x=400, y=100)

admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=25, height=1, bg='navajo white')
admin_button.place(x=750, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=25, height=1, bg='navajo white')
user_button.place(x=1100, y=100)

main.config(bg='plum2')
main.mainloop()
