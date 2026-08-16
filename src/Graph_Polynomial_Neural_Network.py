# =========================================
# Graph_Polynomial_Neural_Network.py
# Python 3.12 + TensorFlow 2.16 + Keras 3
# =========================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.callbacks import EarlyStopping


class GraphPolynomialNN:

    # -------------------------------------
    # Initialize
    # -------------------------------------
    def __init__(self, input_dim, model_path="graph_poly_model.keras"):

        self.input_dim = input_dim
        self.model_path = model_path
        self.train_required = False

        if os.path.exists(self.model_path):
            print("Loading existing model...")
            self.model = load_model(self.model_path, compile=False)
        else:
            print("Building new model...")
            self.model = self.build_model()
            self.train_required = True


    # -------------------------------------
    # Graph Operation (simple adjacency mix)
    # X = X + A*X approximation
    # -------------------------------------
    def graph_operation(self, x):
        return x + 0.1 * x


    # -------------------------------------
    # Polynomial Expansion
    # -------------------------------------
    def polynomial_expansion(self, x):

        x2 = tf.pow(x, 2)
        x3 = tf.pow(x, 3)

        return tf.concat([x, x2, x3], axis=1)


    # -------------------------------------
    # Build Model
    # -------------------------------------
    def build_model(self):

        inp = Input(shape=(self.input_dim,), name="input_layer")

        # Graph operation layer
        graph_layer = Lambda(self.graph_operation, name="graph_layer")(inp)

        # Polynomial expansion
        poly_layer = Lambda(self.polynomial_expansion, name="poly_layer")(graph_layer)

        # Neural network
        x = Dense(256, activation="relu")(poly_layer)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        feature_layer = Dense(32, activation="relu", name="feature_layer")(x)

        output = Dense(1, activation="sigmoid", name="output")(feature_layer)

        model = Model(inputs=inp, outputs=output)

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model


    # -------------------------------------
    # Train Model
    # -------------------------------------
    def train(self, X_train, y_train):

        print("Training model...")

        early_stop = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )

        self.model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        self.model.save(self.model_path)

        print("Model saved:", self.model_path)


    # -------------------------------------
    # Predict
    # -------------------------------------
    def predict(self, X):

        return self.model.predict(X)


    # -------------------------------------
    # Feature Extraction
    # -------------------------------------
    def extract_features(self, X):

        feature_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("feature_layer").output
        )

        return feature_model.predict(X)

