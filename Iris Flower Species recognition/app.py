from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Create a Flask app
app = Flask(__name__)

# Initialize the model
loaded_model = None

# Define a route for the prediction form
@app.route('/')
def prediction_form():
    return render_template('predict.html')

# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        global loaded_model  # Access the global model

        # Check if the model is loaded and fitted
        if loaded_model is None:
            # Load the pre-trained model
            with open("best_model.pkl", "rb") as model_file:
                loaded_model = pickle.load(model_file)

            # Ensure the model is fitted with some training data
            if not loaded_model._estimator_type:
                # Load your dataset and fit the model
                X, y = load_your_data()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Define the hyperparameters to search for using GridSearchCV
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': [0.1, 1, 10],
                    'kernel': ['poly', 'rbf', 'sigmoid']
                }

                # Create a GridSearchCV object with the SVC classifier
                grid_search = GridSearchCV(SVC(), param_grid, cv=5)

                # Fit the GridSearchCV object to the training data
                grid_search.fit(X_train, y_train)

                # Get the best parameters from the grid search
                best_params = grid_search.best_params_
                print("Best Model Parameters:", best_params)

                # Create a model with the best parameters
                loaded_model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])

                # Fit the best model with the training data
                loaded_model.fit(X_train, y_train)

        # Get input data from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Perform predictions using the loaded model
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = loaded_model.predict([features])[0]

        return render_template('predict.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)})


# Load your dataset (replace this with your actual data loading code)
# Load your dataset and extract features (X) and labels (y)
def load_your_data():
    # Load your dataset (replace 'your_dataset.csv' with the actual dataset path)
    data = pd.read_csv('Iris.csv')

    # Assuming that the last column in the dataset is the target variable
    # You may need to adjust this depending on your dataset's structure
    X = data.iloc[:, 1:-1]  # Features
    y = data.iloc[:, -1]   # Labels

    return X, y


if __name__ == '__main__':
    app.run(debug=True)
