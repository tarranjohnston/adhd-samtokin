import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import time
import nltk
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load dataset
file_path = Path(__file__).parent / "dataset" / "charity_emails_v10.csv"
print("Adjusted Path:", file_path)
dfTrain = pd.read_csv(file_path)

train = np.array(dfTrain)
features = (train[:, 0].astype(object) + " " + train[:, 2].astype(object))
labels = train[:, 3]

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in text.split()]

def preprocess(features):
    features = [" ".join(lemmatize_text(f)) for f in features]
    vect = TfidfVectorizer(max_features=2000, stop_words="english")
    vectors_train = vect.fit_transform(features)
    vector_path = Path(__file__).parent / "model" / "tfidf_vectorizer.pkl"
    joblib.dump(vect, vector_path)
    return np.asarray(vectors_train.todense())

def train_and_evaluate(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) 
    start_time = time.time()
    model.fit(X_train, y_train)
    training_runtime = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate AUROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if len(np.unique(y)) == 2:  # Binary classification
            auroc = roc_auc_score(y_test, y_proba[:, 1])
        else:  # Multi-class classification
            auroc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    else:
        auroc = None
    
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate F1 score
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, training_runtime, cm, auroc, f1, y_test, y_pred

# Define models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
    "SVC": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "LogisticRegression": LogisticRegression(),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "RandomForest": RandomForestClassifier(),
    "MLP": MLPClassifier()
}

# Process features
X_processed = preprocess(features)

# Train models and collect results
comparison_data = []
confusion_matrices = {}

# Initialize comparison_df with Actual column (test set labels)
comparison_df = None

for model_name, model in models.items():
    print(f"Training {model_name}...")
    accuracy, runtime, cm, auroc, f1, y_test, y_pred = train_and_evaluate(model, X_processed, labels)
    model_name_path = f"{model_name}_model.pkl"
    model_path = Path(__file__).parent / "model" / model_name_path
    joblib.dump(model, model_path)
    comparison_data.append({"Model": model_name, "Accuracy": accuracy, "Training Time (s)": runtime, "AUROC": auroc, "F1 Score": f1})
    confusion_matrices[model_name] = {"cm": cm, "auroc": auroc, "f1": f1, "accuracy": accuracy, "training_time": runtime}

    # Initialize comparison_df with Actual column after the first model
    if comparison_df is None:
        comparison_df = pd.DataFrame({"Actual": y_test})
        # Add row numbers
        comparison_df["Row"] = range(1, len(y_test) + 1)
    
    # Add model predictions to comparison_df
    comparison_df[model_name] = y_pred

# Reorder columns to make "Row" the first column
comparison_df = comparison_df[["Row", "Actual"] + list(models.keys())]

# Save results
comparison_df.to_csv("comparison_results.csv", index=False)
comparison_metrics_df = pd.DataFrame(comparison_data)
comparison_metrics_df.to_csv("comparison_metrics.csv", index=False)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "ML Classifier Performance"

# Load results
comparison_df = pd.read_csv("comparison_results.csv")
comparison_metrics_df = pd.read_csv("comparison_metrics.csv")

# Modern color scheme
colors = {
    "background": "#F9F9F9",
    "text": "#2C3E50",
    "accent": "#1ABC9C",
    "secondary": "#3498DB",
    "table_header": "#2C3E50",
    "table_cell": "#FFFFFF",
}

# Layout
app.layout = html.Div(
    style={"backgroundColor": colors["background"], "padding": "20px"},
    children=[
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview", style={"color": colors["text"]}),
                *[
                    dcc.Tab(label=name, value=name, style={"color": colors["text"]})
                    for name in models.keys()
                ],
            ],
        ),
        html.Div(id="tabs-content"),
    ],
)

@app.callback(
    Output("performance-table", "data"),
    [Input("performance-table", "sort_by")],
    [State("performance-table", "data")],
)
def sort_table(sort_by, data):
    if sort_by:
        # Convert data to DataFrame for sorting
        df = pd.DataFrame(data)
        # Get the column to sort by
        col = sort_by[0]["column_id"]
        ascending = sort_by[0]["direction"] == "asc"
        # Sort the DataFrame
        if col == "Model":
            df = df.sort_values(by=col, ascending=ascending, key=lambda x: x.str.lower())
        else:
            df = df.sort_values(by=col, ascending=ascending)
        return df.to_dict("records")
    return data

@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def update_tab(selected_tab):
    if selected_tab == "overview":
        # Create graphs for Accuracy, Training Time, AUROC, and F1 Score
        accuracy_fig = px.bar(
            comparison_metrics_df,
            x="Model",
            y="Accuracy",
            title="Model Accuracy",
            labels={"Accuracy": "Accuracy", "Model": "Model"},
            text="Accuracy",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        accuracy_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        accuracy_fig.update_layout(
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font_color=colors["text"],
        )

        training_time_fig = px.bar(
            comparison_metrics_df,
            x="Model",
            y="Training Time (s)",
            title="Model Training Time",
            labels={"Training Time (s)": "Training Time (seconds)", "Model": "Model"},
            text="Training Time (s)",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        training_time_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        training_time_fig.update_layout(
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font_color=colors["text"],
        )

        auroc_fig = px.bar(
            comparison_metrics_df,
            x="Model",
            y="AUROC",
            title="Model AUROC",
            labels={"AUROC": "AUROC", "Model": "Model"},
            text="AUROC",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        auroc_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        auroc_fig.update_layout(
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font_color=colors["text"],
        )

        f1_fig = px.bar(
            comparison_metrics_df,
            x="Model",
            y="F1 Score",
            title="Model F1 Score",
            labels={"F1 Score": "F1 Score", "Model": "Model"},
            text="F1 Score",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        f1_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        f1_fig.update_layout(
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font_color=colors["text"],
        )

        return html.Div(
            style={"backgroundColor": colors["background"]},
            children=[
                html.H3(
                    "Performance Metrics",
                    style={"color": colors["text"], "marginBottom": "20px"},
                ),
                dash_table.DataTable(
                    id="performance-table",
                    columns=[{"name": col, "id": col} for col in comparison_metrics_df.columns],
                    data=comparison_metrics_df.to_dict("records"),
                    style_table={"overflowX": "auto", "width": "100%"},
                    style_cell={
                        "minWidth": "100px",
                        "width": "150px",
                        "maxWidth": "200px",
                        "whiteSpace": "normal",
                        "textAlign": "left",
                        "backgroundColor": colors["table_cell"],
                        "color": colors["text"],
                    },
                    style_header={
                        "backgroundColor": colors["table_header"],
                        "fontWeight": "bold",
                        "color": "white",
                    },
                    sort_action="native",
                    sort_mode="single",
                    page_size=len(comparison_metrics_df),
                ),
                html.Div(
                    style={"marginTop": "30px"},
                    children=[
                        dcc.Graph(figure=accuracy_fig),
                        dcc.Graph(figure=training_time_fig),
                        dcc.Graph(figure=auroc_fig),
                        dcc.Graph(figure=f1_fig),
                    ],
                ),
                html.H3(
                    "Classifier Performance Overview",
                    style={"color": colors["text"], "marginTop": "30px"},
                ),
                dash_table.DataTable(
                    columns=[{"name": col, "id": col} for col in comparison_df.columns],
                    data=comparison_df.to_dict("records"),
                    style_table={"overflowX": "auto", "width": "100%"},
                    style_cell={
                        "minWidth": "100px",
                        "width": "150px",
                        "maxWidth": "200px",
                        "whiteSpace": "normal",
                        "textAlign": "left",
                        "backgroundColor": colors["table_cell"],
                        "color": colors["text"],
                    },
                    style_header={
                        "backgroundColor": colors["table_header"],
                        "fontWeight": "bold",
                        "color": "white",
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": f'{{{col}}} ne {{Actual}}', "column_id": col},
                            "backgroundColor": colors["accent"],
                        }
                        for col in comparison_df.columns
                        if col not in ["Row", "Actual"]
                    ],
                    page_size=len(comparison_df),
                ),
            ],
        )
    else:
        cm = confusion_matrices[selected_tab]["cm"]
        auroc = confusion_matrices[selected_tab]["auroc"]
        f1 = confusion_matrices[selected_tab]["f1"]
        accuracy = confusion_matrices[selected_tab]["accuracy"]
        training_time = confusion_matrices[selected_tab]["training_time"]

        # Confusion Matrix
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
        )
        cm_fig.update_layout(
            plot_bgcolor=colors["background"],
            paper_bgcolor=colors["background"],
            font_color=colors["text"],
        )

        return html.Div(
            style={"backgroundColor": colors["background"]},
            children=[
                html.H3(
                    f"Performance of {selected_tab}",
                    style={"color": colors["text"], "marginBottom": "20px"},
                ),
                html.P(
                    f"Accuracy: {accuracy:.4f}",
                    style={"color": colors["text"]},
                ),
                html.P(
                    f"Training Time: {training_time:.2f} seconds",
                    style={"color": colors["text"]},
                ),
                html.P(
                    f"AUROC: {auroc:.4f}" if auroc is not None else "AUROC: Not Available",
                    style={"color": colors["text"]},
                ),
                html.P(
                    f"F1 Score: {f1:.4f}",
                    style={"color": colors["text"]},
                ),
                html.Div(
                    style={"display": "flex", "flexWrap": "wrap", "gap": "20px"},
                    children=[
                        html.Div(
                            style={"flex": "1", "minWidth": "400px"},
                            children=[dcc.Graph(figure=cm_fig)],
                        ),
                    ],
                ),
            ],
        )

if __name__ == "__main__":
    app.run_server(debug=False)