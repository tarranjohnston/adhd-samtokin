# Install dependencies:
#   pip install -r requirements.txt

# Run the server:
#   uvicorn main:app --reload

# Check the docs and test the API:
#   http://127.0.0.1:8000/docs

from __future__ import annotations
import numpy as np
import pandas as pd
import nltk
import joblib
import time
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer

############### AI ###############

VECTORIZER_FILENAME = "tfidf.pkl"
MODEL_FILENAME = "model.pkl"

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
vectorizer: TfidfVectorizer | None = None
model: MLPClassifier | None = None

def lemmatize_text(text):
    return [lemmatizer.lemmatize(word) for word in text.split()]

def preprocess(features, train=False):
    # Apply lemmatization
    features = [" ".join(lemmatize_text(f)) for f in features]

    # Vectorizer with stop words
    global vectorizer
    vectors = None
    if train:
        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words="english"
        )
        vectors = vectorizer.fit_transform(features)
        joblib.dump(vectorizer, VECTORIZER_FILENAME)  # Save the vectorizer
        print(f"Vectorizer saved as {VECTORIZER_FILENAME}")
    else:
        if not vectorizer:
            raise RuntimeError("Vectorizer not found. Train the model to create a vectorizer.")
        vectors = vectorizer.transform(features)
    features = np.asarray(vectors.todense())
    return features

# Function for training and testing a single test run of a model
# Also saves the trained model
def single_validate(model, X, y, test_size=0.4, random_state=42) -> tuple[int, int, float, float]:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train the model and record the runtime
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_runtime = end_time - start_time
    
    # Make predictions and compute accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Accuracy on test set: {accuracy:.4f}")
    print(f"Training runtime: {training_runtime:.4f} seconds")

    # Save the model
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved as {MODEL_FILENAME}")
        
    return X_train.shape[0], X_test.shape[0], accuracy, training_runtime

def train_classifier(emails: list[Email]) -> tuple[int, int, float, float]:
    # Create model
    global model
    model = MLPClassifier()

    # Get features and labels
    df = pd.DataFrame.from_records([email.__dict__ for email in emails])
    features = np.array(df['subject'].astype(object) + ' ' + df['body'].astype(object))
    labels = np.array(df['category_id'])
    tfidf_features = preprocess(features, train=True)

    # Train and test the model
    return single_validate(model, tfidf_features, labels) # train and test the model

def predict_category(email: Email) -> int:
    # Get features
    df = pd.DataFrame.from_records([email.__dict__])
    features = np.array(df['subject'].astype(object) + ' ' + df['body'].astype(object))
    tfidf_features = preprocess(features)
    prediction = model.predict(tfidf_features)
    return int(prediction[0])

############### Models ###############

# Define the database URL
DATABASE_URL = "sqlite:///training_data.db"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create a configured session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Category(Base):
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

    def __repr__(self):
        return f"Category(name='{self.name}')"

class Email(Base):
    __tablename__ = 'emails'

    id = Column(Integer, primary_key=True)
    subject = Column(String)
    sender = Column(String)
    body = Column(String)
    category_id = Column(Integer, ForeignKey('categories.id'))
    category = relationship("Category", backref="emails", cascade="all")
    is_validated = Column(Boolean, default=False)   # Whether the category for this email has been manually validated

    def __repr__(self):
        return f"Email(subject='{self.subject}', sender='{self.sender}', category='{self.category}')"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=engine) # TODO: use migrations instead

############### Schemas ###############

# Schemas for /categories/
class CategoryResponse(BaseModel):
    id: int
    name: str

# Schemas for /emails/
class EmailRequest(BaseModel):
    subject: str
    sender: str
    body: str
    category_name: str

class EmailResponse(BaseModel):
    id: int
    subject: str
    sender: str
    body: str
    category_id: int
    is_validated: bool

# Schemas for /classifier/
class EmailClassifierRequest(BaseModel):
    subject: str
    sender: str
    body: str

class EmailClassifierResponse(BaseModel):
    id: int
    category: CategoryResponse

class TrainResponse(BaseModel):
    training_samples: int
    test_samples: int
    accuracy: float
    training_runtime: float

############### API ###############

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global vectorizer
    print("Loading model and vectorizer...")
    try:
        model = joblib.load(MODEL_FILENAME)
        print(f"Model loaded from {MODEL_FILENAME}")
    except:
        model = MLPClassifier()
        print("Model not found. New model created.")
    try:
        vectorizer = joblib.load(VECTORIZER_FILENAME)
        print(f"Vectorizer loaded from {VECTORIZER_FILENAME}")
    except:
        print("Vectorizer not found. Train the model to create a vectorizer.")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/categories/", response_model=list[CategoryResponse])
async def get_all_categories(db: Session = Depends(get_db)) -> list[CategoryResponse]:
    categories = db.query(Category).all()
    return categories

@app.get("/categories/{category_id}", response_model=CategoryResponse)
async def get_category(category_id: int, db: Session = Depends(get_db)) -> CategoryResponse:
    category = db.query(Category).get(category_id)
    if category is None:
        raise HTTPException(status_code=404, detail=f"Category with id {category_id} not found")
    return category

@app.post("/categories/", response_model=list[CategoryResponse])
async def add_categories(category_names: list[str], db: Session = Depends(get_db)) -> list[CategoryResponse]:
    # Add multiple categories to database
    new_categories = [Category(name = category_name) for category_name in category_names]
    db.add_all(new_categories)
    db.commit()
    for category in new_categories:
        db.refresh(category)
    return new_categories

@app.get("/emails/", response_model=list[EmailResponse])
async def get_emails(limit: int = 100, db: Session = Depends(get_db)) -> list[EmailResponse]:
    emails = db.query(Email).limit(limit).all()
    return emails

@app.get("/emails/{email_id}", response_model=EmailResponse)
async def get_email(email_id: int, db: Session = Depends(get_db)) -> EmailResponse:
    email = db.query(Email).get(email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"Email with id {email_id} not found")
    return email

@app.post("/emails/", response_model=list[EmailResponse])
async def add_emails(emails_post: list[EmailRequest], db: Session = Depends(get_db)) -> list[EmailResponse]:
    """Adds multiple pre-categorized emails to database"""

    # Add new categories
    new_categories = {}
    for email_post in emails_post:
        category = db.query(Category).filter_by(name=email_post.category_name).first()
        if category is None and email_post.category_name not in new_categories:
            new_categories[email_post.category_name] = Category(name=email_post.category_name)
    db.add_all(new_categories.values())
    db.commit()

    # Add new emails
    new_emails = []
    for email_post in emails_post:
        category = db.query(Category).filter_by(name=email_post.category_name).first()
        new_email = Email(
            subject=email_post.subject, 
            sender=email_post.sender, 
            body=email_post.body, 
            category=category,
            is_validated=True)
        new_emails.append(new_email)
    db.add_all(new_emails)
    db.commit()
    for email in new_emails:
        db.refresh(email)

    # Return the first 5 emails added
    return new_emails[:min(5, len(new_emails))]

@app.post("/emails/from-csv/", response_model=list[EmailResponse])
async def add_emails_from_csv(filename: str, db: Session = Depends(get_db)) -> list[EmailResponse]:
    """Adds multiple pre-categorized emails from a CSV file to database"""

    # Load CSV file
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Add new categories
    new_categories = {}
    for _, row in df.iterrows():
        category = db.query(Category).filter_by(name=row['category']).first()
        if category is None and row['category'] not in new_categories:
            new_categories[row['category']] = Category(name=row['category'])
    db.add_all(new_categories.values())
    db.commit()

    # Add new emails
    new_emails = []
    for _, row in df.iterrows():
        category = db.query(Category).filter_by(name=row['category']).first()
        new_email = Email(
            subject=row['subject'], 
            sender=row['sender'], 
            body=row['body'], 
            category=category,
            is_validated=True)
        new_emails.append(new_email)
    db.add_all(new_emails)
    db.commit()
    for email in new_emails:
        db.refresh(email)

    # Return the first 5 emails added
    return new_emails[:min(5, len(new_emails))]

@app.post("/classifier/", response_model=EmailClassifierResponse)
async def categorize_email(email_post: EmailClassifierRequest, db: Session = Depends(get_db)) -> EmailClassifierResponse:
    # Check if email is already in database
    email = db.query(Email).filter_by(subject=email_post.subject, sender=email_post.sender, body=email_post.body).first()
    if email is not None:
        return email

    # If not in database, predict email category and add to database
    new_email = Email(subject=email_post.subject, sender=email_post.sender, body=email_post.body)
    predicted_category_id = predict_category(new_email)
    predicted_category = db.query(Category).get(int(predicted_category_id))
    new_email.category = predicted_category
    db.add(new_email)
    db.commit()
    db.refresh(new_email)

    return new_email

@app.get("/classifier/{email_id}", response_model=EmailClassifierResponse)
async def get_email_category(email_id: int, db: Session = Depends(get_db)) -> EmailClassifierResponse:
    """Gets the category of an email from the database"""

    email = db.query(Email).get(email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"Email with id {email_id} not found")
    return email

@app.put("/classifier/{email_id}", response_model=EmailClassifierResponse)
async def confirm_email_category(email_id: int, category_id: int = None, db: Session = Depends(get_db)) -> EmailClassifierResponse:
    """Validates the category of an email from the database. The classifier
    only trains on emails that are marked as validated."""
    
    # Get email
    email = db.query(Email).get(email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"Email with id {email_id} not found")
    
    # Update category if necessary
    if category_id is not None:
        category = db.query(Category).get(category_id)
        if category is None:
            raise HTTPException(status_code=404, detail=f"Category with id {category_id} not found")
        email.category = category

    # Mark email as validated and save
    email.is_validated = True
    db.commit()
    db.refresh(email)
    return email

@app.post("/classifier/train-model/", response_model=TrainResponse)
async def train_model(db: Session = Depends(get_db)) -> TrainResponse:
    """Trains the classifier model on all validated emails in the database,
    then saves the trained model on disk."""
    emails = db.query(Email).filter_by(is_validated=True).all()
    train_samples, test_samples, accuracy, training_runtime = train_classifier(emails) # TODO: Do this asynchronously? or the server might freeze
    train_response = TrainResponse(training_samples=train_samples, test_samples=test_samples, accuracy=accuracy, training_runtime=training_runtime)
    db.commit()
    return train_response