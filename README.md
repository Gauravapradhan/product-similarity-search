# Product Similarity Search Service

## Overview

This project implements a scalable product similarity search service for Amazon Fashion products. The solution uses content-based recommendation techniques to identify similar products based on textual, categorical, and numerical product attributes.

The recommendation engine is exposed through a FastAPI microservice and optimized using dimensionality reduction and FAISS-based vector search.

---

## Problem Statement

Build a product similarity search engine capable of retrieving relevant products from a catalog of approximately 30,000 Amazon Fashion products.

Requirements:

* Perform similarity search based on product attributes
* Expose functionality through a REST API
* Optimize retrieval performance
* Containerize the application
* Provide Kubernetes deployment configuration

---

## Dataset Analysis

Dataset Size:

* ~30,000 Amazon Fashion products
* 33 attributes per product

### Missing Value Analysis

| Feature            | Missing %                        |
| ------------------ | -------------------------------- |
| weight             | ~79% unusable placeholder values |
| colour             | ~80% missing                     |
| seller information | >70% missing                     |
| reviews/offers     | >85% missing                     |

### Observations

* Product title and metadata contained the strongest semantic information.
* Weight contained the placeholder value `999999999` in approximately 79% of records.
* Colour had nearly 80% missing values.
* Several seller-related attributes were sparsely populated.

---

## Feature Engineering

### Features Used

* product_name
* meta_keywords
* brand
* sales_price
* rating

### Features Excluded

#### weight

Excluded because approximately 79% of records contained the placeholder value:

999999999

which does not represent a meaningful product weight.

#### colour

Excluded because approximately 80% of records contained missing values.

---

## Similarity Strategy

The final product representation combines:

### Text Features

Generated using TF-IDF from:

* Product Name
* Meta Keywords

### Brand Features

Brand names are encoded as categorical signals.

### Numerical Features

Normalized:

* Sales Price
* Rating

The resulting features are concatenated into a unified feature vector.

---

## Optimization

### TruncatedSVD

High-dimensional TF-IDF vectors were reduced using TruncatedSVD.

Configuration:

* Components: 256

Benefits:

* Reduced memory footprint
* Faster retrieval
* Noise reduction
* Improved scalability

### FAISS

Facebook AI Similarity Search (FAISS) was used for efficient nearest-neighbor retrieval.

Index:

* IndexFlatIP

Benefits:

* Fast similarity lookup
* Efficient vector search
* Production-grade retrieval system

---

## System Architecture

User Request
↓
FastAPI
↓
Similarity Engine
↓
TF-IDF + Brand + Numerical Features
↓
TruncatedSVD
↓
FAISS Search
↓
Top-K Similar Products

---

## Project Structure

SAP_Assestment/

├── app.py

├── requirements.txt

├── Dockerfile

├── deployment.yaml

├── README.md

├── artifacts/

├── data/

└── src/

```
├── similarity_engine.py

├── build_artifacts.py

├── config.py

└── __init__.py
```

---

## Artifacts

Precomputed artifacts are included in the repository to enable immediate execution.

Artifacts include:

* TF-IDF Vectorizer
* TruncatedSVD Model
* FAISS Index
* Product Lookup Tables
* Vector Representations

---

## API Endpoints

### Health Check

GET /health

Response:

{
"status": "healthy"
}

### Similar Product Search

GET /find_similar_products

Example:

/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5

---

## Running Locally

### Create Virtual Environment

python -m venv venv

source venv/bin/activate

### Install Dependencies

pip install -r requirements.txt

### Generate Artifacts

python src/build_artifacts.py

### Start FastAPI

python app.py

### Open Swagger UI

http://localhost:8000/docs

---

## Docker

### Build Image

docker build -t product-similarity .

### Run Container

docker run -p 8000:8000 product-similarity

### Open Swagger UI

http://localhost:8000/docs

---

## Kubernetes

### Deploy

kubectl apply -f deployment.yaml

### Verify Deployment

kubectl get deployments

### Verify Pods

kubectl get pods

### Verify Services

kubectl get svc

---

## Scalability Considerations

Current optimizations:

* Dimensionality reduction using TruncatedSVD
* FAISS vector search
* Docker containerization
* Kubernetes deployment support

These design choices allow the system to scale beyond the initial dataset size.

---

## Future Improvements

* Sentence Transformer embeddings
* Image-based similarity search
* Hybrid semantic search
* HNSW indexing
* Redis caching
* Incremental index updates
* Online learning pipeline

---

## Technologies Used

* Python
* FastAPI
* Scikit-Learn
* FAISS
* NumPy
* Pandas
* Docker
* Kubernetes

---
