# PMLDL Assignment 1: Deployment

This repository contains the solution for **Assignment 1 (Deployment)** in the PMLDL course.  
The task is to deploy a machine learning model via an API and create a web application that interacts with it.

## 📌 Task Description

- Train a regression model on the **California Housing dataset**.  
- Deploy the model with **FastAPI** as a REST API service.  
- Build a web application using **Streamlit**, which interacts with the API.  
- Containerize both API and App with **Docker** and orchestrate with **docker-compose**.  


## ⚙️ Technologies

- Python 3.11  
- scikit-learn (Gradient Boosting Regressor)  
- FastAPI  
- Streamlit  
- Docker, docker-compose  


## 📂 Repository Structure

```

├── code
│   ├── datasets/             # (not required in this project, placeholder)
│   ├── deployment
│   │   ├── api/              # FastAPI service
│   │   │   ├── api.py
│   │   │   └── Dockerfile
│   │   └── app/              # Streamlit web app
│   │       ├── app.py
│   │       └── Dockerfile
│   └── models/
│       └── train\_model.py    # training script
├── data/                     # (placeholder for datasets if needed)
├── models/                   # trained model saved here (california\_gb.joblib)
└── docker-compose.yml

````

## 🚀 How to Run

### Clone repository
```bash
git clone https://github.com/<your-username>/PMLDL-Assignment-1.git
cd PMLDL-Assignment-1/code/deployment
````

### Train the model (optionally, because the .joblib file is already in repo)

```bash
cd ../models
python train_model.py
```

This will save the trained model into the `models/` directory.

### Build and run containers

```bash
cd ../deployment
docker-compose up --build
```

* FastAPI API will run at [http://localhost:8000](http://localhost:8000)
* Streamlit app will run at [http://localhost:8501](http://localhost:8501)

## 🌟 Features

* Gradient Boosting model trained on **California Housing dataset**
* Interactive web UI with sliders and number inputs for feature values
* Visualization of feature distributions with user input highlight
* Warning messages when inputs go out of realistic ranges

