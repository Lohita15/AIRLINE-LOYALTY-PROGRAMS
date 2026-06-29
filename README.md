# ✈️ Airline Customer Intelligence Dashboard
Live Dashboard: https://airline-loyality-program.streamlit.app 


## 📌 Overview

This project analyzes airline loyalty data to identify **customers at risk of churn** and provides **actionable business recommendations** through an interactive dashboard.

The solution combines:

* Behavioral analytics
* Machine learning
* Customer segmentation
* A Streamlit-based decision interface

👉 The goal is simple:
**Help business teams identify who needs attention and what to do about it — instantly.**

---

## 🎯 Problem Statement

Airlines face increasing challenges in retaining customers. Not all customers are equally valuable, and not all churn risks require the same response.

This project answers:

> **Which customers are likely to churn, and how should we prioritize retention efforts?**

---

## 🧠 Key Features

### 🔹 1. Churn Prediction Model

* Built using **Random Forest**
* Predicts probability of customer churn
* Tuned to avoid overfitting and unrealistic predictions

---

### 🔹 2. Customer Segmentation

Customers are grouped into 4 actionable segments:

| Segment              | Meaning                            |
| -------------------- | ---------------------------------- |
| High Value – Loyal   | Valuable and engaged               |
| High Value – At Risk | Critical customers likely to churn |
| Low Value – Stable   | Low engagement but stable          |
| Low Value – At Risk  | Low-value churn risk               |

---

### 🔹 3. 🚨 Priority Customers Panel

* Automatically highlights **Top High-Value At-Risk Customers**
* Sorted by churn probability
* Helps teams **focus immediately on the most important users**

---

### 🔹 4. 💡 Action Recommendation Engine

Each customer gets a **clear business action**:

* 🎯 High Value - At Risk → VIP offers, retention campaigns
* 📢 Low Value - At Risk → Discounts, re-engagement
* 🌟 High Value - Loyal → Upsell & premium perks
* 👍 Low Value - Stable → Maintain engagement

---

### 🔹 5. 📊 Interactive Dashboard (Streamlit)

Features:

* Customer search & filtering
* Real-time churn prediction
* Visual insights (flights, distance, points)
* Segment distribution
* Risk prioritization

---

## 🛠️ Tech Stack

* **Python**
* **Pandas / NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**

---

## 📂 Project Structure

```bash
├── app.py                  # Streamlit dashboard
├── customer_features.csv   # Final dataset
├── model.pkl               # Trained model
├── notebook.ipynb          # Data analysis & modeling
└── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Lohita15/AIRLINE-LOYALTY-PROGRAMS.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📊 Key Insights

* ✈️ Flight frequency is the strongest predictor of retention
* 📉 Declining engagement signals early churn risk
* 💰 High-value customers can still churn — and require immediate attention
* 📊 Most customers are low-value → need cost-efficient strategies

---

## 💼 Business Impact

This project enables:

* 🎯 Targeted retention campaigns
* 💰 Better ROI on marketing spend
* 📊 Data-driven decision making
* 🚨 Early identification of revenue risk

---

## 📸 Dashboard Preview

<img width="1830" height="827" alt="ap8" src="https://github.com/user-attachments/assets/ef5220b8-184a-4c7e-8284-ac7df70dd099" />


* Priority Customers Panel
* Customer Overview
* Churn Prediction
* Segment Distribution

---

## 🏆 Conclusion

This project goes beyond prediction — it provides a **complete decision system**:

> **Identify → Prioritize → Act**

Designed for non-technical users, the dashboard ensures that a marketing manager can immediately understand:

* Who needs attention
* Why they are at risk
* What action to take

---

## 👤 Author

**G.Lohita Reddy**
B.E CSE Student-NGIT 

---

## ⭐ If you found this useful

Give this repo a ⭐ and feel free to fork!
