# **Customer Segmentation & Recommendation System**  

This repository contains a **customer segmentation and recommendation system**, built using **RFM analysis, K-Means clustering, and hybrid collaborative-content-based filtering**. The notebook provides **data-driven customer segmentation, personalized product recommendations, and evaluation using Precision@K & Recall@K.**

---

## **1. Setup & Dependencies**  

### **Required Libraries**  
Ensure the following Python libraries are installed before running the notebook:  

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## **2. Execution Steps**  

### **Easiest Way to Run the Code**
The best way to **run, test, visualize results, and interactively explore outputs** is by using a **Jupyter Notebook**. Simply execute the cells sequentially to see insights, visualizations, recommendations, and evaluation metrics.

---

### **Step 1: Load or Generate the Data**  

- If real-world data is available, load it into a Pandas DataFrame (`df`).  
- Otherwise, **run the synthetic dataset generation script** in the notebook to create a simulated dataset.  

#### **Synthetic Dataset Generation**  
Since real-world transactional data is not available, a **synthetic dataset** has been generated to simulate realistic purchase behaviors. The dataset includes:  
- **Customer transactions** with fields:  
  - `Customer ID`  
  - `Product ID`  
  - `Product Category`  
  - `Purchase Amount`  
  - `Purchase Date`  
- **Purchase patterns** were modeled based on realistic assumptions, ensuring diversity in spending habits, category preferences, and recency trends.

---

### **Step 2: Perform Exploratory Data Analysis (EDA)**  
The notebook generates **key visualizations** to analyze:  
- Customer purchase behaviors  
- Popular product categories  
- Purchase trends over time  
- Distribution of RFM features (Recency, Frequency, Monetary)  

---

### **Step 3: Customer Segmentation (Clustering Algorithm)**  
#### **Methodology:**  
1. The notebook applies **RFM-based segmentation** using **K-Means clustering**.  
2. The optimal number of clusters is determined using the **Elbow method & PCA visualization**.  
3. A final segmentation summary is provided, labeling customers into **six segments**.

#### **Code Execution:**  
To generate customer clusters, run the following in the notebook:  

```python
segmenter = CustomerSegmentation(df)
rfm_df = segmenter.preprocess_data()
rfm_df = segmenter.apply_clustering(n_clusters=6)
segmenter.visualize_clusters()  # PCA visualization of clusters
segmenter.plot_category_distribution_by_cluster()  # Cluster-wise category analysis
```

---

### **Step 4: Recommendation System Execution**  

#### **Recommendation Algorithm Summary**
The system uses a **hybrid recommendation approach**, combining:  
1. **Collaborative Filtering (User Segmentation)** → Groups customers based on RFM analysis.  
2. **Content-Based Filtering (TF-IDF Product Similarity)** → Recommends similar products within the user's cluster.  
3. **Diversity Strategy** →  
   - 1 in 10 times, a **random product outside the cluster** is recommended.  
   - 1 in 10 times, a product **from the closest cluster** is introduced.  

#### **Code Execution:**  
To generate recommendations for a specific user, run:  

```python
recommender = RecommendationSystem(df)
recommended_products = recommender.get_cbf_recommendations("C0011", top_n=5)
print(f"🔹 Recommended Products for User C0011: {recommended_products}")
```

Example Output:
```
🔀 Introducing a product from closest cluster 4: P0019
🔹 Recommended Products for User C0011: ['P0046', 'P0041', 'P0022', 'P0019', 'P0049']
```

---

### **Step 5: Model Evaluation (Precision@K & Recall@K)**  

#### **Evaluation Methodology**
1. **Train-Test Split:**  
   - Each user's purchase history is split into **train (80%)** and **test (20%)** to ensure proper evaluation.  

2. **Metrics Used:**  
   - **Precision@K** → Measures the fraction of recommended items that are actually relevant.  
   - **Recall@K** → Measures the proportion of actual purchases that were successfully recommended.  

3. **Evaluation Process:**  
   - Iterate over users in the test set.  
   - Compare model-generated recommendations against actual purchases.  
   - Compute **Precision@K** and **Recall@K** for each user.  
   - Average the results across all users to get overall evaluation scores.  

📌 *Full implementation details are in the notebook under the evaluation section.*  

#### **Code Execution:**  
To evaluate the recommendation system, instantiate the evaluator and compute metrics:  

```python
evaluator = RecommendationEvaluator(recommender, df, test_size=0.2, seed=42)
precision_at_5, recall_at_5 = evaluator.precision_recall_at_k(K=5)

print(f"📊 Precision@5: {precision_at_5:.4f}")
print(f"📊 Recall@5: {recall_at_5:.4f}")
```

---

## **3. Outputs**  

### **1️⃣ Customer Clusters & Visualizations**
- Cluster summaries based on RFM (Recency, Frequency, Monetary).  
- Category preferences of customers within each cluster.  
- PCA-based visualizations of customer segments.

### **2️⃣ Example Product Recommendations**
- Personalized product recommendations for a given user based on purchase history & product similarity.

### **3️⃣ Evaluation Metrics**
- Final **Precision@K & Recall@K** scores measuring recommendation accuracy.

---

## **4. Notes & Considerations**
- The notebook is **fully executable in a Jupyter Notebook environment**.  
- Ensure that the dataset is **formatted correctly before execution**.  
- Modify parameters such as `top_n` (number of recommendations) or `test_size` (train-test split) to fine-tune performance.

---

## **📌 Quick Start**
1. Install dependencies.  
2. Run the notebook cell by cell.  
3. If needed, generate the synthetic dataset by running the dataset creation script.  
4. Generate recommendations using:  
   ```python
   recommender.get_cbf_recommendations("C0011", top_n=5)
   ```
5. Evaluate model performance using:  
   ```python
   evaluator.precision_recall_at_k(K=5)
   ```

