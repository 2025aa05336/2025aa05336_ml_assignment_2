Assignment Specification Report: Machine Learning Assignment 2 (Classification & Deployment)

1. Administrative Overview

Birla Institute of Technology & Science (BITS) Pilani Pilani | Dubai | Goa | Hyderabad Work Integrated Learning Programmes Division

Detail	Description
Programme Name	M.Tech (AIML/DSE)
Course	Machine Learning
Assignment Number	Assignment - 2
Total Marks	15
Submission Deadline	15-Feb-2026, 23:59 PM


--------------------------------------------------------------------------------


2. Assignment Purpose and Objectives

This assignment is designed to provide comprehensive exposure to the end-to-end Machine Learning deployment workflow. Students will navigate the full technical lifecycle: from data selection and rigorous model implementation to user interface design and cloud-based deployment. By the conclusion of this task, you will have demonstrated proficiency in translating a static model into a functional, interactive web application.

Primary Tasks:

* Model Implementation: Execute six distinct classification algorithms on a selected dataset.
* Streamlit App Development: Construct a professional interactive interface for model demonstration.
* Cloud Deployment: Host the final application on the Streamlit Community Cloud (Free Tier).
* Evaluation Link Sharing: Submit verified, clickable links for both the source repository and the live application.


--------------------------------------------------------------------------------


3. Infrastructure and Technical Support

All implementation and development tasks for this assignment must be performed using the BITS Virtual Lab.

Technical Support: Should you encounter technical difficulties within the BITS Virtual Lab environment, contact our support lead immediately:

* Email: neha.vinayak@pilani.bits-pilani.ac.in
* Required Subject Line: ML Assignment 2: BITS Lab issue


--------------------------------------------------------------------------------


4. Dataset Selection Criteria

Students are required to select one classification dataset from a public repository (Kaggle or UCI) that adheres to the following mandatory technical constraints:

* Problem Type: Binary or Multi-class classification.
* Source Authority: Must be sourced exclusively from Kaggle or UCI.
* Minimum Feature Size: The dataset must contain at least 12 features.
* Minimum Instance Size: The dataset must contain at least 500 instances.


--------------------------------------------------------------------------------


5. Technical Implementation Requirements

5.1 Machine Learning Models

The following six models must be implemented on the chosen dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbour (kNN) Classifier
4. Naive Bayes Classifier (Gaussian or Multinomial)
5. Ensemble Model: Random Forest
6. Ensemble Model: XGBoost

5.2 Evaluation Metrics

For each of the six models implemented, you must calculate and report the following metrics:

1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC Score)

5.3 Environment Configuration

A requirements.txt file is mandatory for environment reproducibility. Note: Missing dependencies are the #1 cause of deployment failure. Ensure all libraries are explicitly listed.

streamlit
scikit-learn
numpy
pandas
matplotlib
seaborn



--------------------------------------------------------------------------------


6. GitHub Repository and Documentation Structure

6.1 Directory Hierarchy

Your repository must follow this specific structure to facilitate automated checks and deployment:

project-folder/
│-- app.py (or streamlit_app.py)
│-- requirements.txt
│-- README.md
│-- model/ (saved model files/artifacts for all implemented models)


6.2 README Requirements (Reference: Section 3 - Step 5)

The README.md must be professional, structured, and include the following five sections:

* a. Problem statement
* b. Dataset description
* c. Models used (including the metrics comparison table below)
* d. Performance observations (summarized in the performance table below)
* e. Submission links (Direct links to the App and Repository)

6.3 Model Comparison Templates

Table 1: Model Metrics Comparison

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression						
Decision Tree						
kNN						
Naive Bayes						
Random Forest (Ensemble)						
XGBoost (Ensemble)						

Table 2: Performance Observations

ML Model Name	Observation about model performance
Logistic Regression	
Decision Tree	
kNN	
Naive Bayes	
Random Forest (Ensemble)	
XGBoost (Ensemble)	


--------------------------------------------------------------------------------


7. Web Application and Deployment

7.1 Deployment Steps

1. Navigate to https://streamlit.io/cloud.
2. Sign in using your GitHub account.
3. Select "New App".
4. Select the relevant GitHub repository and the deployment branch (usually 'main').
5. Select your app.py file.
6. Click "Deploy".

7.2 Mandatory App Features

Your Streamlit application must include the following four features:

* [ ] Dataset Upload: CSV upload option. Warning: Use test data only (the Streamlit free tier has limited memory capacity).
* [ ] Model Selection: Dropdown menu to toggle between different implemented models.
* [ ] Metric Display: Dynamic display of the evaluation metrics for the selected model.
* [ ] Visual Reports: Generation of a confusion matrix or a detailed classification report.


--------------------------------------------------------------------------------


8. Marking Scheme

The total weightage is 15 marks, distributed as follows:

Category	Component	Marks
Model & GitHub	Total Implementation & Documentation	10
	Dataset description (README)	1
	Implementation of 6 models (1 mark for all metrics per model)	6
	Performance observations (README)	3
Streamlit App	Total Web Application Development	4
	CSV upload functionality	1
	Model selection dropdown	1
	Display of metrics	1
	Confusion matrix / Classification report	1
Evidence	BITS Virtual Lab Execution Screenshot	1

Note: There is no leaderboard for this assignment; model performance will not be compared across the student cohort.


--------------------------------------------------------------------------------


9. Submission Protocols

9.1 Format and Order

Submissions must be a single PDF file. Maintain this exact order:

1. GitHub Repository Link: Must be public and contain source code, requirements.txt, and README.md.
2. Live Streamlit App Link: A clickable, active link to the hosted application.
3. BITS Lab Screenshot: High-resolution proof of execution on the BITS Virtual Lab.
4. README Content: The full text and tables from your README.md must be appended here.

9.2 Policy Constraints

* Taxila Submission: Students must follow the standard Taxila submission process.
* Single Submission Rule: Only ONE submission will be accepted. No resubmission requests will be entertained.
* No Drafts: Submissions in "Draft" status will not be graded. You must finalize and submit.
* Deadline: The deadline of 15-Feb-2026 is absolute. No extensions will be provided.


--------------------------------------------------------------------------------


10. Academic Integrity and Plagiarism Policy

The Work Integrated Learning Programmes Division maintains a zero-tolerance policy toward plagiarism. We will perform the following three levels of verification:

1. Code-Level Checks: Analysis of GitHub commit history, repository structures, and variable naming conventions.
2. UI-Level Checks: Detection of copy-pasted Streamlit templates without original customization.
3. Model-Level Checks: Investigation into identical datasets paired with identical models and identical outputs across different students.

Consequence: Any evidence of plagiarism will result in ZERO (0) marks for the entire assignment—no exceptions. AI tools are permitted for learning support only; direct copy-pasting of AI-generated code as a final submission is prohibited.


--------------------------------------------------------------------------------


11. Final Submission Checklist

Ensure every item below is verified before final submission to Taxila:

* [ ] GitHub repository is public and link is functional.
* [ ] Streamlit app link is live and interactive.
* [ ] App successfully loads and processes test data without errors.
* [ ] All 6 models and 6 metrics are calculated and documented.
* [ ] BITS Virtual Lab execution screenshot is clear and included.
* [ ] The full README.md content is included within the PDF file.
* [ ] The final document is a single PDF following the specified ordering.
