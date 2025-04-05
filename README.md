# Machine Learning. Professional

## 1. Review of basic machine learning concepts using regression as an example

### 1.1. Basic Machine Learning Paradigms

#### 1.1.1. Supervised Learning
The model is trained based on examples (stimulus-response pairs), with the aim of reconstructing/approximating the potential functional relationship between input and output signals.

#### 1.1.2. Unsupervised Learning
The model is trained to discover structure in a certain object space (internal relationships, dependencies, patterns).

#### 1.1.3. Reinforcement Learning
An agent interacts with an environment through trial and error to develop the most optimal behavior for maximizing received rewards. The agent can influence subsequent states of the environment and the rewards received (including delayed rewards) through its actions.

### 1.2. Basic ML Concepts

#### 1.2.1. Definitions
- **Object (example/observation)** — for instance, a house and information about it
- **Object Space** — all possible houses
- **Target Variable (target/answer)** — for example, the price of a house
- **Answer Space** — for example, positive real numbers
- **Features** — characteristics of an object (house area, number of rooms, district, etc.)

#### 1.2.2. Dataset
A set of objects and their corresponding answers. N — dataset size (number of examples).

#### 1.2.3. Model
A function that predicts an answer for any object x: a: X → Y

**Example:** linear model a(x) = w₀ + w₁x₁ + ... + wₙxₙ

#### 1.2.4. Parameters and Hyperparameters
- **Parameters** — what we learn from the data (coefficients of our function)
- **Hyperparameters** — model configuration that we set manually when choosing an algorithm

#### 1.2.5. Loss Function
A function that measures how much the prediction differs from the actual answer:
L(y, a(x))

**Example:** squared difference (for regression): L(y, a(x)) = (y - a(x))²
**Mean Squared Error:** ℒ(a, X) = (1/N) * Σ(yi - a(xi))²

#### 1.2.6. Model Training
1. We have a dataset and a loss function
2. We choose a family of models (e.g., linear regression)
3. **Training** — finding the optimal model (from the chosen family) in terms of the loss function
4. Informally: training — finding the best set of parameters among all possible ones

### 1.3. ML Pipeline
1. Model selection
2. Defining the loss function
3. Training on a training dataset
4. Selecting the best model through validation

### 1.4. ML Tasks

#### 1.4.1. Supervised vs Unsupervised
- **Supervised Learning**: training data contains correct answers
- **Unsupervised Learning**: training data contains only examples, and the algorithm finds dependencies on its own

#### 1.4.2. Classification
The answer y belongs to one of the classes:
- **Binary Classification**
- **Multiclass Classification**

**Examples:**
- Credit Scoring
- Sentiment Analysis
- Spam Classification

#### 1.4.3. Regression
The answer is a real number

**Examples:**
- Predicting Housing Prices
- Predicting Currency Exchange Rates

#### 1.4.4. Clustering
Grouping objects based on their similarity

#### 1.4.5. Anomaly Detection
Finding atypical, standout objects

### 1.5. Types of Features

#### 1.5.1. Numerical Features
**Examples:**
- House Area
- Client's Salary
- Patient's Blood Pressure

#### 1.5.2. Binary Features
Flag variables

**Examples:**
- Whether a house has a garage
- Whether a client has children
- Whether a patient has an allergy to citrus fruits

#### 1.5.3. Categorical Features
Feature values cannot be ordered!

**Examples:**
- District where the house is built
- Client's Citizenship
- Specialist who initially treated the patient

**Encoding methods for categorical features:**
- **One-hot-encoding**: instead of a feature with m values, create m binary features
- **Frequency Encoding**: how often a category appears in the training data
- **Mean Target Encoding**: the average value of the target variable for a category

#### 1.5.4. Ordinal Features
Feature values can be ordered!

**Examples:**
- City Type (small city, large city, metropolis)
- Number of client's children
- Patient's risk factor

### 1.6. Key Takeaways
1. Got acquainted with the course
2. Understood what the course consists of
3. Understood which machine learning paradigms exist and which ones we will study in the course
4. Understood the difference between machine learning and classical programming
5. Understood what the model training process involves
6. Identified what tasks ML solves and what types of features exist


## 2. Gradient descent method

### 2.1 Gradient Descent Lecture

#### 2.1.1 Learning Objectives
After this webinar, you will understand:

##### 2.1.1.1 Supervised Learning:
- Classification
- Regression

##### 2.1.1.2 Unsupervised Learning:
- Clustering
- Dimensionality reduction
- Anomaly detection

#### 2.1.2 Linear Regression Recap

##### 2.1.2.1 Basic Concept
- It's easy to draw exactly one straight line through two points on a plane
- But what about three or more points?

##### 2.1.2.2 The Challenge
When we have multiple points, we need a systematic way to fit a line.

- Assumption: The relationship is linear
- A straight line has the form: y = ω₀ + ω₁x
- Question: How to determine the best values for ω₀ and ω₁?

##### 2.1.2.3 Quality Metric for Line Fitting
We need an indicator to measure how well our line fits the data.

##### 2.1.2.4 First Attempt: Sum of Errors
One approach is to sum up all errors (differences between actual points and predicted values on the line).

**Problem:** Positive and negative errors cancel each other out, potentially giving a misleading zero error even with poor fits.

##### 2.1.2.5 Better Approach: Sum of Squared Errors
**RSS - Residual Sum of Squares**

The sum of squared differences between each point and the corresponding value on the fitted line:

RSS = Σ(y_i - (ω₀ + ω₁x_i))²

We choose ω₀ and ω₁ that minimize RSS!

#### 2.1.3 Finding Minimums: Gradient Descent

##### 2.1.3.1 The Concept
Gradient descent is an optimization algorithm for finding the minimum of a function.

1. Start at a random point
2. Calculate the gradient (direction of steepest increase)
3. Move in the opposite direction (steepest decrease)
4. Repeat until reaching a minimum

##### 2.1.3.2 Application to RSS
To optimize RSS using gradient descent:

RSS(ω₀, ω₁) = Σ(y_i - (ω₀ + ω₁x_i))²

The gradient of RSS:
- ∂RSS/∂ω₀ = -2Σ(y_i - (ω₀ + ω₁x_i))
- ∂RSS/∂ω₁ = -2Σ(y_i - (ω₀ + ω₁x_i))·x_i

Update rules:
- ω₀ = ω₀ - α·∂RSS/∂ω₀
- ω₁ = ω₁ - α·∂RSS/∂ω₁

Where α is the learning rate.

##### 2.1.3.3 Multivariate Gradient Descent
Gradient descent works well even with many features!

For multiple parameters:
- ω = ω - α·∇RSS(ω)

Where ω is the vector of parameters and ∇RSS(ω) is the gradient vector.

#### 2.1.4 Quality Metrics

Several metrics can be used to evaluate regression models:

##### 2.1.4.1 Mean Squared Error (MSE)
MSE = (1/n)·Σ(y_i - ŷ_i)²

##### 2.1.4.2 Mean Absolute Error (MAE)
MAE = (1/n)·Σ|y_i - ŷ_i|

##### 2.1.4.3 Root Mean Squared Error (RMSE)
RMSE = √MSE

##### 2.1.4.4 R² (Coefficient of Determination)
Measures the proportion of variance explained by the model:
R² = 1 - RSS/TSS

Where TSS is the Total Sum of Squares: TSS = Σ(y_i - ȳ)²

#### 2.1.5 Regularization

##### 2.1.5.1 Problem: Outliers and Overfitting
Models can become too complex, fitting noise in the data rather than underlying patterns.

##### 2.1.5.2 Solution: Penalize Excessive Complexity
**Regularization** imposes constraints on the norm of the weight vector.

##### 2.1.5.3 Types of Regularization:

###### 2.1.5.3.1 L2 Regularization (Ridge Regression)
Minimizes: RSS + λ·Σω_j²

###### 2.1.5.3.2 L1 Regularization (Lasso Regression)
Minimizes: RSS + λ·Σ|ω_j|

Where λ is the regularization parameter that controls the strength of the penalty.

##### 2.1.5.4 Effect of Regularization
- Reduces model complexity
- Prevents overfitting
- Improves generalization to new data
- Can handle correlated features (especially Ridge)
- Can perform feature selection (especially Lasso)

#### 2.1.6 Advantages and Disadvantages of Linear Regression

##### 2.1.6.1 Advantages
- Very simple and fast to train
- Interpretable model (!)
- Foundational concept

##### 2.1.6.2 Disadvantages
- Requires linear relationship in data
- Too simplistic for complex patterns
- Highly sensitive to data preprocessing

#### 2.1.7 Conclusion
Gradient descent is a powerful optimization technique for finding the parameters that minimize the error in linear regression models. With regularization, we can control model complexity and prevent overfitting.

## 3. Review of basic classification concepts in practice: EDA, cross-validation, quality metrics

## 4. Decision trees

## 5. Model ensembles

## 6. Gradient boosting

## 7. Support vector machines

## 8. Dimensionality reduction methods

## 9. Unsupervised learning. K-means

## 10. Unsupervised learning. Hierarchical clustering. DB-Scan

## 11. Anomaly detection in data

## 12. Practical session - Building end-to-end pipelines and model serialization

## 13-14. Graph algorithms

## 15. Introduction to neural networks

## 16, 19. PyTorch

## 17. Advanced optimization methods, backpropagation and neural network training

## 18. Combating neural network overfitting, exploding and vanishing gradients

## 20. Convolutional Neural Networks

## 21. Recurrent networks

## 22. Data collection

## 23. Preprocessing and tokenization

## 24. Vector word representations, working with pre-trained embeddings

## 25. Language model concept, RNN for working with text

## 26. Transformer architecture

## 27. Transfer Learning. BERT architecture

## 28. Named Entity Recognition

## 29. Topic modeling

## 30-32. Time series analysis

### 30. Problem statement, simplest methods. ARIMA model

### 31. Feature extraction and application of machine learning models. Automatic forecasting

### 32. Time series clustering (finding related stock quotes)

## 33. Introduction to recommender systems

## 34. Simple recommendation models. Collaborative filtering

## 35. Content filtering, hybrid approaches. Association rules

## 36. Matrix factorization methods

## 37. Practical session on recommender systems

## 38. ML in Apache Spark