# Data Science and Machine Learning Interview Question Bank

A curated set of interview practice questions across statistics, machine learning, deep learning, NLP, MLOps, recommendation systems, and large-scale LLM systems. Answers have been intentionally omitted so the document can be used for self-study, mock interviews, or discussion prompts.

## Table of Contents
- [Statistics & Probability](#statistics-probability)
- [Classical ML, Regression, Trees & Evaluation](#classical-ml-regression-trees-evaluation)
- [Clustering & Unsupervised Learning](#clustering-unsupervised-learning)
- [Optimization, Math & Operations Research](#optimization-math-operations-research)
- [Neural Networks & Deep Learning](#neural-networks-deep-learning)
- [NLP, Transformers, LLMs & Generative AI](#nlp-transformers-llms-generative-ai)
- [Recommendation, Ranking & User Modeling](#recommendation-ranking-user-modeling)
- [Computer Vision, Speech & Multimodal ML](#computer-vision-speech-multimodal-ml)
- [Data Quality, Data Strategy & Preprocessing](#data-quality-data-strategy-preprocessing)
- [MLOps, Production Monitoring & Deployment](#mlops-production-monitoring-deployment)
- [Large-Scale / Distributed LLM Systems](#large-scale-distributed-llm-systems)

## Statistics & Probability

1. Explain the difference between covariance and correlation.
2. Compare Bayesian inference with frequentist inference. How does each framework interpret probability and unknown parameters?
3. For right-skewed and left-skewed distributions, describe the typical ordering of the mean, median, and mode.
4. A dataset contains many highly correlated variables, and your manager asks you to apply principal component analysis (PCA). Would you remove correlated variables before running PCA? Explain your reasoning.
5. A fair coin and an unfair coin with tails on both sides are placed in a bag. You select one coin uniformly at random, flip it five times, and observe five tails. What is the probability that the selected coin is the unfair coin?
6. Let X be a standard normal random variable, X ~ N(0, 1). If you draw one independent value per day, what is the expected number of days until you observe a value greater than 2?
7. Given a biased coin with unknown probability of heads, how can you use it to generate a fair binary outcome?
8. Customer lifetimes are modeled using an exponential distribution with rate parameter lambda. Given lifetime observations for n customers, derive the maximum likelihood estimator for lambda.
9. A product team is considering shipping a new feature. Among 1,000 users, 50 dislike the feature. The team samples five distinct users and ships only if all five like it. What is the probability that the feature ships? How would the analysis generalize if N users dislike the feature, and how would you determine the maximum acceptable N under a desired shipping probability?
10. Suppose X and Y are mean-zero, unit-variance random variables. If the least-squares regression of Y on X without an intercept has slope b, what is the slope of the least-squares regression of X on Y?
11. When is Naive Bayes with Laplace smoothing appropriate? Provide a practical example.
12. Explain the difference between statistical dependence and correlation. Can two variables be dependent but uncorrelated?
13. State the Law of Large Numbers and explain how it is useful in data science.
14. What are common ways to quantify uncertainty in statistical estimation, Bayesian modeling, and machine learning prediction?
15. Data are generated from a degree-5 polynomial with Gaussian noise. You train one polynomial regression model of degree 4 and another of degree 6. Which model is more likely to perform better on test data, and why?
16. For a vector y, is softmax(y) always equal to softmax(cy) for any nonzero scalar c? Explain why or why not.
17. Can a probability density function take values greater than 1? Explain the constraints that a valid density must satisfy.
18. How would you simulate rolls of a fair six-sided die using a U(0, 1) random-number generator, and how would you validate that the simulated rolls are fair?
19. Define Type I and Type II errors. In what situations might one type of error be more costly than the other?
20. Describe the receiver operating characteristic (ROC) curve and explain how it is used to evaluate binary classifiers.
21. What is a confidence interval, and how should a 95% confidence interval be interpreted?
22. What is a p-value, and what are common misconceptions about its interpretation?
23. What is hypothesis testing? Describe the roles of the null hypothesis, alternative hypothesis, significance level, test statistic, and p-value.
24. How does batch normalization work in neural networks, and why can it improve training?
25. How does a Gaussian mixture model perform clustering, and how does it differ from k-means?
26. How are partial dependence plots used to interpret machine learning models?
27. How would you assess whether a dataset follows a Gaussian distribution? How does the approach differ for univariate and multivariate data?
28. When should a t-distribution be used instead of a normal distribution? How does sample size affect the choice between a t-test and a z-test?
29. Which probability distributions are commonly used for count data, binary outcomes, and waiting-time or time-to-event outcomes?
30. For linear regression, how would you test or diagnose the assumptions of linearity, independent errors, homoscedasticity, normally distributed residuals, and absence of multicollinearity?

## Classical ML, Regression, Trees & Evaluation

31. Is an 80/20 train-test split always necessary? How would you choose an appropriate split ratio?
32. Which regression error metric is more robust to outliers: MAE, MSE, or RMSE? Explain your reasoning.
33. A dataset has 50 variables, and eight of them have more than 30% missing values. How would you handle those variables?
34. What happens when standard logistic regression is fit to perfectly linearly separable data? If logistic regression must be retained, how would you address the issue?
35. What sources of randomness are used in a random forest, and why are they beneficial?
36. A company runs an A/B test for a donation feature, but the conversion rate does not increase or the increase is not statistically significant. How would you analyze the result and decide what to do next?
37. What analysis would you conduct to measure the effect on a social-network user when a younger family member joins the platform?
38. When would you prefer logistic regression over a decision tree? Which model is more appropriate for a perfectly linearly separable classification problem?
39. A model has many predictors but weak predictive performance. How would you diagnose and improve it?
40. How would you adapt a pre-trained neural network originally designed for classification so that it performs regression?
41. Given N distinct logistic regression models for binary classification, how can a weighted ensemble of their predictions be represented as an artificial neural network?
42. In a classification problem with 1,000 features, 50 informative features, 50 exact copies of those informative features, and 900 uninformative features, how many features would mutual-information filtering select under reliable thresholds?
43. How would you approach modeling when the number of features is much larger than the number of observations?
44. For a law-enforcement classification system where false accusations are extremely costly but public safety also matters, which evaluation metrics would you prioritize and why?
45. A binary classification dataset is generated from two overlapping normal distributions, and you have infinitely many samples. Can logistic regression achieve 100% training accuracy? Explain.
46. How does increasing k affect the bias-variance tradeoff in k-nearest neighbors?
47. How does gradient boosting work, and why is it often effective?
48. Describe the components of a confusion matrix and define accuracy, precision, recall, specificity, and F1-score.
49. Compare bagging and boosting in ensemble learning.
50. What is feature engineering, and why is it important for tabular prediction tasks using XGBoost?
51. What metrics are commonly used to evaluate binary classification models, and when is each metric most appropriate?
52. How do you perform feature selection in machine learning? Compare filter, wrapper, and embedded methods.
53. How do decision trees handle categorical and numerical features?
54. How would you detect and handle outliers in a dataset?
55. How do you perform feature scaling, and why is it important for some machine learning algorithms?
56. How do you interpret coefficients in a linear regression model?
57. What are the advantages of tree-based machine learning algorithms?
58. Explain the difference between homoscedasticity and heteroscedasticity in regression analysis.
59. What are common pitfalls in feature selection?
60. How can overfitting be controlled when training gradient boosting models?
61. How can random forests handle missing values, and what preprocessing choices might still be needed?
62. What challenges arise when training machine learning models with limited data?
63. How would you implement feature selection using information gain?
64. How do newer evaluation metrics such as BERTScore and MoverScore address limitations of traditional text-generation metrics?

## Clustering & Unsupervised Learning

65. If you had to choose between stochastic gradient descent and batch gradient descent for k-means, which would you choose? Does standard k-means use gradient descent in practice?
66. When is the Expectation-Maximization algorithm useful? Provide several examples.
67. If ground-truth labels are available, how can you evaluate the performance of a clustering algorithm?
68. How do you evaluate clustering performance when ground-truth labels are not available?
69. How do you interpret the results of a hierarchical clustering algorithm?
70. How can the number of clusters be selected for k-means?

## Optimization, Math & Operations Research

71. Why are small learning rates often preferred during model training instead of large values such as 1 or 2?
72. What is the angle between the hour and minute hands of a clock at 3:15?
73. Find a function whose sum with its own derivative is zero.
74. Give an example of a continuous function that can represent a smooth version of a binary state, and explain why it is appropriate.
75. How does the computational complexity of the dot product of two length-N vectors scale with N?
76. A neural network's training and test accuracies converge to approximately the same value. Does this imply success? How would you decide what to do next?
77. You start with $50 and repeatedly gamble on fair coin flips: heads wins $2 and tails loses $1. What is the probability that you eventually run out of money?
78. Which optimization methods are commonly used for minimax problems?
79. Does stochastic gradient descent necessarily decrease the loss at every iteration? Explain.
80. Why can an approximate solution to an optimization problem be acceptable during model training?
81. Why do we often use gradient descent instead of directly finding the global minimum of a high-dimensional loss surface?
82. What role does the Hessian matrix play in optimization, and why is it not commonly used directly for training deep neural networks?
83. If independent noise is added to all input variables in a well-specified linear regression problem, how does this affect the objective function and the estimated relationship?
84. What are common reasons that neural-network training loss may fail to decrease during the first few epochs?
85. What can happen if the momentum parameter in SGD is set extremely close to 1, such as 0.9999?
86. Why does weight decay include a scaling factor, and how does it relate to learning rate and batch size?
87. Is training with very large batch sizes always beneficial? How is batch size related to flat and sharp minima?
88. A model predicts a car's fuel efficiency from attributes such as weight and engine size. Is it appropriate to optimize the car design by choosing input attributes that maximize the model's predicted fuel efficiency? Explain the risks.
89. Compare fine-tuning, parameter-efficient fine-tuning, prompt engineering, and retrieval-augmented generation for adapting large language model applications.
90. How does proximal policy optimization optimize policies in continuous action spaces?
91. What are objective functions and constraints in optimization problems?
92. Compare linear programming and mixed-integer programming.
93. When would you use optimization methods rather than machine learning methods, and when would you combine them?
94. How can operational uncertainty be handled in planning and decision-making systems?

## Neural Networks & Deep Learning

95. What are the main steps involved in few-shot learning or meta-learning?
96. What is greedy layer-wise pretraining, and how does it compare with freezing layers in transfer learning?
97. Why might transfer-learning layers be frozen when adapting transformer models?
98. What happens to dropout during inference, and how does it differ from dropout during training?
99. Why is the variational component important in a variational autoencoder? What would happen if it were removed, and how does this relate to generation versus understanding?
100. Why might a model that predicts a numerical quantity use a sigmoid activation in its final layer?
101. Given that neural-network fundamentals have existed for decades, what factors explain the recent success of deep learning?
102. Which operations are typically most computationally expensive during backpropagation, and why?
103. How can two embedding features of dimensions 1 x N and 1 x M be combined with a scalar feature to produce a classification or regression output?
104. Among tanh, ReLU, sigmoid, and ELU, which activation produces the largest output for x = 2? Explain without explicitly computing all functions.
105. While training a model with ReLU activations, many units stop activating. What steps could you take to address this issue?
106. In stable diffusion, why are hidden states often called latent variables rather than embeddings?
107. If a ReLU is inserted immediately before the sigmoid output in a binary classifier, so that y_hat = sigmoid(ReLU(z)), what problem can occur?
108. A neural network trained on 20 samples converges but has high training loss. Is training the same network on 10,000 examples the correct fix? If not, what should be done first?
109. For a four-layer neural network in a binary classification task, is initializing all weights to 0.5 a good idea? Explain.
110. Does weight sharing generally increase model bias or reduce model variance? Explain.
111. Compare batch normalization and instance normalization. Give an example where instance normalization may be preferred.
112. Explain how backpropagation works in neural networks.
113. What is transfer learning, and how is it applied in deep learning?
114. What role do activation functions play in neural networks?
115. How does the softmax function support multi-class classification?
116. What is data augmentation, and why is it important in training deep learning models?
117. What is early stopping, and how does it help control overfitting in neural-network training?

## NLP, Transformers, LLMs & Generative AI

118. How does a generative text model behave differently during training and inference?
119. What is subword tokenization, why is it often preferred to word-level tokenization, and when might it be less appropriate?
120. How can multimodal large language models align information from different modalities, and how does attention operate across modalities?
121. What is the time complexity of a self-attention layer with respect to sequence length?
122. What is retrieval-augmented generation?
123. What are common limitations of retrieval-augmented generation systems?
124. What are two major technical differences between transformer encoders and decoders?
125. Why have decoder-only models become more common than encoder-only models for many modern language-model applications?
126. Why are encoder-decoder models still useful if decoder-only models can handle many tasks?
127. Why have T5-style encoder-decoder models not generally been scaled to compete directly with the largest GPT-style models?
128. What is the purpose of an embedding layer in a neural network?
129. Explain the bag-of-words representation in natural language processing.
130. How can soft prompt tuning adapt a large language model to a new NLP task without modifying the model's core parameters? What are its benefits and challenges?

## Recommendation, Ranking & User Modeling

131. Compare content-based filtering with collaborative filtering in recommender systems.
132. How would you build a restaurant recommendation system for a travel-review platform?
133. If you were modeling the annual revenue of new short-term rental listings, what features, preprocessing steps, and model types would you consider? Would a neural network be appropriate?
134. Walk through how you would build a model to predict whether a brokerage-app user will churn.
135. How does collaborative filtering work in recommendation systems?
136. Design a personalized news-feed ranking system that can support millions of users and articles while maintaining freshness and relevance.

## Computer Vision, Speech & Multimodal ML

137. Explain focal loss and its application in object detection.
138. What are the main steps in building a machine-learning-based face verification system, and would a CNN be an appropriate model choice?
139. How can transformers be applied to computer vision tasks, such as Vision Transformers?
140. You are building a preterm-birth risk model from 500 ultrasound images, including 175 positive examples. You duplicate all positive examples and then split into train, validation, and test sets. What is problematic about this approach?
141. Why are convolutional neural networks usually preferred over fully connected networks for image recognition, even though fully connected layers can access all pixels? Provide two reasons.
142. How do you perform data augmentation for computer vision tasks?
143. How can convolutional neural networks be used for feature extraction?

## Data Quality, Data Strategy & Preprocessing

144. Why do ensembles often outperform individual models? Can an ensemble perform worse than one of its constituent models? Provide a concrete example.
145. Because labeled data are expensive, how would you reduce the amount of labeled data required? Describe three common industry strategies.
146. What is the difference between a model parameter and a hyperparameter?
147. What happens to the variance of a dataset if every observation is duplicated?
148. Given a large corpus of words, how would you identify synonyms? Describe several possible approaches.
149. What is fuzzy logic?
150. Explain the bias-variance tradeoff and its importance in machine learning.
151. What is the curse of dimensionality, and how does it affect machine learning algorithms?
152. What challenges arise when working with large-scale datasets?
153. How should cross-validation be performed for time series data?
154. How do you implement model stacking in ensemble learning?
155. What challenges arise when training machine learning models with noisy data?
156. How can problem reframing simplify a machine learning task or label definition to improve performance? Provide two examples and discuss potential challenges.
157. What strategies can reduce bias in automated evaluations performed by large language models, and how do they improve reliability?

## MLOps, Production Monitoring & Deployment

158. How do recurrent neural networks differ from transformers? State one similarity and two differences.
159. How would you perform classification when labels are noisy or many labels are incorrect?
160. Training, validation, and test accuracy all exceed 90%, but the model behaves unexpectedly in production. How would you diagnose and correct the issue?
161. What is selection bias, and how can it be avoided or mitigated?
162. In a CNN architecture using convolution, batch normalization, and activation layers, would disabling the convolutional bias terms affect performance? Explain.
163. Describe, at a high level, how a GPT-style decoder-only language model generates coherent text and what input-output structure is used during training.
164. What are the main challenges in deploying machine learning models to production?
165. Design a large-scale image classification system capable of processing and categorizing billions of images efficiently.
166. Design an autonomous-vehicle perception system that processes sensor data in real time and supports safe driving decisions.
167. For a large-scale recommendation system, how would you design a feature store? Address real-time versus batch features, storage choices, and serving architecture.
168. Design a demand-forecasting system for supply-chain management using large volumes of historical sales data.
169. Design a scalable reinforcement-learning system for training autonomous agents in complex simulated environments.
170. Design a large-scale genomic analysis platform that uses machine learning to identify disease markers and predict patient outcomes from DNA sequencing data.
171. Design a real-time speech recognition system that transcribes live audio streams accurately and supports multiple languages.
172. What are the key steps in creating infrastructure for MLOps?
173. When can concept drift occur, and how do you detect and address model drift or concept drift in deployed systems?
174. What strategies help ensure data privacy and security in MLOps pipelines?
175. What strategies support scalability in MLOps processes?
176. What is Infrastructure as Code, and what role does it play in MLOps?
177. How do you detect and address model bias within an MLOps framework?
178. What distinguishes batch inference from real-time inference in machine learning applications?
179. How would you design a system that generates related search suggestions for user queries in a search engine?
180. How would you design a system that returns the top 10 rental listings for a location-based property search?
181. How would you design an algorithm to detect the trigger word 'activate' within a 10-second audio clip?
182. How would you design a question-answering system that extracts answers from a large document collection given a user query?
183. How would you design a machine learning pipeline for a large-scale social-media content recommendation system that handles data freshness, novelty effects in A/B testing, and personalization for over a billion users?
184. How would you design a machine learning system for a food-delivery platform to understand and expand user queries, including a domain-specific knowledge graph, query expansion, ambiguous intent handling, and real-time performance?
185. How would you design a caching system for large language model responses that reduces latency and cost while preserving accuracy and reliability? Address semantic similarity and likely challenges.
186. How would you design a defensive user experience for an LLM-based product to anticipate errors, guide user behavior, prevent misuse, and maintain accessibility, trust, and usability?
187. How would you design an ML system using the cascade pattern to decompose a complex problem into smaller sequential tasks? Discuss advantages, drawbacks, and examples.
188. How would you design a human-in-the-loop system for collecting explicit labels for a supervised learning task while balancing annotation quality, cost, and scalability? How could large language models assist?
189. How would you design and implement a system that uses large language models as evaluators to assess the quality, correctness, and safety of another model's outputs?
190. How would you align LLM evaluators with user-defined criteria so that assessments remain accurate, reliable, and consistent with human judgment?
191. How does Fusion-in-Decoder improve open-domain question-answering systems, and what advantages does it have over other retrieval-based approaches?
192. How do internet-augmented language models use search engines to improve question-answering performance, and what are the key components of such systems?
193. How can machine learning teams implement retrieval-augmented generation with hybrid retrieval methods, and what considerations matter for large-scale retrieval optimization?

## Large-Scale / Distributed LLM Systems

194. You are designing a distributed training system for large language models. How would you implement model parallelism and data parallelism, and what are the trade-offs between them?
195. Explain strong scaling in the context of large language model training in one sentence.
196. A GPU or TPU rated at 500 TFLOPS sustains only about 50 TFLOPS on a large-model kernel. What chip-level factors could explain this tenfold performance gap?
197. Let A be an int8 array of shape [128, 2048] sharded as A[I_XY, J] over Mesh({'X': 2, 'Y': 8, 'Z': 2}), for 32 devices total. How much memory does A use per device, and how much total memory does it use across all devices?
198. For a transformer with hidden size D = 4096, feed-forward width F = 4D, vocabulary size V = 32,000, and depth L = 64, estimate the total number of parameters. What fraction are attention parameters? Assume multi-head attention with int8 key-value tensors and hidden size D split across N heads of size H.
199. Name four methods for improving generation throughput or latency in large language models.
200. How would you interleave local-window attention layers with global attention layers to reduce KV-cache growth at long context lengths? Discuss the pros and cons.
201. How does reducing per-device batch size, especially when scaling with fully sharded data parallelism, affect the communication cost of one-dimensional tensor parallelism compared with data parallelism?
