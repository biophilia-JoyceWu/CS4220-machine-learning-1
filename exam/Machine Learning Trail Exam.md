# Review and remark on trial exam

## PCA

Given mean-centered data in 3D for which the covariance matrix is given by $C = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 4 \end{pmatrix}$. Also given is a data transformation matrix $R = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \tfrac{1}{2} & -\tfrac{\sqrt{3}}{2} \\ 0 & \tfrac{\sqrt{3}}{2} & \tfrac{1}{2} \end{pmatrix},$ by which we can linearly transform every data vector $\mathbf{x}$ (taken as a column vector) to a new 3D column vector $\mathbf{z}$ through $\mathbf{z} = \mathbf{Rx}$. We note that $\mathbf{R}$ is actually a rotation matrix that rotates in the second and third coordinate. Also note that for its inverse, we have $R^{-1} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \tfrac{1}{2} & \tfrac{\sqrt{3}}{2} \\ 0 & -\tfrac{\sqrt{3}}{2} & \tfrac{1}{2} \end{pmatrix}.$
### 1. principal component
Q: What is the first principal component of the original data for which we have the covariance matrix $\mathbf{C}$?

A: According to the covariance matrix, we choose the maximum variance as our principal component. Therefore, $\begin{pmatrix} 0\\ 0  \\ 1 \end{pmatrix}$.
### 2. covariance of the transformed data
Q: Assume we transform all the data by the transformation matrix $\mathbf{R}$ what does the covariance of the transformed data become?

A: $\mathbf{C'} = \mathbf{RX(RX)}^T = \mathbf{RXX}^T\mathbf{R}^T = \mathbf{RCR}^T = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \tfrac{7}{2} & -\tfrac{\sqrt{3}}{2} \\ 0 & -\tfrac{\sqrt{3}}{2} & \tfrac{5}{2} \end{pmatrix}$
### 3. Principal component of the transformed data
Q: What is the first principal component for the transformed data? [not sure]

A: $\mathbf{R}\begin{pmatrix} 0\\ 0  \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ \tfrac{-\sqrt{3}}{2} \\ \frac{1}{2} \end{pmatrix}$ or $\begin{pmatrix} 0 \\ \tfrac{\sqrt{3}}{2} \\ -\frac{1}{2} \end{pmatrix}$

## 2D Classification
Assume we have 2-dimensional two-class classification problem. The first class is distributed uniformly in a square between $0≤x_1≤2$ and $0≤x_2≤2$. The second class is distributed uniformly in a circle with center $(2,2)$, and radius 1. See picture below:
<img src = "https://img-blog.csdnimg.cn/20210111173129111.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d5dzk3MDYyNA==,size_16,color_FFFFFF,t_70#pic_center" width = "30%">
### 1. Bayes error
Q: Assume that both classes are equally likely. What is the Bayes error for this problem?

A: First question is where is the decision boundary. That is when $p(x|\omega_1)p(\omega_1)><p(x|\omega_2)p(\omega_2)$. Because both classes are equally likely, the prior of both classes are $p(\omega_1)
= p(\omega_2) = \frac{1}{2}$ respectively. The **height of the pdf of class 1**$^{Note}$ is $p(x|\omega_1)=\frac{1}{2*2}=0.25$, and the height of the pdf of class 2 is $\frac{1}{\pi^2}=0.32$. So the overlap region is assigned to class 2. We make an error of class 1 of: $var_{\epsilon_1} = \frac{\textrm{overlapped}}{\textrm{area of square}} = \frac{\frac{1}{4}\pi r^2}{2*2} = 0.196$ Total error $\epsilon = \epsilon_1*\frac{1}{2} + \epsilon_2*\frac{1}{2} = 0.098$

>  **Note**:
> 
>   The area under the graph of a probability density function is 1. The use of 'density' in this term relates to the height of the graph. 
>   The height of the probability density function represents how closely the values of the random variable are packed at places on the x-axis. 
>   [Probability density function (for a continuous random variable)](https://nzmaths.co.nz/category/glossary/probability-density-function-continuous-random-variable)

### 2. Changed prior
Q: Now assume that the prior of class 1 is changed to 0.8. What will be the Bayes error now?

A: Now $p(x|\omega_1)p(\omega_1)=\frac{0.8}{2*2}=0.2$, and $p(x|\omega_2)p(\omega_2)=\frac{0.2}{\pi r^2}=0.06$.
So the overlap region is assigned to class 1.
We make an error of class 2 of: $var_{\epsilon_2} = \frac{\textrm{overlapped}}{\textrm{area of circle}} = \frac{1}{4}$
Total error $\epsilon = \epsilon_1*0.8 + \epsilon_2*0.2 = 0.05$

### 3. Logistic classifier
Q: Assume we fit a logistic classifier: $p(\omega_1|x)=\frac{1}{1+\exp(-w^T x-w_0)}$ on a very large training set. In which direction will 𝐰 point towards?

A: The more you move to the upper right corner of the feature space, the more likely it is to find class 2, and then you will have the lower $p(\omega_1|x)$.
When x gets larger and larger ($x=[4,4]$ for instance) we want that the denominator becomes larger (then $p(\omega_1|x)$ gets smaller). Therefore $-w^Tx-w_0$ should get very large (positive) and **$w^Tx+w_0$ should get very negative** . Because all elements in x are positive, all elements in w should get negative.
So $w = [-1,-1]^T$

### 4. Choose a classifier
Q: Now we have three classifiers available: (1) the nearest mean classifier, (2) the quadratic classifier and (3) the 1-nearest neighbour classifier. What classifier should you choose for (a) very small training set sizes, and for (b) very large training set sizes?

> A:  
> If we have a small number of training samples, we need a very simple, inflexible and stable classifier: the nearest mean. 
> If we havevery large raining samples, we can afford a complex, flexible classifier. 
> The most flexible of all given classifiers is the 1NN.

## Alternative perceptron classifier
Q: Assume we optimise a linear classifier $\hat{y}=sign(\mathbf{w}^T x+w_0)$ by minimising an alternative perceptron loss:$$J(\mathbf{w},w_0)=\sum_{misclassified\,\, \mathbf{x}_i} \sqrt{-y_i (\mathbf{w}^T\mathbf{x}_i + w_0)}$$
We start with initialisation $\mathbf{w}=[1,0]^T$,$w_0=0.01$, and we use a learning rate of $\eta=0.1$.
Given dataset $(\mathbf{x}_1=[0,-1]^T,y_1=-1), (\mathbf{x}_2=[1.5,0]^T,y_2=+1), (\mathbf{x}_3=[0,+1]^T,y_3=+1)$, what are the parameters values after one update step?

A: The (alternative) perceptron should minimise the given loss. This is done by gradient descent: $\mathbf{w}_{\textrm{new}} = \mathbf{w}_{\textrm{old}} -\eta \frac{\partial J}{\partial w}$.So we need the derivative of the loss w.r.t. $\mathbf{w}$ and $w_0$.
If we fill in the derivative: $\frac{\partial J}{\partial \mathbf{w}} = \sum_{misclassified\,\, \mathbf{x}_i} \frac{-\mathbf{x}_i \mathbf{y}_i}{2\sqrt{-y_i (\mathbf{w}^T\mathbf{x}_i + w_0)}}$, 
we need to find which objects are misclassified. Only $x_1$ is misclassified, so $\frac{\partial J}{\partial w} = \frac{1}{2}\frac{[0,-1]}{\sqrt{0.01}} = [0,-5]$
For $w_0$ we get the derivative like $\frac{\partial J}{\partial w_0} = \sum_{misclassified\,\, \mathbf{x}_i} \frac{-\mathbf{y}_i}{2\sqrt{-y_i (\mathbf{w}^T\mathbf{x}_i + w_0)}}=\frac{1}{2}\frac{1}{\sqrt{0.01}}=5$
$\mathbf{w} = \mathbf{w} - \eta[0,-5] = [1,0.5]$
$w_0 = w-\eta * 5 = -0.49$ 

> **Note: [线性分类算法：感知器Perceptron](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-2%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E6%84%9F%E7%9F%A5%E5%99%A8-perceptron-%E4%BB%8B%E7%B4%B9-84d8b809f866)**

## 1D Regression
Given are 5 one-dimensional input data points $\mathbf{X}=(−1,−1,0,1,1)^T$ and their 5 corresponding outputs $\mathbf{Y}=(0,0,1,0,1)^T$.
We are going to have a look at linear regression using polynomial basis functions.
### 1. bias term
Q: Fit a linear function (including the bias term) to this data under the standard least-squares loss. What value does the bias term take on?

A: Add a column vector with ones to the original X to model the intercept, then$X^TX =\begin{bmatrix}
4 & 0 \\
0 & 5 
\end{bmatrix}$, $X^TY = \begin{bmatrix}
1 \\
2
\end{bmatrix}$. $(X^TX)^{-1}X^TY =\begin{bmatrix}
\frac{1}{4}\\
\frac{2}{5}
\end{bmatrix}$. So, the intercept equals $\frac{2}{5}$.

### 2. slope of linear function
Q: Fit a linear function (including the bias term) to this data under the standard least-squares loss. What value does the slope take on (i.e. what is the coefficient for the linear term)?

A: According to the result calculated in the above question, the slope is $\frac{1}{4}$.

### 3. total loss
Q: Let us now fit a parabola, a second-order polynomial, to this data. Again, we use the standard squared loss as our optimality criterion. What total loss (i.e., the loss added over all training data point) does the optimal second-order polynomial attain? (Rather than doing the computations, you may want to have a look at a sketch of the situation.)

A: With three degrees of freedom, the squared loss can fit three points perfectly, if they are not in the same input location. Input -1 occurs twice, but the corresponding output is the same, so this is basically one point that needs to be fitted. And +1 we find two different outputs, so the best we can do there is to go right in between. All in all, we can get 0 error on the left point, 0 error on the right point, and $2*(1/2)^2 = 1/2$ due to the errors in the middle.
So the total loss is $\frac{1}{2}$.

### 4. total loss with higher order polynomial
Q:Again determine the total loss over the training data, but now assume we optimally fitted a third-order polynomial.

A: Since going to a third-order polynomial [or **any higher-order** for that matter] cannot improve the performance of the second-order polynmoial, the total loss remains the same $\frac{1}{2}$.

### 5. MLE
Q: Rather than just fitting a least-squares model, we consider a maximum likelihood solution under an assumed Gaussian noise model. That is, we assume that outputs are obtained as a function $f$ from $x$ plus some fixed-variance, independent Gaussian noise.
If our fit to the 5 data point equals the constant zero function, i.e. $f(x)=0$, what then is the maximum likelihood estimate for the variance of the Gaussian noise?

A: The variance equal $\frac{1}{\textbf{precision}}$ and is simply estimated by the average squared loss achieved on the training data, i.e., (0 + 0 + 1 + 0 + 1)/5 = 2/5.
> **Note:** 
> 
> [Maximum Likelihood Estimation Explained - Normal Distribution](https://towardsdatascience.com/maximum-likelihood-estimation-explained-normal-distribution-6207b322e47f)

## Curves
Assume that we have a two-class classification problem. Each of the classes has a Gaussian distribution in $k$ dimensions: $p(\mathbf{x}|\omega_i) = \mathcal{N}(\mathbf{x};\mu_i,\mathbf{I})$, where $\mathbf{I}$ is a $k \times k$ identity matirx. The means of the two classes are $\mu_1 = [0,0,…,0]^T$ and $\mu_2 = [2,2,…,2]^T$. Per class, we have $n$ objects per class. On this data a nearest mean classifier is trained.
### 1. number of feature
Q: **When the number of features increases**...
A: **The bayes error decreases as well**. Because each of the features contributes a bit to the discrimination between the classes. If we know the distributions perfectly, the class overlap would decrease and decrease.
A: **the true error first decreases, then increases again.**  Because the classifier is trained on some training data, which is finite. So at a certain moment it will suffer from **the curse of dimensionality**, and the performance will deteriorate. The true error will first go down (more useful information) and later goes up (overfitting in a too large feature space).

### 2. feature reduction & influence of the number of feature
Q: Before we train a classifier, we also perform a forward feature selection to reduce the number of features to $m=⌈\frac{k}{2}⌉$. When the number of features increases…
A: the true error first decreases, then increases again. Because feature reduction may be tried to combat the curse of dimensionality, but to **when you push it too far** (you increase the number of features further and further), it will anyway suffer from the curse. Fundamentally nothing has changed; **first the true error goes down, but at a certain moment it will increase again**.


