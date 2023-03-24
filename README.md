# OUTLIERS
An outlier is a single data point that goes far outside the average value of a group of statistics. Outliers may be exceptions that stand outside individual samples of populations as well. In a more general context, an outlier is an individual that is markedly different from the norm in some respect.

### Titanic Dataset
![image](https://user-images.githubusercontent.com/98966968/227246774-cad4c912-b074-4ba3-8f57-c9f4f0cd02d1.png)

This is the boxplot of the "age" variable :

![image](https://user-images.githubusercontent.com/98966968/227056416-f0c56ec2-e492-4e12-910e-97d68409891f.png)

As seen there are outliers in this dataset.

### How to find outlier thresholds ?
#### Formula:
![image](https://user-images.githubusercontent.com/98966968/227252286-112ca111-0d1f-4e52-9296-6a88b76a553f.png)


![image](https://user-images.githubusercontent.com/98966968/227252909-b867e289-de9a-44cc-9899-777350322145.png)

The formula functionalized and applied.

Now, we can grab the outliers and take a look.

### How to Solve the Outlier Problem?
#### -Trimming 
Outliers can be deleted. We have implemented it but this is not recommended as it will often result in data loss.
#### -Imputation
As with the methods of dealing with missing data, the method of assigning values ​​can be preferred instead of outliers. It is more advantageous than the problems caused by data loss in deletion operations.
The values to be assigned instead of outliers can be representative statistics such as mean, median, mode, or any fixed value.

#### -Data suppression (re-assignment with thresholds)
Data suppression refers to the various methods or restrictions that are applied to ACS estimates to limit the disclosure of information about individual respondents and to reduce the number of estimates with unacceptable levels of statistical reliability.
### Multivariate Outlier Analysis: Local Outlier Factor (LOF)
When we looked at the variables separately, we detected outliers. So, if we look at the variables together, can we get outlier variables? For example, if a person was married 3 times at the age of 18.

Being 18 years old or getting married 3 times are not problems, but being 18 years old and married 3 times can be an outlier.

Applied to the "LocalOutlierFactor" dataset.
If the values close to -1, it indicates that it is INLIER.

### Elbow Method

![image](https://user-images.githubusercontent.com/98966968/227468851-c4b13a46-5d9c-47ce-abf1-233acef86759.png)

A graph was created according to the threshold values, and when we examined the graph, the point where the slope change was the hardest was determined.
The determined slope change point was chosen as threshold.

The individual variables may appear as outliers, but we found outliers depending on the situation between the variables.

Note : If working with tree methods, these values should not be changed.
