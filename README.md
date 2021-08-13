# Unsupervisied learning in machine learning

As we all know, machine learning is categorised into two significant spectrums: supervised learning and unsupervised learning. Supervised learning gives you classifying and predicting results by having us provide the machines independent and dependent values. But unsupervised learning gives you more than prediction. They provide pattern and structures for you dataset that you don't know it exists. In this type of algorithm, we do not have labelled data(or the dependent variable is absent), such as clustering data, recommendation systems, etc. Unsupervised Learning provides excellent results as one can deduce many hidden relations between different attributes or features.

# About the dataset
Malls or shopping complexes often indulge in the race to increase their customers and make huge profits. To achieve this task, machine learning is being applied by many stores already. This dataset contains information regarding customers of a mall. The job is to find any correlation among the customers to increase the business for profit enhancement by sending advertisements and marketing to these targeted customers only. This will not only increase sales but also reduces the margin on ads.

This dataset contains informations as follows:

Customer ID - It is the unique ID assigned to the customer

Gender - Gender of the customer

Age - Age of the customer(in years)

Annual Income(k$) - Annual income of the customer in k$

Spending Score - Score assigned to the customer by the mall/shopping complex based on the customer spending nature and behaviour

# Data wrangling
In this step, I dropped the ‘CustomerID’ column,  Checked for any missing value and find out the data type to do any connversion if required. 

# Data exploration and visualization 

First, a distplot was made for the following columns (Age, Annual income and Spending score)
A Distplot or distribution plot, depicts the variation in the data distribution. Seaborn Distplot represents the overall distribution of continuous data variables. It represents the univariate distribution of data i.e. data distribution of a variable against the density distribution.

![image](https://user-images.githubusercontent.com/87844891/129387011-860b62e0-d673-4001-8c98-c8541d5c6620.png)
From the plot we can say that all the columns contains univariated disturbation among them. 

Secondly, we made a gender plot graph for the male and females customers. 
![image](https://user-images.githubusercontent.com/87844891/129387331-c8db6476-84a2-4c95-a645-19859f8bdbd5.png)

Lastly, we made correlation distrubation plots among the columns. This is to computed the relationship between the variables.
Ploting the Relation between Age , Annual Income and Spending Score 
![image](https://user-images.githubusercontent.com/87844891/129387655-e2504379-c8cd-46ae-aef2-570256bced51.png)

Age VS Annual income among the male and female customers
![image](https://user-images.githubusercontent.com/87844891/129387814-4ac08369-58ce-4e8f-a152-57e5c38f59ae.png)

Annual income VS Spending score among the male and female customers
![image](https://user-images.githubusercontent.com/87844891/129387851-7aca5666-573f-40ea-8925-efa8c62a7a2e.png)

# Data transformation
Since we dont have independent value in this learning. All the columns were transformed as X (depedent variable). 

# Clustering using K-means clustering algorithms
1. First segmentation using Age and spending

To implement K-Means clustering, we need to look at the Elbow Method.
The Elbow method is a method of interpretation and validation of consistency within-cluster analysis designed to help to find the appropriate number of clusters in a dataset.I have also used the WCSS algorithms (WCSS is the sum of squared distance between each point and the centroid in a cluster) to identify the number of clusters for this dataset

![image](https://user-images.githubusercontent.com/87844891/129389251-00cc7483-198e-4e64-8730-c5ee2d0e907c.png)

From the elbow method graph we can see that after the range at 4, the curve starts to flat steadily. With this, I can say the number of clusters for this dataset is 4.

Next, training the dataset on the K-means clustering were performed.

![image](https://user-images.githubusercontent.com/87844891/129389444-adfd897b-9744-427a-87ec-9abf48686602.png)

Analyzing the results for Age VS Spending score segmentation

We can see that the mall customers can be broadly grouped into five groups based on their purchases made in the mall.

Cluster 1(red)- All class age that has low spending score
In cluster 1(red-coloured), we can see people from all ages (20 till 70) with low spending scores. These people might be having less income or people who don't like spending. 
The marketing people of the mall should consider spending less interest on this category of people as they don't bring many sales. 

Cluster 2(blue)- Age class from early 20 till 40 that has high spending score
In cluster 2(blue coloured), we can see that people from age 20 till 40 love spending money buying stuff. They might love buying things and using their money more to engage themselves in high social groups. The marketing people must consider this category people more cause good advertisement attracts them and their social group too.

Cluster 3(green)- Age class from early 20 till 40 that has low to average spending score
In cluster 3(blue coloured), we can see that people from age 20 to 40 have an average spending score. These people might have high responsibilities like household and children, making them spend averagely. The marketing people should consider attracting this category by showing more household needs and children centred ads. 

Cluster 4(cyan)- Age class from early 50 till 70 that has average spending score
In cluster 4(cyan coloured), we can see that people from age 50 to 70 have an average spending score. These people might be sick or unable to visit the mall much due to their ageing. The marketing people can attract this category by sending more online platform sales rather than shop visiting sales.


2. Second segmentation using Annual income(K$) and Spending score

![image](https://user-images.githubusercontent.com/87844891/129393035-87906f38-3b0a-4893-95ec-106fef27b0e1.png)
From the elbow method graph we can see that after the range at 5, the curve starts to flat steadily. With this, I can say the number of clusters for this dataset is 5.

![image](https://user-images.githubusercontent.com/87844891/129393120-6ac6e902-c2e2-48ae-849e-1fc2e88f59ec.png)

Analyzing the results for Annual income VS Spending score segmentation

We can see that the mall customers can be broadly grouped into 5 groups based on their purchases made in the mall.

In cluster 1(red-coloured)- Has low annual income but high spending score, we can see that people have low income but higher spending scores; these are those who love to buy products more often even though they have a low income. Maybe it’s because these people are more than satisfied with the mall services. The shops/malls might not target these people that effectively but still will not lose them.

In cluster 2(blue coloured)- Has average annual income and average spending score
we see that people have average income and an average spending score; these people again will not be the prime targets of the shops or mall, but again they will be considered, and marketing analytics may use other data analysis techniques to increase their spending score.

In cluster 3(green-coloured)- Has high annual income and high spending score
we see that people have high income and high spending scores, this is the ideal case for the mall or shops as these people are the prime sources of profit. These people might be regular customers of the mall and are convinced by the mall’s facilities.

In cluster 4(cyan coloured)- Has low annual income and low spending score
we can see people have low annual income and low spending scores, this is quite reasonable as people having low salaries prefer to buy less these are the wise people who know how to spend and save money. The shops/mall will be least interested in people belonging to this cluster.


Cluster 3(purple coloured)- Has high annual income but low spending score; we see that people have high income but low spending scores; this is interesting. Maybe these are the people who are unsatisfied or unhappy with the mall’s services. These can be the prime targets of the mall, as they have the potential to spend money. So, the mall authorities will try to add new facilities to attract these people and meet their needs.















 

