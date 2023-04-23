import pandas as pd
import numpy as np
import datetime as dt



import streamlit as st
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans



import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout='wide')

#ccreate columns to center pic
col1, col2, col3 = st.columns([2,5,2])
col2.image('online_retail.jpg')

# Create a sidebar
st.sidebar.title("Customer Segmentation")
@st.cache()
def read_data(filename):
    df1 = pd.read_csv(filename)
    return df1

df = pd.read_csv(r"C:\Users\marwa\OneDrive\Desktop\Final Project Epsilon\OnlineRetail.csv",encoding= 'unicode_escape')

st.header("Data ~")
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.write(df)

st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)


st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)
st.markdown("<h3></h3>",unsafe_allow_html=True)
rad = st.sidebar.radio(' ',['Introduction', "Data Exploration", 'RFM Analysis', "K-Means Clustering"
                            , "Cluster Calculator"])

df.drop_duplicates(keep = 'first', inplace=True)
df['Description'] = df.Description.str.lower()

df = df.dropna()

df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0.05]

df['TotalAmount'] = df['Quantity']*df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
retail = df.copy()


#introduction section
if rad == 'Introduction':
    st.markdown("<h1 style='text-align: center; color: Yellow;'>Customer Segmentation using RFM and K-Means Clustering</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: Yellow;'>by Marwan Hafez</h3>", unsafe_allow_html=True)
    st.write("Customer segmentation is a key aspect of Marketing, allowing the businesses to better understand the behavior of their customers and targeting them more efficiently. Traditional methods include certain segmentation bases such as Geographical, Demographic, or Behavioral. One of the most famous methods is by using RFM which tracks customers' buying behavior including the recency, frequency and monetary value of their purchases. However, RFM scores are usually pre-determined and can take a long time to calculate and apply. Here is where Machine Learning comes in and makes things much easier. By using unsupervised ML models, we can automatically detect different clusters in our customers based on their transactions.")
    




#data exploration
if rad == 'Data Exploration':
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>Data Exploration</h1>", unsafe_allow_html=True)
    
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.image('Top_Selling_products.png')

    st.write('')
    st.write('')
    st.write('')
    st.write('') 

    st.image('Country_count.png')

    retail_month = retail[retail.InvoiceDate.dt.year==2010]
    monthly_gross = retail_month.groupby(retail_month.InvoiceDate.dt.month).TotalAmount.sum()

    st.write('')
    st.write('')
    st.write('')
    st.write('') 

    st.image('Total_income_2010png.png')

    st.write('')
    st.write('')
    st.write('')
    st.write('')

    retail_month = retail[retail.InvoiceDate.dt.year==2011]
    monthly_gross = retail_month.groupby(retail_month.InvoiceDate.dt.month).TotalAmount.sum()

    st.image('income_2011.png')

    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.image('transactions.png')

    st.write('')
    st.write('')
    st.write('')
    st.write('')


    fig5 = plt.figure(figsize = (20,5))
    fig5.suptitle("Visualisation of outliers",size=20)

    axes = fig5.add_subplot(1, 3, 1)
    sns.boxplot(data=df,y="UnitPrice")

    axes = fig5.add_subplot(1, 3, 2)
    sns.boxplot(data=df,y="Quantity")

    axes = fig5.add_subplot(1, 3, 3)
    sns.boxplot(data=df,y="TotalAmount")

    st.pyplot(fig5)

    st.write('')
    st.write('')
    st.write('')
    st.write('')

#RFM
if rad == 'RFM Analysis':
    st.markdown("<h1 style='text-align: center; color: Yellow;'>RFM Analysis</h1>", unsafe_allow_html=True)
    st.subheader("Now, we must calculate each element of the RFM. But before doing that, we must create a new dataframe that contains each unique customer id and then we can add the relevant values.")

    # Recency
    st.write("To calculate Recency, we must first get the date of the most recent purchase.")
    st.write("We will take our reference point as the max invoice date in our dataset which represents the most recent date, and our recency will be based on days.")
    # Frequency
    st.write("For Frequency, we will count the distinct number of times that each customer has placed an order.")

    ## Monetary
    st.write("And finally, for Monetary Value, we will sum up the Sales of each customer to find how much he has spent in total.")

    today = dt.date(2011,12,10)
    #Create a new column called date which contains the date of invoice only
    df['date'] = df['InvoiceDate'].dt.date

    #group by customers and check last date of purshace
    recency_df = df.groupby(by='CustomerID', as_index=False)['date'].max()
    recency_df.columns = ['CustomerID','LastPurshaceDate']
    st.table(recency_df.head())

    #calculate recency
    recency_df['Recency'] = recency_df['LastPurshaceDate'].apply(lambda x: (today - x).days)
    st.table(recency_df.head())

    #drop LastPurchaseDate as we don't need it anymore
    recency_df.drop('LastPurshaceDate',axis=1,inplace=True)

    # drop duplicates
    df.copy = df
    df.copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
    #calculate frequency of purchases
    frequency_df = df.copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
    frequency_df.columns = ['CustomerID','Frequency']
    st.table(frequency_df.head())

    monetary_df = df.groupby(by='CustomerID',as_index=False).agg({'TotalAmount': 'sum'})
    monetary_df.columns = ['CustomerID','Monetary']
    st.table(monetary_df.head())



    #merge recency dataframe with frequency dataframe
    temp_df = recency_df.merge(frequency_df,on='CustomerID')

    #merge with monetary dataframe to get a table with the 3 columns
    rfm_df = temp_df.merge(monetary_df,on='CustomerID')
    #use CustomerID as index
    rfm_df.set_index('CustomerID',inplace=True)
    #check the head
    st.table(rfm_df.head())

    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.image('RFM_outliers.png')

    # Removing (statistical) outliers for Monetary
    Q1 = rfm_df.Monetary.quantile(0.05)
    Q3 = rfm_df.Monetary.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Monetary >= Q1 - 1.5*IQR) & (rfm_df.Monetary <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Recency
    Q1 = rfm_df.Recency.quantile(0.05)
    Q3 = rfm_df.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Recency >= Q1 - 1.5*IQR) & (rfm_df.Recency <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Frequency
    Q1 = rfm_df.Frequency.quantile(0.05)
    Q3 = rfm_df.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Frequency >= Q1 - 1.5*IQR) & (rfm_df.Frequency <= Q3 + 1.5*IQR)]

    # Rescaling the attributes
    import sklearn
    from sklearn.preprocessing import StandardScaler

    rfm = rfm_df[['Recency','Frequency', 'Monetary']]

    # Instantiate
    scaler = StandardScaler()

    # fit_transform
    rfm_scaled = scaler.fit_transform(rfm)
    rfm_scaled.shape

    rfm_scaled = pd.DataFrame(rfm_scaled)
    rfm_scaled.columns = ['Recency', 'Frequency', 'Monetary']
    st.table(rfm_scaled.head())

    
    st.image('heatmap.png')

    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.image('PAIRPLOT.png')

    st.write('')
    st.write('')
    st.write('')
    st.write('')

    st.header('To get a better understanding of the dataset, we can construct a scatter matrix of each of the three features present in the RFM data')

    st.write('')
    st.write('')

    st.image('scatter1.png')

    st.write('')
    st.write('')
    
    st.header('We can notice that we have a skewed distribution of the 3 variables and there exist outliers. This indicates how normalization is required to make the data features normally distributed as clustering algorithms require them to be normally distributed.')

    st.write('')
    st.write('')

    #log transformation
    rfm_r_log = np.log(rfm['Recency']+0.1) #can't take log(0) and so add a small number
    rfm_f_log = np.log(rfm['Frequency'])
    rfm_m_log = np.log(rfm['Monetary']+0.1)

    log_data = pd.DataFrame({'Monetary': rfm_m_log,'Recency': rfm_r_log,'Frequency': rfm_f_log})

    st.table(log_data.head())

    st.image('scatter2.png')

    st.header('The distributions of Frequency and Monetary are better, more normalized, but it"s not the case with Recency Distribution, which is improved but not as much.')

    st.write('')
    st.write('')

    st.image('heatmap2.png')

    customers_rank = rfm_df
    # Create a new column that is the rank of the value of coverage in ascending order
    customers_rank['Rank'] = customers_rank['Monetary'].rank(ascending=0)
    #customers_rank.drop('RevenueRank',axis=1,inplace=True)
    st.table(customers_rank.head())

    st.write('')
    st.write('')
    st.header('Top Customers')
    st.table(customers_rank.sort_values('Rank',ascending=True).head())

    #get top 20% of the customers
    top_20 = 3863 *20 /100
    

    #sum the monetary values over the customer with rank <=773
    RevenueByTop20 = customers_rank[customers_rank['Rank'] <= 772]['Monetary'].sum()
    
    st.subheader('#### In our case, the 80% of total revenue is not achieved by the 20% of TOP customers but approximately, it does, because they are less than our 20% TOP customers who achieve it. It would be interesting to study this group of customers because they are those who make our most revenue.')

    st.subheader('Applying RFM score formula. The simplest way to create customers segments from RFM Model is to use Quartiles. We assign a score from 1 to 4 to Recency, Frequency and Monetary. Four is the best/highest value, and one is the lowest/worst value. A final RFM score is calculated simply by combining individual RFM score numbers.')

    quantiles = rfm_df.quantile(q=[0.25,0.5,0.75])

    quantiles.to_dict()

    # Arguments (x = value, p = recency, monetary_value, frequency, d = quartiles dict)
    def RScore(x,p,d):
        if x <= d[p][0.25]:
            return 4
        elif x <= d[p][0.50]:
            return 3
        elif x <= d[p][0.75]: 
            return 2
        else:
            return 1
    # Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
    def FMScore(x,p,d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.50]:
            return 2
        elif x <= d[p][0.75]: 
            return 3
        else:
            return 4


    #create rfm segmentation table
    rfm_segmentation = rfm_df
    rfm_segmentation['R_Quartile'] = rfm_segmentation['Recency'].apply(RScore, args=('Recency',quantiles,))
    rfm_segmentation['F_Quartile'] = rfm_segmentation['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
    rfm_segmentation['M_Quartile'] = rfm_segmentation['Monetary'].apply(FMScore, args=('Monetary',quantiles,)) 

    st.table(rfm_segmentation.head())

    rfm_segmentation['RFMScore'] = rfm_segmentation.R_Quartile.map(str) \
                                + rfm_segmentation.F_Quartile.map(str) \
                                + rfm_segmentation.M_Quartile.map(str)
    st.table(rfm_segmentation.head())

    st.write('')
    st.write('')

    st.subheader('Best Recency score = 4: Most recently purchase. Best Frequency score = 4: Most quantity purchase. Best Monetary score = 4: Spent the most.')
    
    st.write('')

    st.subheader("Let's take a look on our best customers ")
    st.table(rfm_segmentation[rfm_segmentation['RFMScore']=='444'].sort_values('Monetary', ascending=False).head(10))

    st.write('')
    st.write('')

    st.subheader("And then, Let's take a look on how many customer do we have in each segment")

    st.write("Best Customers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))
    st.write('Loyal Customers: ',len(rfm_segmentation[rfm_segmentation['F_Quartile']==4]))
    st.write("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_Quartile']==4]))
    st.write('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='244']))
    st.write('Lost Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='144']))
    st.write('Lost Cheap Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))

    st.write('')
    st.write('')

    st.subheader('Now that we knew our customers segments we can choose how to target or deal with each segment.')

    st.subheader('Apparently there are 15% of customers considered as Champions. These customers are responsible for a big share of your revenue so we can put a lot of effort into keeping imroving their experience. What we can do:  Give them something extra that the regulars do not get, for example, limited series of products or special discounts to make them feel valued. Use communication similar to the Loyal segment. For example making them ambassadors, giving them a margin of your profits for bringing you, new customers. Ask them for feedbacks as they might know the products and services very well.')


#K-Means
if rad == 'K-Means Clustering':
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>K-Means Clustering for Segmentation</h1>", unsafe_allow_html=True)
    st.subheader('For this task, we will be using an unsupervised Machine Learning algorithm which is K-Means which identifies k number of centroids, and then allocates every data point to the nearest cluster (based on similarities), while keeping the centroids as small as possible.')
    # # # Applying K-Means Clustering
    st.write("The algorithm works as follows, First we initialize k points called means, randomly. We categorize each item to its closest mean and we update the means coordinates, which are the averages of the items categorized in that mean so far.We repeat the process for a given number of iterations and at the end, we have our clusters.")
    
    st.write('')
    st.write('')

    today = dt.date(2011,12,10)
    #Create a new column called date which contains the date of invoice only
    df['date'] = df['InvoiceDate'].dt.date

    #group by customers and check last date of purshace
    recency_df = df.groupby(by='CustomerID', as_index=False)['date'].max()
    recency_df.columns = ['CustomerID','LastPurshaceDate']

    #calculate recency
    recency_df['Recency'] = recency_df['LastPurshaceDate'].apply(lambda x: (today - x).days)

    #drop LastPurchaseDate as we don't need it anymore
    recency_df.drop('LastPurshaceDate',axis=1,inplace=True)

    # drop duplicates
    df.copy = df
    df.copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
    #calculate frequency of purchases
    frequency_df = df.copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
    frequency_df.columns = ['CustomerID','Frequency']

    monetary_df = df.groupby(by='CustomerID',as_index=False).agg({'TotalAmount': 'sum'})
    monetary_df.columns = ['CustomerID','Monetary']

    #merge recency dataframe with frequency dataframe
    temp_df = recency_df.merge(frequency_df,on='CustomerID')

    #merge with monetary dataframe to get a table with the 3 columns
    rfm_df = temp_df.merge(monetary_df,on='CustomerID')
    #use CustomerID as index
    rfm_df.set_index('CustomerID',inplace=True)
    

    # Removing (statistical) outliers for Monetary
    Q1 = rfm_df.Monetary.quantile(0.05)
    Q3 = rfm_df.Monetary.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Monetary >= Q1 - 1.5*IQR) & (rfm_df.Monetary <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Recency
    Q1 = rfm_df.Recency.quantile(0.05)
    Q3 = rfm_df.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Recency >= Q1 - 1.5*IQR) & (rfm_df.Recency <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Frequency
    Q1 = rfm_df.Frequency.quantile(0.05)
    Q3 = rfm_df.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Frequency >= Q1 - 1.5*IQR) & (rfm_df.Frequency <= Q3 + 1.5*IQR)]

    # Rescaling the attributes
    import sklearn
    from sklearn.preprocessing import StandardScaler

    rfm = rfm_df[['Recency','Frequency', 'Monetary']]

    # Instantiate
    scaler = StandardScaler()

    # fit_transform
    rfm_scaled = scaler.fit_transform(rfm)

    rfm_scaled = pd.DataFrame(rfm_scaled)
    rfm_scaled.columns = ['Recency', 'Frequency', 'Monetary']
    

    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(rfm_scaled)

    st.header('Finding the Optimal Number of Clusters')
    st.subheader('The Elbow Method is one of the most popular methods to determine this optimal value of k.')

    st.write('')
    st.write('')

    # Elbow-curve

    ssd = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
        kmeans.fit(rfm_scaled)
        
        ssd.append(kmeans.inertia_)
        
    st.image('elbow.png')

    


    
    # Final model with k=3
    kmeans = KMeans(n_clusters=3, max_iter=50)
    kmeans.fit(rfm_scaled)

    # assign the label
    rfm['Cluster_Id'] = kmeans.labels_

    st.write('')
    st.write('')

    st.image('monetary.png')

    st.write('')
    st.write('')

    st.image('frequency.png')

    st.write('')
    st.write('')

    st.image('recency.png')

    st.write('')
    st.write('')

    st.image('3D.png')

    st.write('')
    st.write('')

    st.image('maxtrans.png')

    st.write('')
    st.write('')
    
    st.image('recenttrans.png')


if rad == "Cluster Calculator":
    st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>Cluster Calculator</h1>", unsafe_allow_html=True)
    st.header('Now you can try inputting some values for RFM and see which cluster this imaginary customer can be apart of')
    st.subheader("To be able to do that, we will consider that any customer whose RFM values fall between the IQR of the Cluster average values, can be part of that cluster.")
    today = dt.date(2011,12,10)
    #Create a new column called date which contains the date of invoice only
    df['date'] = df['InvoiceDate'].dt.date

    #group by customers and check last date of purshace
    recency_df = df.groupby(by='CustomerID', as_index=False)['date'].max()
    recency_df.columns = ['CustomerID','LastPurshaceDate']

    #calculate recency
    recency_df['Recency'] = recency_df['LastPurshaceDate'].apply(lambda x: (today - x).days)

    #drop LastPurchaseDate as we don't need it anymore
    recency_df.drop('LastPurshaceDate',axis=1,inplace=True)

    # drop duplicates
    df.copy = df
    df.copy.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep="first", inplace=True)
    #calculate frequency of purchases
    frequency_df = df.copy.groupby(by=['CustomerID'], as_index=False)['InvoiceNo'].count()
    frequency_df.columns = ['CustomerID','Frequency']

    monetary_df = df.groupby(by='CustomerID',as_index=False).agg({'TotalAmount': 'sum'})
    monetary_df.columns = ['CustomerID','Monetary']



    #merge recency dataframe with frequency dataframe
    temp_df = recency_df.merge(frequency_df,on='CustomerID')

    #merge with monetary dataframe to get a table with the 3 columns
    rfm_df = temp_df.merge(monetary_df,on='CustomerID')
    #use CustomerID as index
    rfm_df.set_index('CustomerID',inplace=True)
    
    # Removing (statistical) outliers for Monetary
    Q1 = rfm_df.Monetary.quantile(0.05)
    Q3 = rfm_df.Monetary.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Monetary >= Q1 - 1.5*IQR) & (rfm_df.Monetary <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Recency
    Q1 = rfm_df.Recency.quantile(0.05)
    Q3 = rfm_df.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Recency >= Q1 - 1.5*IQR) & (rfm_df.Recency <= Q3 + 1.5*IQR)]

    # Removing (statistical) outliers for Frequency
    Q1 = rfm_df.Frequency.quantile(0.05)
    Q3 = rfm_df.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm_df = rfm_df[(rfm_df.Frequency >= Q1 - 1.5*IQR) & (rfm_df.Frequency <= Q3 + 1.5*IQR)]

    # Rescaling the attributes
    import sklearn
    from sklearn.preprocessing import StandardScaler

    rfm = rfm_df[['Recency','Frequency', 'Monetary']]

    # Instantiate
    scaler = StandardScaler()

    # fit_transform
    rfm_scaled = scaler.fit_transform(rfm)

    rfm_scaled = pd.DataFrame(rfm_scaled)
    rfm_scaled.columns = ['Recency', 'Frequency', 'Monetary']
    

    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(rfm_scaled)

    st.write('')
    st.write('')

    # Elbow-curve

    ssd = []
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
        kmeans.fit(rfm_scaled)
        
        ssd.append(kmeans.inertia_)
        
    

    
    # Final model with k=3
    kmeans = KMeans(n_clusters=3, max_iter=50)
    kmeans.fit(rfm_scaled)

    # assign the label
    rfm['Cluster_Id'] = kmeans.labels_

    unique_clusters = rfm['Cluster_Id'].unique()
    col1, col2, col3 = st.columns((1,1,1))
    r= col1.number_input('Add Recency')
    f= col2.number_input('Add Frequency')
    m= col3.number_input('Add Monetary Value')
    c= ' '

    #Recency
    rec_q1 = rfm.groupby('Cluster_Id')['Recency'].quantile(0.25)
    rec_q3 = rfm.groupby('Cluster_Id')['Recency'].quantile(0.75)

    #Frequency
    freq_q1 = rfm.groupby('Cluster_Id')['Frequency'].quantile(0.25)
    freq_q3 = rfm.groupby('Cluster_Id')['Frequency'].quantile(0.75)

    #monetary
    monetary_q1 = rfm.groupby('Cluster_Id')['Monetary'].quantile(0.25)
    monetary_q3 = rfm.groupby('Cluster_Id')['Monetary'].quantile(0.25)

    

    #iterate for each cluster to see if it fits
    if st.button("Click here to calculate!"):
        for n in range(len(unique_clusters)):
            if (rec_q1[n] <= r <= rec_q3[n]) and (freq_q1[n] <= f <= freq_q3[n]) and (monetary_q1[n] <= m <= monetary_q3[n]):
                c = n
                st.balloons()
                st.success('Congratulations! This customer can be added to cluster number $d' % c)
            else:
                st.error('Try again! This customer does not fit in cluster %d' % n)