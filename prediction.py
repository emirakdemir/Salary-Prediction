import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score


warnings.simplefilter(action="ignore")
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("hitters.csv")

df.head(11)
df.shape
df.dtypes
df.tail()
df.head()
df.isnull()
df.isnull().sum()
df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

def grabColNames(dataframe, catTh=10, carTh=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

Parameters
------
    dataframe: dataframe
            Değişken isimleri alınmak istenilen dataframe
    catTh: int, optional
            numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    carTh: int, optinal
            kategorik fakat kardinal değişkenler için sınıf eşik değeri

Returns
------
    catCols: list
            Kategorik değişken listesi
    numCols: list
            Numerik değişken listesi
    catButCar: list
            Kategorik görünümlü kardinal değişken listesi

Examples
------
    import seaborn as sns
    df = sns.load_dataset("iris")
    print(grabColNames(df))


Notes
------
    catCols + numCols + catButCar = toplam değişken sayısı
    numButCat catCols'un içerisinde.
    Return olan 3 liste toplamı toplam değişken sayısına eşittir: catCols + numCols + catButCar = değişken sayısı

    """
    # catCols, catButCar
    catCols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numButCat = [col for col in dataframe.columns if dataframe[col].nunique() < catTh and
             dataframe[col].dtypes != "O"]
    catButCar = [col for col in dataframe.columns if dataframe[col].nunique() > carTh and
             dataframe[col].dtypes == "O"]
    catCols = catCols + numButCat
    catCols = [col for col in catCols if col not in catButCar]

    # numCols
    numCols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    numCols = [col for col in numCols if col not in numButCat]
   
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'catCols: {len(catCols)}')
    print(f'numCols: {len(numCols)}')
    print(f'catButCar: {len(catButCar)}')
    print(f'numButCat: {len(numButCat)}')
    return catCols, numCols, catButCar

catCols, numCols, catButCar = grabColNames(df)
# Observations: 320
# Variables: 26
# catCols: 5
# numCols: 21
# catButCar: 0
# numButCat: 1
catCols
numCols

# numerical and categorical variables analysis #

def catSummary(dataframe, colName, plot=False):
    print(pd.DataFrame({colName: dataframe[colName].value_counts(), "Ratio": 100 * dataframe[colName].value_counts() / len(dataframe)}))
    print("     ")
    if plot:
        sns.countplot(x=dataframe[colName], data=dataframe)
        plt.show(block=True)
for col in catCols:
    if df[col].dtypes == "bool":
        print(col)
    else:
        catSummary(df, col, True)
        
        
def numSummary(dataframe, numericalCol, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    print(dataframe[numericalCol].describe(quantiles).T)
    if plot:
        dataframe[numericalCol].hist()
        plt.xlabel(numericalCol)
        plt.title(numericalCol)
        plt.show(block=True)
for col in numCols:
    numSummary(df, col, True)


# target variable #

def targetSummaryWithNum(dataframe, target, numCol):
    print(dataframe.groupby(target).agg({numCol: "mean"}), end="\n\n\n")
for col in numCols:
    targetSummaryWithNum(df, "Salary", col)
    
def targetSummaryWithCat(dataframe, target, numCol):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(numCol)[target].mean()}), end="\n\n\n")
for col in catCols:
    targetSummaryWithCat(df, "Salary", col)
    

# correlation analysis #
df.corr()

def highCorrelatedCols(dataframe,plot=False,corrTh=0.90):
    corr=dataframe.corr()
    corrMatrix=corr.abs()
    upperTriangleMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))
    dropList = [col for col in upperTriangleMatrix.columns if any(upperTriangleMatrix[col] > 0.90)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return dropList
dropList = highCorrelatedCols(df)
dropList

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[numCols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# missing value analysis #

df.isnull().any()

def missingValuesTable(dataframe,naName = False):
    naColums = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    nMiss = dataframe[naColums].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[naColums].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missingDf = pd.concat([nMiss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missingDf,end='\n')
    if naName:
        return naColums
naColumns = missingValuesTable(df,True)

imputer = KNNImputer(n_neighbors=5)
df[numCols] = pd.DataFrame(imputer.fit_transform(df[numCols]), columns=numCols)

df.head()


# outlier analysis #

def outlierThresholds(dataframe, colName, q1=0.25, q3=0.75):
    quartile1 = dataframe[colName].quantile(q1)
    quartile3 = dataframe[colName].quantile(q3)
    interquantileRange = quartile3 - quartile1
    upLimit = quartile3 + 1.5 * interquantileRange
    lowLimit = quartile1 - 1.5 * interquantileRange
    return lowLimit, upLimit

def checkOutlier(dataframe, colName):
    lowLimit, upLimit = outlierThresholds(dataframe, colName)
    if dataframe[(dataframe[colName] > upLimit) | (dataframe[colName] < lowLimit)].any(axis=None):
        return True
    else:
        return False

def replaceWithThresholds(dataframe, colName):
    low, up = outlierThresholds(dataframe, colName)

    dataframe.loc[dataframe[colName] > up, colName] = up
    dataframe.loc[dataframe[colName] < low, colName] = low

for col in numCols:
    print(col,checkOutlier(df, col))
    if checkOutlier(df, col):
        replaceWithThresholds(df, col)

clf=LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[numCols])

df_scores = clf.negative_outlier_factor_
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True,xlim=[0,50],style=".-")
np.sort(df_scores)[0:10]
th = np.sort(df_scores)[2]
print(f'th: {th}')
print(df[df_scores < th].shape)

df[df_scores < th].index
df.drop(axis=0, labels=df[df_scores < th].index, inplace=True)


# model #

dff = df.copy()

def one_hot_encoder(dataframe, categoricalCols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categoricalCols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, catCols, drop_first=True)

y = dff["Salary"]
X = dff.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=159)

lin_model = LinearRegression().fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

lin_model.score(X_test, y_test) #0.6750230821350279

# RMSE #
#train
yPred = lin_model.predict(X_train)
np.sqrt((mean_squared_error(y_train, yPred))) #229.0817466114684
#test
yPred = lin_model.predict(X_test)
np.sqrt((mean_squared_error(y_test, yPred))) #253.59961072567535

# R-Squared #
#train
lin_model.score(X_train, y_train) #0.5659626989283189
#test
lin_model.score(X_test, y_test) #0.6750230821350279


# feature extraction #

df.columns = [col.upper() for col in df.columns]

# player's experience
df["NEW_YEARS_CAT"] = pd.cut(x=df["YEARS"], bins=[0, 5, 10, 15], labels=["rookie", "experienced", "veteran"])

# the average of the player's achievements over the course of his career, by time played
df['NEW_CAREER_YEARS'] = (df['CATBAT'] + df['CHITS'] + df['CHMRUN'] + df['CRUNS'] + df['CRBI'] + df['CWALKS']) / df['YEARS']

# ability of the player to get home run points
df.loc[df['HMRUN'] >= 10, 'NEW_HMRUN_CAT'] = 'homerunner'
df.loc[df['HMRUN'] < 10, 'NEW_HMRUN_CAT'] = 'not_homerunner'

# rate of conversion of the player's hits to score in the years 1986-1987
df['NEW_RUNS_ATBAT'] = df['RUNS'] / df['ATBAT']

# Net error contribution of the player in 1986-1987
df['NEW_ERRORS_RBI'] = df['WALKS'] - df['ERRORS']

# experience*benefit
df['NEW_RUNS_YEARS'] = df['RUNS'] * df['YEARS']

catCols, numCols, catButCar = grabColNames(df)
# Observations: 320
# Variables: 26
# catCols: 5
# numCols: 21
# catButCar: 0
# numButCat: 1


# encoding #

def labelEncoder(dataframe, binaryCol):
    labelencoder = LabelEncoder()
    dataframe[binaryCol] = labelencoder.fit_transform(dataframe[binaryCol])
    return dataframe
binaryCols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binaryCols

for col in binaryCols:
    df = labelEncoder(df, col)
catCols

df = one_hot_encoder(df, catCols, drop_first=True)


# scaling #

scaler = MinMaxScaler()
df[numCols] = scaler.fit_transform(df[numCols])
df.head()


# model #
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=159)

reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

reg_model.score(X_test, y_test) #0.7127301155265928 
reg_model.intercept_
reg_model.coef_

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.coef_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('significance.png')
        
plot_importance(reg_model,X)

# RMSE #
#train
y_pred=reg_model.predict(X_train)
np.sqrt((mean_squared_error(y_train, y_pred))) #0.1368804666505278
#test
y_pred=reg_model.predict(X_test)
np.sqrt((mean_squared_error(y_test, y_pred))) #0.1595406860830553


# R-Squared #
#train
reg_model.score(X_train,y_train) #0.6538846376472665
#test
reg_model.score(X_test,y_test) #0.7127301155265928


# Cross Validation #
np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=10,scoring="neg_mean_absolute_error"))) #0.33711594702043335