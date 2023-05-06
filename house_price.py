import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from scipy.stats import stats
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from pylab import *

# options for data plot
matplotlib.use('TkAgg')  # backend
plt.rc('axes', axisbelow=True)  # axis position below data
plt.rcParams["figure.figsize"] = (7, 6)  # plot dimensions

pd.set_option('display.expand_frame_repr', False)
df = pd.read_csv('utils/kc_house_data.csv')

# https://www.kaggle.com/harlfoxem/housesalesprediction

'''King is a county in the Washington state, the capital is Seattle. This dataset contains some features of the 
houses like number of bedrooms and bathrooms, year when the house was built and so on. The aim of this project is to 
predict the price of the houses. The data are refer to the house sold between May 2014 and May 2015.'''

# show political map of the Washington state and King county
plt.figure(1)
plt.subplot(211)
plt.imshow(mpimg.imread('utils/washington_state.png'))
plt.title('Washington state')
plt.axis('off')
plt.subplot(212)
plt.imshow(mpimg.imread('utils/king_county.png'))
plt.axis('off')
plt.title('King county')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

print(df.head())
print(df.shape)

# Let's start with exploratory data analysis and feature eng.

# check for null values
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.xticks(rotation=45)
plt.title('Heatmap of null values')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()
# there are not null values


# the features 'id' seems not useful for this work, let's drop it
df = df.drop(['id'], axis=1)

# 'date' feature is in object format, better convert it in datetime
df['date'] = pd.to_datetime(df['date'])

print(df['date'].nlargest(5))
print(df['date'].nsmallest(5))

'''the data with day when the house was sold is too much accurate, we're interested about the month, the year will be 
discarded because we have only two years and different months of that years '''
df['date'] = df['date'].dt.month
df = df.rename(columns={'date': 'month'})
sns.countplot(x='month', data=df)
plt.title('Houses sold per month')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# consider May is counted twice (2014 and 2015)
'''we are not really interested at the number of house sold in a month but at the variation of the average price from 
one month to another '''

avg_price = df.groupby('month', as_index=False)['price'].mean()  # compute average price of the house per month
avg_price.sort_values(by=['month'])
sns.catplot(x='month', y='price', kind='point', linestyles='--', data=avg_price)
plt.ylabel('avg_price')
plt.grid()
plt.title('Average price per month')
plt.tight_layout()
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# there is a significant variation in the average price

'''If we represent months with a number we are imposing an order. Let's group features by trimester and dummying'''
df['trimester'] = pd.qcut(df.month, q=4, labels=[1, 2, 3, 4]).astype(np.int64)
df = df.drop(['month'], axis=1)

avg_price = df.groupby('trimester', as_index=False)['price'].mean()  # compute average price of the house per trimester

avg_price.sort_values(by=['trimester'])

sns.catplot(x='trimester', y='price', kind='point', linestyles='--', data=avg_price)
plt.ylabel('avg_price')
plt.grid()
plt.title('Average price per trimester')
plt.tight_layout()
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# now let's dummying trimester features
# df = pd.get_dummies(df, columns=['trimester'], drop_first=True)

'''the avg_price per trimester has very small variation, seems that these feature isn't useful'''
df = df.drop(['trimester'], axis=1)

# let's check bedrooms and bathrooms values

sns.countplot(x='bedrooms', data=df)
plt.title('Houses grouped by number of bedrooms')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()


'''there are some values that are too big, and other house doesn't have a bedroom. Furthermore the samples with a
more than 7 bedroom is very few. Let's remove this samples. Can be seen as high leverage point'''
df = df[df.bedrooms <= 7]
df = df[df.bedrooms != 0]
sns.countplot(x='bedrooms', data=df)
plt.title('Houses grouped by number of bedrooms')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

sns.countplot(x='bathrooms', data=df)
plt.xticks(rotation=45)
plt.title('Houses grouped by number of bathrooms')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

'''there are values that are not integer because some bathroom doesn't have all the bathroom object. for example 0.75 
indicate there is a sick, toilet and shower or bath. A full bath consist of all of this four object. After this 
considerations we can drop values less than 0.75. Furthermore values bigger then 5.00 are very few and we will remove 
these too. Can be seen as high leverage point.'''
df = df[0.75 <= df.bathrooms]
df = df[df.bathrooms <= 5]

# we can consider 0.75 like 1 bathroom
df['bathrooms'] = df.bathrooms.apply(lambda x: x // 0.75)
df['bathrooms'] = df['bathrooms'].astype(np.int64)

sns.countplot(x='bathrooms', data=df)
plt.title('Houses grouped by number of bathrooms')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# let's see correlation between sqft features and sqft features and price
corr_sqft = df[['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'price']]
sns.heatmap(corr_sqft.corr(), annot=True, cmap='flare')
plt.title('Correlation matrix between sqft features and price')
plt.xticks(rotation=45)
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

df = df.drop(['sqft_lot15'], axis=1)  # there isn't correlation between sqft_lot15 and price
df = df.drop(['sqft_above'], axis=1)  # high correlation with sqft_living
df = df.drop(['sqft_living15'], axis=1)  # high correlation with sqft_living

'''remember that sqft_living represents the area livable of the house, we can think this as the are we can heat. The 
correlation between sqft_above and sqft_living is quite high as we expected, the correlation between sqft_living and 
sqft_basement is quite low because with sqft_basement should indicate the presence of the garage and this isn't 
considered as a living area '''

# sqm_basement is 0 if there isn't a basement, vice versa the corresponding value. we can discretize this values
df['has_basement'] = df.sqft_basement.apply(lambda x: 1 if x != 0 else 0)
plt.title("Houses grouped by 'has_basement'")
sns.countplot(x='has_basement', data=df)
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

df = df.drop(['sqft_basement'], axis=1)

sns.boxplot(data=df, x='sqft_living')
plt.ylabel('surface')
plt.title('Distribution of sqft_living')
plt.grid()
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# boxplot shows that there are some influence point, let's drop it
df = df[df.sqft_living <= 5000]  # ~464mq

sns.boxplot(data=df, x='sqft_living')
plt.ylabel('surface')
plt.title('Distribution of sqft_living')
plt.grid()
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

sns.boxplot(x='sqft_lot', data=df)
plt.xlabel('surface')
plt.title('Distribution of sqft_lot')
plt.grid()
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# there are a lot of outliers
df = df[df.sqft_lot <= 20000]  # ~1800mq
sns.boxplot(x='sqft_lot', data=df)
plt.xlabel('surface')
plt.title('Distribution of sqft_lot')
plt.grid()
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# now consider the number of floors
sns.countplot(x='floors', data=df)
plt.title('Houses grouped by number of floors')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

'''0.5 indicates the presence of a garage or attic, there aren't strange values. we can drop samples with value of 
3.5 because there are very few '''
df = df[df.floors < 3.5]
sns.countplot(x='floors', data=df)
plt.title('Houses grouped by number of floors')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# now consider waterfront, view, condition, grade, zipcode
# waterfront {0,1}
# view {0,4}
# condition {1,5}
# grade {1,13}

figure, ax = plt.subplots(ncols=2, nrows=2)
sns.countplot(x='waterfront', data=df, ax=ax[0, 0])
sns.countplot(x='view', data=df, ax=ax[0, 1])
sns.countplot(x='condition', data=df, ax=ax[1, 0])
sns.countplot(x='grade', data=df, ax=ax[1, 1])
plt.tight_layout()
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# there aren't wrong values

corr_quality = df[['waterfront', 'view', 'condition', 'grade', 'zipcode', 'price']]
sns.heatmap(corr_quality.corr(), annot=True, cmap='crest')
plt.xticks(rotation=45)
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# zipcode and price are practically uncorrelated, drop it.
df = df.drop(['zipcode'], axis=1)
# same for condition. We can think the condition 1 is however an acceptable condition of the house
df = df.drop(['condition'], axis=1)

# the yr_built must be less than the yr_renovated (if yr_renovate != 0)
yr_check = df[df.yr_built > df.yr_renovated]
sns.scatterplot(x='yr_built', y='yr_renovated', data=yr_check)
plt.title('Check for wrong values of yr_renovated')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()
# there aren't wrong values

# with the features yr_renovated we can apply the same idea as sqm_basement
df['has_renovated'] = df.yr_renovated.apply(lambda x: 1 if x != 0 else 0)
sns.countplot(x='has_renovated', data=df)
plt.title("Houses grouped by 'has_renovated'")
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

df = df.drop(['yr_renovated'], axis=1)

# plot price in relation with position(lat, long)
vMin = df['price'].min()
vMax = df['price'].max()
bounds = [vMin, 250000, 500000, 750000, 1000000, 2000000, vMax]

plt.figure(1)
plt.imshow(mpimg.imread('utils/king_county.png'))
plt.axis('off')

plt.figure(2)
plt.scatter(x=df['long'], y=df['lat'], c=df['price'], cmap='summer_r')
plt.colorbar(label='price', format='%i', boundaries=bounds)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Price variations for latitude and longitude')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# there are only few samples in the est of the county
df = df[df.long < -121.7]

plt.scatter(x=df['long'], y=df['lat'], c=df['price'], cmap='summer_r')
plt.colorbar(label='price', format='%i', boundaries=bounds)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Price variations for latitude and longitude')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# reordering features
df = df[[
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
    'grade', 'yr_built', 'lat', 'long', 'has_basement', 'has_renovated']]

# verify the final correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='rocket_r')
plt.xticks(rotation=45)
plt.title('Final correlation matrix')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# there is a very low correlation between yr_build-price and long-price
df = df.drop(['yr_built'], axis=1)
df = df.drop(['long'], axis=1)

# verify the final correlation matrix (bis)
sns.heatmap(df.corr(), annot=True, cmap='rocket_r')
plt.xticks(rotation=45)
plt.title('Final correlation matrix')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

print(df.head())  # final dataset

sns.boxplot(x='price', data=df)
plt.title('Price distribution')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

df = df[df.price <= 1500000]  # remove outliers

sns.boxplot(x='price', data=df, )
plt.title('Price distribution')
thisManager = get_current_fig_manager()  # for plot position
thisManager.window.wm_geometry('+750+100')  # plot position
plt.show()

# now let's go with model selection and model assessment

#X = df.drop(['price'], axis=1)
#y = df['price']
#
#print(X.head())
#print(X.shape)
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#
## normalization
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)  # mean and variance must be calculated only on training set
#
## initialize models and search for the best params
#
#rr = Ridge()
#param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 10, 100]}
#clf_rr = GridSearchCV(rr, param, n_jobs=-1, cv=5, scoring='r2', verbose=0)
#
#mlp_r = MLPRegressor(max_iter=500)
#parameters = {
#    'hidden_layer_sizes': [(50,), (50, 100), (100, 100)],
#    'activation': ['relu', 'tanh'],
#    'solver': ['sgd', 'adam'],
#    'alpha': [0.0001, 0.001, 0.01, 0.1, 10, 100]
#}
#clf_mlp = GridSearchCV(mlp_r, parameters, n_jobs=-1, cv=5, scoring='r2', verbose=0)
#
## Ridge Regression
#
#clf_rr.fit(X_train, y_train)
#
#y_pred_train = clf_rr.predict(X_train)
#
#print(clf_rr.best_estimator_)
#print(clf_rr.best_params_)
#
#print('MAE on the training set: ', mean_absolute_error(y_train, y_pred_train))
#print('R2 on the training set: ', r2_score(y_train, y_pred_train))
#
#y_pred_test = clf_rr.predict(X_test)
#
#print('MAE on the test set: ', mean_absolute_error(y_test, y_pred_test))
#print('R2 on the test set: ', r2_score(y_test, y_pred_test))
#
## MLPRegressor
#
#clf_mlp.fit(X_train, y_train)
#
#y_pred_train = clf_mlp.predict(X_train)
#
#print(clf_mlp.best_estimator_)
#print(clf_mlp.best_params_)
#
#print('MAE on the training set: ', mean_absolute_error(y_train, y_pred_train))
#print('R2 on the training set: ', r2_score(y_train, y_pred_train))
#
#y_pred_test = clf_mlp.predict(X_test)
#
#print('MAE on the test set: ', mean_absolute_error(y_test, y_pred_test))
#print('R2 on the test set: ', r2_score(y_test, y_pred_test))

#######################################################################################

'''
Ridge(alpha=10)
{'alpha': 10}

MAE on the training set:  102753.03819620753
R2 on the training set:  0.6535360355867399

MAE on the test set:  101457.36094719908
R2 on the test set:  0.6573766002580392


MLPRegressor(alpha=10, hidden_layer_sizes=(100, 100), max_iter=500)
{'activation': 'relu', 'alpha': 10, 'hidden_layer_sizes': (100, 100), 'solver': 'adam'}

MAE on the training set:  75667.36255464946
R2 on the training set:  0.7910437388284552

MAE on the test set:  75427.56613267734
R2 on the test set:  0.786244546079363'''
