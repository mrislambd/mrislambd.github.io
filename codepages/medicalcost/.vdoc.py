# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: false
import pandas as pd

insurance = pd.read_csv('insurance.csv')

insurance.sample(5, random_state=111)
#
#
#
#
#
#| code-fold: false
insurance.info()
#
#
#
#
#
#
#
#
#
#| code-fold: false
print(insurance.describe().T)
#
#
#
#
#
#| code-fold: false
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3,1, figsize = (6,18))

sns.histplot(insurance['age'], color='red', kde=True, ax= axes[0])
sns.histplot(insurance['bmi'], color='green', kde=True, ax= axes[1])
sns.histplot(insurance['charges'], color='blue', kde=True, ax= axes[2])

for ax in axes:
    ax.set_facecolor('#f4f4f4')
plt.gcf().patch.set_facecolor('#f4f4f4')
#
#
#
#
#
#| code-fold: false

fig = plt.figure(figsize=(8, 12))
grid = fig.add_gridspec(3, 2)

ax1 = fig.add_subplot(grid[0, 0])
sns.countplot(x=insurance['sex'], hue=insurance['sex'], palette='Set1', legend=False, ax=ax1)
ax1.set_title('Gender Distribution')
ax1.set_facecolor('#f4f4f4')

ax2 = fig.add_subplot(grid[0, 1])
sns.countplot(x=insurance['smoker'], hue=insurance['smoker'], palette='Set2', ax=ax2)
ax2.set_title('Smoker Distribution')
ax2.set_facecolor('#f4f4f4')

ax3 = fig.add_subplot(grid[1, :])
sns.countplot(x='smoker', data=insurance, hue='sex', palette='Set3', ax=ax3)
ax3.set_title('Smoker Distribution by Gender')
ax3.set_facecolor('#f4f4f4')

ax4 = fig.add_subplot(grid[2, :])
sns.countplot(x=insurance['region'], hue=insurance['region'], palette='Set1', ax=ax4)
ax4.set_title('Region Distribution')
ax4.set_facecolor('#f4f4f4')

plt.gcf().patch.set_facecolor('#f4f4f4')
plt.tight_layout()
plt.show()
#
#
#
#
#
#| code-fold: false
corr_matrix = insurance[['age','bmi','children','charges']].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.gca().set_facecolor('#f4f4f4') 
plt.gcf().patch.set_facecolor('#f4f4f4')
plt.show()
#
#
#
#
#
#| code-fold: false

fig, axes = plt.subplots(5,1, figsize=(8,25))
sns.scatterplot(x='age', y='charges', data=insurance, hue='sex' ,ax=axes[0])
axes[0].set_title('Age vs Charges')
sns.scatterplot(x='bmi', y='charges', data=insurance, hue='sex' ,ax=axes[1])
axes[1].set_title('BMI vs Charges')
sns.boxplot(x='children', y='charges', data=insurance, ax=axes[2])
axes[2].set_title('Children vs Charges')
sns.boxplot(x='sex', y='charges', data=insurance, ax=axes[3])
axes[3].set_title('Gender vs Charges')
sns.boxplot(x='smoker', y='charges', data=insurance, ax=axes[4])
axes[4].set_title('Smoking vs Charges')

for ax in axes:
    ax.set_facecolor('#f4f4f4')
plt.gcf().patch.set_facecolor('#f4f4f4')
#
#
#
#
#
#| code-fold: false
sns.boxplot(y='charges', data=insurance)
plt.title('Boxplot of Charges')
plt.gca().set_facecolor('#f4f4f4') 
plt.gcf().patch.set_facecolor('#f4f4f4')
plt.show()
#
#
#
#
#
#
#
#
#| code-fold: false
# Binary Encoding for the variables with two categories
insurance['sex'] = insurance['sex'].map({'male':1, 'female':0})
insurance['smoker'] = insurance['smoker'].map({'yes':1, 'no':0})

# One-Hot Encoding for the multiclas variable: region
insurance = pd.get_dummies(
    insurance, columns=['region'],
    drop_first=True,
    dtype=int
    )
#
#
#
#
#
#| code-fold: false
# Round the continuous charge variable to 2 decimal places
insurance['charges'] = insurance['charges'].round(2)

# Mofe the predicting variable at the end of the dataframe
insurance_charges = insurance.pop('charges')
insurance.insert(loc = len(insurance.columns), column='charges', value=insurance_charges)

# Quick look of the dataframe
insurance.head()
#
#
#
#
#
#| code-fold: false
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = insurance.drop('charges', axis=1)
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
print(vif_data)
#
#
#
#
#
#
#
#| code-fold: false
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(insurance.age.values.reshape(-1,1), insurance.bmi.values)
insurance['bmi_adusted_for_age'] = insurance.bmi.values - reg.predict(insurance.age.values.reshape(-1,1))
#
#
#
#
#
#| code-fold: false
insurance['bmi_age_interact'] = insurance['bmi']*insurance['age']
#
#
#
#
#
#| code-fold: false
bins = [18,30,40,50,60,70]
labels = ['18-30','31-40','41-50','51-60','61-70']
insurance['age_group'] = pd.cut(insurance['age'], bins= bins, labels=labels)
insurance['bmi_zscore'] = insurance.groupby('age_group', observed=False)['bmi'].transform(lambda x: (x-x.mean())/x.std())
insurance.sample(5, random_state=111)
#
#
#
#
#
#| code-fold: false
insurance.info()
#
#
#
#
#
#| code-fold: false
insurance = insurance.dropna()
insurance.info()
#
#
#
#
#
#
#| code-fold: false
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
#
#
#
#| code-fold: false
insurance_train, insurance_test = train_test_split(
    insurance, test_size=0.2,
    random_state=111
)
#
#
#
#
#| code-fold: false
#| code-overflow: scroll
kfold = KFold(9, shuffle=True, random_state=111)
mses = np.zeros((9,9))
k = 12
for i, (train_index, test_index) in enumerate(kfold.split(insurance_train)):
    # Training Set
    insurance_train_train = insurance_train.iloc[train_index]
    # Holdout Set 
    insurance_train_test = insurance_train.iloc[test_index]

    prediction0 = insurance_train_train.charges.mean()*np.ones(len(test_index))

    model1 = LinearRegression()
    model2 = LinearRegression()
    model3 = LinearRegression()
    model4 = LinearRegression()
    model5 = LinearRegression()
    model6 = LinearRegression()
    model7 = LinearRegression()
    model8 = KNeighborsRegressor(k)

    # Fit the models 

    model1.fit(
        insurance_train_train.bmi_adusted_for_age.values.reshape(-1,1), insurance_train_train.charges.values
    )
    model2.fit(
        insurance_train_train.bmi_age_interact.values.reshape(-1,1), insurance_train_train.charges.values
    )
    model3.fit(
        insurance_train_train.bmi_zscore.values.reshape(-1,1), insurance_train_train.charges.values
    )
    model4.fit(
        insurance_train_train[
            ['age','bmi_adusted_for_age','sex','children','region_northwest','region_southeast','region_southwest']
        ], insurance_train_train.charges.values
    )
    model5.fit(
        insurance_train_train[
            ['age','bmi','bmi_age_interact','sex','children','region_northwest','region_southeast','region_southwest']
        ], insurance_train_train.charges.values
    )
    model6.fit(
        insurance_train_train[
            ['age','bmi_zscore','sex','children','region_northwest','region_southeast','region_southwest']
        ], insurance_train_train.charges.values
    )
    model7.fit(
        insurance_train_train[
            ['bmi_adusted_for_age','bmi_age_interact','bmi_zscore','sex','children','region_northwest','region_southeast','region_southwest']
        ], insurance_train_train.charges.values
    )
    model8.fit(
        insurance_train_train[
            ['age','bmi','sex','children','region_northwest','region_southeast','region_southwest']
        ], insurance_train_train.charges.values
    )
    prediction1 = model1.predict(
        insurance_train_test.bmi_adusted_for_age.values.reshape(-1,1)
    )
    prediction2 = model2.predict(
        insurance_train_test.bmi_age_interact.values.reshape(-1,1)
    )
    prediction3 = model3.predict(
        insurance_train_test.bmi_zscore.values.reshape(-1,1)
    )
    prediction4 = model4.predict(
        insurance_train_test[['age','bmi_adusted_for_age','sex','children','region_northwest','region_southeast','region_southwest']]
    )
    prediction5 = model5.predict(
        insurance_train_test[
            ['age','bmi','bmi_age_interact','sex','children','region_northwest','region_southeast','region_southwest']
        ]
    )
    prediction6 = model6.predict(
        insurance_train_test[
            ['age','bmi_zscore','sex','children','region_northwest','region_southeast','region_southwest']
        ]
    )
    prediction7 = model7.predict(
        insurance_train_test[
            ['bmi_adusted_for_age','bmi_age_interact','bmi_zscore','sex','children','region_northwest','region_southeast','region_southwest']
        ]
    )
    prediction8 = model8.predict(
        insurance_train_test[
            ['age','bmi','sex','children','region_northwest','region_southeast','region_southwest']
        ]
    )
    
    mses[0,i] = mean_squared_error(insurance_train_test.charges.values, prediction0)
    mses[1,i] = mean_squared_error(insurance_train_test.charges.values, prediction1)
    mses[2,i] = mean_squared_error(insurance_train_test.charges.values, prediction2)
    mses[3,i] = mean_squared_error(insurance_train_test.charges.values, prediction3)
    mses[4,i] = mean_squared_error(insurance_train_test.charges.values, prediction4)
    mses[5,i] = mean_squared_error(insurance_train_test.charges.values, prediction5)
    mses[6,i] = mean_squared_error(insurance_train_test.charges.values, prediction6)
    mses[7,i] = mean_squared_error(insurance_train_test.charges.values, prediction7)
    mses[8,i] = mean_squared_error(insurance_train_test.charges.values, prediction8)
#
#
#
#| code-fold: false
#| code-overflow: wrap
plt.figure(figsize=(9.5,5))

plt.scatter(np.zeros(9), mses[0,:], s=60, c='white', edgecolors='black', label= 'Single Split')
plt.scatter(np.ones(9), mses[1,:], s=60, c='white', edgecolors='black')
plt.scatter(2*np.ones(9), mses[2,:], s=60, c='white', edgecolors='black')
plt.scatter(3*np.ones(9), mses[3,:], s=60, c='white', edgecolors='black')
plt.scatter(4*np.ones(9), mses[4,:], s=60, c='white', edgecolors='black')
plt.scatter(5*np.ones(9), mses[5,:], s=60, c='white', edgecolors='black')
plt.scatter(6*np.ones(9), mses[6,:], s=60, c='white', edgecolors='black')
plt.scatter(7*np.ones(9), mses[7,:], s=60, c='white', edgecolors='black')
plt.scatter(8*np.ones(9), mses[8,:], s=60, c='white', edgecolors='black')
plt.scatter([0,1,2,3,4,5,6,7,8], np.mean(mses, axis=1), s=60, c='r', marker='X', label='Mean')
plt.legend(loc='upper right',fontsize=12)
plt.xticks([0,1,2,3,4,5,6,7,8],['Baseline','Model 1','Model 2','Model 3','Model 4','Model 5','Model 6','Model 7', ' Model 8'])
plt.yticks(fontsize=10)
plt.ylabel('MSE', fontsize=12)
plt.gca().set_facecolor('#f4f4f4') 
plt.gcf().patch.set_facecolor('#f4f4f4')
plt.show()
#
#
#
#| code-fold: false
print(np.mean(np.sqrt(mses), axis=1))
# Minimum and the position
print('Minimum RMSE={} and Model {}'.format(min(),np.argmin(np.mean(np.sqrt(mses), axis=1))))
#
#
#
#
#
#| code-fold: false
model = LinearRegression()
model.fit(insurance_train[['age','bmi']],insurance_train.charges.values)
print("RMSE on the training set: ",
    np.round(np.sqrt(mean_squared_error(insurance_train.charges.values, model.predict(insurance_train[['age','bmi']]))),2)
)

print("RMSE on the test set: ",
    np.round(np.sqrt(mean_squared_error(insurance_test.charges.values, model.predict(insurance_test[['age','bmi']]))),2)
)
#
#
#
#
#
#
#
#
#
#