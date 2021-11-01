# EDA 기초 예제

타이타닉으로 예제 수행

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
titanic = pd.read_csv("titanic.csv")
```

# 결측치 확인 df.isnull().sum(), 이상치 확인[¶](http://localhost:8889/lab/tree/Desktop/%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3/%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3%E1%84%86%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%B7/%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA%20EDA.ipynb#chapter-3.-%EA%B2%B0%EC%B8%A1%EC%B9%98-%ED%99%95%EC%9D%B8-df.isnull().sum(),-%EC%9D%B4%EC%83%81%EC%B9%98-%ED%99%95%EC%9D%B8)

null 값이 존재한다면 1. 제거하는 방법 complete data analysis 2. 다른 값으로 대치하는 방법 Imputation 두가지를 선택해서 행해야 한다.

```python
titanic.isnull().sum().plot(kind='bar')
```

![스크린샷 2021-08-25 오후 4.28.56.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.28.56.png)

```python
k = titanic.isnull().sum().reset_index()
k.columns = ['column', 'count']
k['ratio'] = k['count'] / titanic.shape[0]
K
```

![스크린샷 2021-08-25 오후 4.29.35.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.29.35.png)

## 종속 변수 체크

```python
titanic['Survived'].value_counts().plot(kind='bar')
```

![스크린샷 2021-08-25 오후 4.30.09.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.30.09.png)

## 명목형 변수들 살펴보기

```python
categorical_columns = [column for column in titanic.columns if titanic[column].dtypes == 'object']
categorical_columns
```

```python
categorical_columns = list(set(categorical_columns) - set(['Survived']))
categorical_columns
```

```python
for col in categorical_columns:
    titanic[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()
```

![스크린샷 2021-08-25 오후 4.30.51.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.30.51.png)

![스크린샷 2021-08-25 오후 4.31.02.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.31.02.png)

![스크린샷 2021-08-25 오후 4.31.13.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.31.13.png)

## 이변수 탐색.. 두 변수 간의 관계

단변수

```python
2
```

![스크린샷 2021-08-25 오후 4.32.10.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.32.10.png)

```python
tes = titanic.groupby(by=['Sex','Survived'])['Survived'].count().unstack('Survived')
tes
```

![스크린샷 2021-08-25 오후 4.32.29.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.32.29.png)

```python
tes.plot(kind='bar', legend=True, grid=True, title='Seg')
```

![스크린샷 2021-08-25 오후 4.32.57.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.32.57.png)

## 수치형 변수 탐색

```python
numerical_columns = list(set(titanic.columns) - set(categorical_columns) - set(['PassengerId','Survived']))
numerical_columns = np.sort(numerical_columns)
numerical_columns
>>>
array(['Age', 'Fare', 'Parch', 'SibSp'], dtype='<U5')
```

```python
for col in numerical_columns:
    sns.distplot(titanic.loc[titanic[col].notnull(), col])
    plt.grid()
    plt.legend()
    plt.title(col)
    plt.show()
```

![스크린샷 2021-08-25 오후 4.33.38.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.33.38.png)

![스크린샷 2021-08-25 오후 4.33.50.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.33.50.png)

## 다변수 탐색

```python
sns.pairplot(titanic[list(numerical_columns) + ['Survived']], hue='Survived', 
             x_vars=numerical_columns, y_vars=numerical_columns)
plt.show()
```

![스크린샷 2021-08-25 오후 4.34.30.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.34.30.png)

# 수치형, 명목형 변수 간의 관계 탐색[¶](http://localhost:8889/lab/tree/Desktop/%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3/%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3%E1%84%86%E1%85%A9%E1%84%8B%E1%85%B3%E1%86%B7/%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%A8%E1%84%92%E1%85%AA%20EDA.ipynb#chapter7.-%EC%88%98%EC%B9%98%ED%98%95,-%EB%AA%85%EB%AA%A9%ED%98%95-%EB%B3%80%EC%88%98-%EA%B0%84%EC%9D%98-%EA%B4%80%EA%B3%84-%ED%83%90%EC%83%89)

앞서서 수치형-수치형 간의 관계, 그리고 명목형-명목형 간의 관계에 종속변수까지 포함해서 보았습니다. 이 번에는 수치형-명목형 간의 관계를 파악해 보는 것입니다. 예를 들어, 성별, 나이, 생존여부 3개의 변수를 동시에 탐색하고 싶을 수 있습니다. 이 경우에 명목형 변수에 따라 수치형변수의 boxplot을 그려봄으로써 대략적인 데이터의 형태를 살펴볼 수 있습니다.

```python
unique_list = titanic['Sex'].unique()
 
for col in numerical_feature:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Sex', y=col, hue='Survived', data=titanic.dropna())
    plt.title("Sex - {}".format(col))
    plt.show()
```

![스크린샷 2021-08-25 오후 4.37.30.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.37.30.png)

![스크린샷 2021-08-25 오후 4.37.51.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.37.51.png)

![스크린샷 2021-08-25 오후 4.38.05.png](EDA%20%E1%84%80%E1%85%B5%E1%84%8E%E1%85%A9%20%E1%84%8B%E1%85%A8%E1%84%8C%E1%85%A6%203e8ec11413c54d73956ad6d3571647c5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.38.05.png)

# EDA 데이터 분포 확인

먼저 회귀분석, 상관분석 등으로 중요 변수 후보를 추린다.

## < 클래스의 각 값에 따른 특정 컬럼의 분포 차이 확인 >

확인 요소 1. 분포가 다르면 중요한 변수가 되는 것이다.

확인 요소 2. boxplot으로 보면 시각적으로 이상치들의 존재를 확인 가능하다..

```python
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()
```

## 각 컬럼의 데이터 분포 확인

```python
from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()
```