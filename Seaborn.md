# Seaborn

# 

## 타이타닉 데이터를 통해 예제 형식 표현

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
df = pd.read_csv("titanic.csv")
```

# 1. Countplot 명목형 변수에 사용

항목별 갯수를 세어주는 countplot

알아서 해당 column을 구성하고 있는 value들을 구분하여 보여준다.

```python
sns.countplot(x="class", hue="who", data=titanic, palette='Paired')
plt.show()

# 배경을 darkgrid 로 설정 
sns.set(style='darkgrid')

for col in categorical_columns:
    sns.countplot( x = col , data = df, palette='Paired' ,)
    

    plt.show()
```

![스크린샷 2021-08-25 오후 4.18.28.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3aade21f-4e94-4779-bc99-d95cf0fb1540/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.18.28.png)
![1.png](/img/1.png)

![스크린샷 2021-08-25 오후 4.18.39.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/03e90f1f-c82b-4082-8e7f-8eba8c03dc2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.18.39.png)

![스크린샷 2021-08-25 오후 4.18.51.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/322ccb84-7217-4761-a6cc-c4c588ccff72/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.18.51.png)

위와 같은 결과로 모든 변수가 표현됨.

## hue 옵션을 통해 groupby 기능 수행

```python
# hue 옵션을 통해 groupby를 따로 사용하지 않고도 여러 변수를 같이 볼 수 있다.
sns.countplot(x = 'Sex' , hue = 'Survived', data = df )
```

![스크린샷 2021-08-25 오후 4.20.05.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/170f73f9-4707-4d02-8993-0e69b072c63a/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.20.05.png)

# 2. distplot 수치형 변수에 단변수 탐색에 사용[¶](http://localhost:8889/lab/tree/Desktop/코드/seaborn 코드 모음.ipynb#2.-distplot-수치형-변수에-단변수-탐색에-사용)

matplotlib의 hist 그래프와 kdeplot을 통합한 그래프 입니다.

분포와 밀도를 확인할 수 있습니다.

```python
for col in numerical_columns:
    sns.distplot(df.loc[df[col].notnull(), col])
    plt.show()
```

![스크린샷 2021-08-25 오후 4.21.32.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4dbd8aef-963a-4874-92ac-5e653965b37b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.21.32.png)

![스크린샷 2021-08-25 오후 4.21.47.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f9c09956-ffd2-4256-88f2-56703181a97b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.21.47.png)

![스크린샷 2021-08-25 오후 4.22.24.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f48b3168-ee7f-47e1-8d04-1eece403c706/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.22.24.png)

위와 같은 형태로 모든 변수에 대해 표현됨.

```python
# 데이터가 Series 일 경우
x = np.random.randn(100)
x = pd.Series(x, name="x variable")
sns.distplot(x)
plt.show()
# rugplot : rug는 rugplot이라고도 불리우며, 데이터 위치를 x축 위에 작은 선분(rug)으로 나타내어 데이터들의 위치 및 분포를 보여준다.
sns.distplot(x, rug=True, hist=False)
plt.show()
# kde (kernel density) : kde는 histogram보다 부드러운 형태의 분포 곡선을 보여주는 방법
ns.distplot(x, rug=False, hist=False, kde=True, color="y")
plt.show()
```

# 3. pairplot : 수치형 변수 다변수 탐색에 사용 .. 선형이면 상관관계가 있다

pairplot은 그리도(grid) 형태로 각 집합의 조합에 대해 히스토그램과 분포도를 그립니다.

또한, 숫자형 column에 대해서만 그려줍니다.

```python
sns.pairplot(data = df, vars = numerical_columns,  hue='Sex', palette="rainbow", height=5,)
plt.show()
```

![스크린샷 2021-08-25 오후 4.23.37.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f0665c5-9ff7-4b59-8902-767fa0599a52/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.23.37.png)

(잘린화면)

# 4. heatmap

색상으로 표현할 수 있는 다양한 정보를 일정한 이미지위에 열분포 형태의 비쥬얼한 그래픽으로 출력하는 것이 특징이다

```python
pivot = tips.pivot_table(index='day', columns='size', values='tip')

sns.heatmap(pivot, annot=True)
plt.show()

#  correlation(상관관계)를 시각화
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()
```

![스크린샷 2021-08-25 오후 4.24.28.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2501ed5-4ef0-47a1-bd67-00a12171adeb/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.24.28.png)

# 5. violinplot

바이올린처럼 생긴 violinplot 입니다.

column에 대한 데이터의 비교 분포도를 확인할 수 있습니다.

- 곡선진 부분 (뚱뚱한 부분)은 데이터의 분포를 나타냅니다.
- 양쪽 끝 뾰족한 부분은 데이터의 최소값과 최대값을 나타냅니다.

```python
# 비교 분포 확인 : x, y축을 지정해줌으로썬 바이올린을 분할하여 비교 분포를 볼 수 있습니다

sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted")
plt.show()
```

# 6. lmplot : lmplot은 column 간의 선형관계를 확인하기에 용이한 차트. 두 수치형 변수간의 관계를 볼 때 사용

또한, outlier도 같이 짐작해 볼 수 있습니다.

```python
sns.lmplot(x="total_bill", y="tip", height=8, data=tips)
plt.show()

# 다중 선형
sns.lmplot(x="total_bill", y="tip", hue="smoker", height=8, data=tips)
plt.show()

# 또한, col_wrap으로 한 줄에 표기할 column의 갯수를 명시할 수 있습니다.
```

# 7. relplot : 두 column간 상관관계를 보지만 lmplot처럼 선형관계를 따로 그려주지는 않습니다.¶

```python
sns.relplot(x="total_bill", y="tip", hue="day", col="time", data=tips)
plt.show()

# row와 column에 표기할 데이터 column 선택
sns.relplot(x="total_bill", y="tip", hue="day", row="sex", col="time", data=tips)
plt.show()
```

# 8. jointplot : scatter(산점도)와 histogram(분포)을 동시에 그려줍니다.

숫자형 데이터만 표현 가능하니, 이 점 유의하세요.

```python
sns.jointplot("total_bill", "tip", height=8, data=tips, kind="reg")
plt.show()

# 옵션에 kind='reg'을 추가해 선형관계를 표현하는 regression 라인 그리기
# kind='hex' 옵션을 통해 hex 모양의 밀도를 확인할 수 있습니다.
# kind='kde' 옵션으로 등고선 모양, 데이터의 밀집도를 보다 부드러운 선으로 확인할 수 있습니ㅏ.
```

# 9. barplot, pointplot : 막대 상자, 꺾은선 그래프. 단지 countplot 말고 계절별 평균 값 등으로 구할 때 사용해라

```python
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,1,1)
ax1 = sns.pointplot(x='month',y='count',hue='weather',data=df.groupby(['weather','month'])['count'].mean().reset_index())
```

# 10. boxplot . 수치형 데이터, 혹은 범주형-수치형 데이터를 확인할 때 사용

```python
sns.boxplot(x = "day", y = "total_bill",  hue = "smoker", data = tips)

plt.show()
```



타이타닉 데이터를 통해 예제 형식 표현

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
df = pd.read_csv("titanic.csv")
```

# 1. Countplot 명목형 변수에 사용

항목별 갯수를 세어주는 countplot

알아서 해당 column을 구성하고 있는 value들을 구분하여 보여준다.

```python
sns.countplot(x="class", hue="who", data=titanic, palette='Paired')
plt.show()

# 배경을 darkgrid 로 설정 
sns.set(style='darkgrid')

for col in categorical_columns:
    sns.countplot( x = col , data = df, palette='Paired' ,)
    

    plt.show()
```

![스크린샷 2021-08-25 오후 4.18.28.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3aade21f-4e94-4779-bc99-d95cf0fb1540/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.18.28.png)

![스크린샷 2021-08-25 오후 4.18.39.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/03e90f1f-c82b-4082-8e7f-8eba8c03dc2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.18.39.png)

![스크린샷 2021-08-25 오후 4.18.51.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/322ccb84-7217-4761-a6cc-c4c588ccff72/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.18.51.png)

위와 같은 결과로 모든 변수가 표현됨.

## hue 옵션을 통해 groupby 기능 수행

```python
# hue 옵션을 통해 groupby를 따로 사용하지 않고도 여러 변수를 같이 볼 수 있다.
sns.countplot(x = 'Sex' , hue = 'Survived', data = df )
```

![스크린샷 2021-08-25 오후 4.20.05.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/170f73f9-4707-4d02-8993-0e69b072c63a/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.20.05.png)

# 2. distplot 수치형 변수에 단변수 탐색에 사용[¶](http://localhost:8889/lab/tree/Desktop/코드/seaborn 코드 모음.ipynb#2.-distplot-수치형-변수에-단변수-탐색에-사용)

matplotlib의 hist 그래프와 kdeplot을 통합한 그래프 입니다.

분포와 밀도를 확인할 수 있습니다.

```python
for col in numerical_columns:
    sns.distplot(df.loc[df[col].notnull(), col])
    plt.show()
```

![스크린샷 2021-08-25 오후 4.21.32.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4dbd8aef-963a-4874-92ac-5e653965b37b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.21.32.png)

![스크린샷 2021-08-25 오후 4.21.47.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f9c09956-ffd2-4256-88f2-56703181a97b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.21.47.png)

![스크린샷 2021-08-25 오후 4.22.24.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f48b3168-ee7f-47e1-8d04-1eece403c706/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.22.24.png)

위와 같은 형태로 모든 변수에 대해 표현됨.

```python
# 데이터가 Series 일 경우
x = np.random.randn(100)
x = pd.Series(x, name="x variable")
sns.distplot(x)
plt.show()
# rugplot : rug는 rugplot이라고도 불리우며, 데이터 위치를 x축 위에 작은 선분(rug)으로 나타내어 데이터들의 위치 및 분포를 보여준다.
sns.distplot(x, rug=True, hist=False)
plt.show()
# kde (kernel density) : kde는 histogram보다 부드러운 형태의 분포 곡선을 보여주는 방법
ns.distplot(x, rug=False, hist=False, kde=True, color="y")
plt.show()
```

# 3. pairplot : 수치형 변수 다변수 탐색에 사용 .. 선형이면 상관관계가 있다

pairplot은 그리도(grid) 형태로 각 집합의 조합에 대해 히스토그램과 분포도를 그립니다.

또한, 숫자형 column에 대해서만 그려줍니다.

```python
sns.pairplot(data = df, vars = numerical_columns,  hue='Sex', palette="rainbow", height=5,)
plt.show()
```

![스크린샷 2021-08-25 오후 4.23.37.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1f0665c5-9ff7-4b59-8902-767fa0599a52/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.23.37.png)

(잘린화면)

# 4. heatmap

색상으로 표현할 수 있는 다양한 정보를 일정한 이미지위에 열분포 형태의 비쥬얼한 그래픽으로 출력하는 것이 특징이다

```python
pivot = tips.pivot_table(index='day', columns='size', values='tip')

sns.heatmap(pivot, annot=True)
plt.show()

#  correlation(상관관계)를 시각화
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()
```

![스크린샷 2021-08-25 오후 4.24.28.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2501ed5-4ef0-47a1-bd67-00a12171adeb/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2021-08-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.24.28.png)

# 5. violinplot

바이올린처럼 생긴 violinplot 입니다.

column에 대한 데이터의 비교 분포도를 확인할 수 있습니다.

- 곡선진 부분 (뚱뚱한 부분)은 데이터의 분포를 나타냅니다.
- 양쪽 끝 뾰족한 부분은 데이터의 최소값과 최대값을 나타냅니다.

```python
# 비교 분포 확인 : x, y축을 지정해줌으로썬 바이올린을 분할하여 비교 분포를 볼 수 있습니다

sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, palette="muted")
plt.show()
```

# 6. lmplot : lmplot은 column 간의 선형관계를 확인하기에 용이한 차트. 두 수치형 변수간의 관계를 볼 때 사용

또한, outlier도 같이 짐작해 볼 수 있습니다.

```python
sns.lmplot(x="total_bill", y="tip", height=8, data=tips)
plt.show()

# 다중 선형
sns.lmplot(x="total_bill", y="tip", hue="smoker", height=8, data=tips)
plt.show()

# 또한, col_wrap으로 한 줄에 표기할 column의 갯수를 명시할 수 있습니다.
```

# 7. relplot : 두 column간 상관관계를 보지만 lmplot처럼 선형관계를 따로 그려주지는 않습니다.¶

```python
sns.relplot(x="total_bill", y="tip", hue="day", col="time", data=tips)
plt.show()

# row와 column에 표기할 데이터 column 선택
sns.relplot(x="total_bill", y="tip", hue="day", row="sex", col="time", data=tips)
plt.show()
```

# 8. jointplot : scatter(산점도)와 histogram(분포)을 동시에 그려줍니다.

숫자형 데이터만 표현 가능하니, 이 점 유의하세요.

```python
sns.jointplot("total_bill", "tip", height=8, data=tips, kind="reg")
plt.show()

# 옵션에 kind='reg'을 추가해 선형관계를 표현하는 regression 라인 그리기
# kind='hex' 옵션을 통해 hex 모양의 밀도를 확인할 수 있습니다.
# kind='kde' 옵션으로 등고선 모양, 데이터의 밀집도를 보다 부드러운 선으로 확인할 수 있습니ㅏ.
```

# 9. barplot, pointplot : 막대 상자, 꺾은선 그래프. 단지 countplot 말고 계절별 평균 값 등으로 구할 때 사용해라

```python
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,1,1)
ax1 = sns.pointplot(x='month',y='count',hue='weather',data=df.groupby(['weather','month'])['count'].mean().reset_index())
```

# 10. boxplot . 수치형 데이터, 혹은 범주형-수치형 데이터를 확인할 때 사용

```python
sns.boxplot(x = "day", y = "total_bill",  hue = "smoker", data = tips)

plt.show()
```
