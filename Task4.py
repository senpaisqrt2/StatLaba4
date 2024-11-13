import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
file_path = "Статистика.txt"
with open(file_path, 'r') as file:
    crime_ages = [int(line.strip()) for line in file]

ages_df = pd.DataFrame(crime_ages, columns=["Age"])

# Определяем интервалы возрастов
bins = np.arange(ages_df["Age"].min(), ages_df["Age"].max() + 9, 9)
labels = [f'{int(bins[i])}-{int(bins[i + 1]) - 1}' for i in range(len(bins) - 1)]
labels[-1] = f'{int(bins[-2])}-{int(bins[-1]) - 1}'  # Последний интервал

# Группировка данных по интервалам
ages_df['Age Group'] = pd.cut(ages_df['Age'], bins=bins - 1, labels=labels, include_lowest=True)
age_group_counts = ages_df['Age Group'].value_counts().sort_index()

# Сумма возрастов в каждой группе
grouped_age_sums = ages_df.groupby('Age Group')['Age'].sum()

# Общее кол-во
age_sums = age_group_counts.sum()

# Общее среднее
mean_value = ages_df['Age'].mean()

# Суммируем количество преступников в каждой группе
grouped_counts = ages_df.groupby('Age Group')['Age'].count()

# Вычисляем средний возраст в каждой группе
grouped_means = grouped_age_sums / grouped_counts

# Вычисляем сумму квадратов отклонений от среднего
grouped_variances = ages_df.groupby('Age Group').apply(
    lambda x: np.sum((x['Age'] - grouped_means[x['Age Group'].iloc[0]]) ** 2) / (x.shape[0] - 1)
)

# Промежуточные вычисления для внутригрупповой дисперсии
inside_group = []
for i in range(len(age_group_counts)):
    inside_group.append(grouped_variances[i] * age_group_counts[i] / age_sums)
    # print(inside_group[i])

# Внутригрупповая дисперсия
com_inside_group = sum(inside_group)

# Промежуточные вычисления для межгрупповой дисперсии
outside_group = []
for i in range(len(age_group_counts)):
    outside_group.append(((grouped_means[i] - mean_value) ** 2) * age_group_counts[i] / age_sums)
    # print(outside_group[i])

# Межгрупповая дисперсия
com_outside_group = sum(outside_group)

# Общая дисперсия
com_variances = com_inside_group + com_outside_group

# Корреляционное отношение
n_var = (com_outside_group / com_variances) ** (0.5)

# print(age_group_counts[0])

# print(mean_value)

# Печатаем результаты
print("Возрастные группы:\n", age_group_counts)
print("Средний возраст в каждой группе:\n", grouped_means)
print("Промежуточные значения для внутригрупповой дисперсии:\n", inside_group)
print("Внутригрупповая дисперсия:\n", com_inside_group)
print("Промежуточные значения для межгрупповой дисперсии:\n", outside_group)
print("Межрупповая дисперсия:\n", com_outside_group)
print("Общая дисперсия:\n", com_variances)
print("Корреляционное отношение:\n", n_var)
