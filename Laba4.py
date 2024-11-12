import numpy as np
from scipy.stats import pearsonr, t
import matplotlib.pyplot as plt

# Путь к файлу с исходными данными
data_file_path = '/Статистика.txt'


print('ЗАДАНИЕ 1')
# ЗАДАНИЕ 1

# Считываем данные из файла и создаем списки age и crimes
with open(data_file_path, 'r') as file:
    element = list(map(int, file.readlines()))

# Сортировка и расчет частоты для каждого возраста
element.sort()
age = list(range(min(element), max(element) + 1))
crimes = [element.count(i) for i in age]

# Преобразование списков в массивы для дальнейшего анализа
ages = np.array(age)
frequencies = np.array(crimes)

# Расчет коэффициента линейной корреляции
correlation_coefficient, _ = pearsonr(ages, frequencies)
print("Коэффициент линейной корреляции между возрастом и частотой преступлений:", correlation_coefficient)



print('')
print('')
print('ЗАДАНИЕ 2')

# Число наблюдений
n = len(ages)

# Расчет t-статистики для проверки значимости корреляции
t_statistic = correlation_coefficient * np.sqrt((n - 2) / (1 - correlation_coefficient ** 2))

# p-value для двустороннего теста
p_value = 2 * (1 - t.cdf(abs(t_statistic), df=n - 2))

# Уровень значимости
alpha = 0.05

# Результаты проверки значимости
print("t-статистика:", t_statistic)
print("p-value:", p_value)

# Оценка значимости
if p_value < alpha:
    print("Коэффициент корреляции значим на уровне 0.05.")
else:
    print("Коэффициент корреляции не значим на уровне 0.05.")



print('')
print('')
print('ЗАДАНИЕ 3 - на графике')

# Построение корреляционного поля
plt.scatter(ages, frequencies, color='orange', label='Данные')

# Добавление трендовой линии
z = np.polyfit(ages, frequencies, 1)  # Параметры для линейной регрессии
p = np.poly1d(z)
plt.plot(ages, p(ages), color='blue', linestyle='--', label='Линия тренда')

plt.title('Корреляционное поле: Возраст преступника vs Частота преступлений')
plt.xlabel('Возраст преступника')
plt.ylabel('Частота преступлений')
plt.legend()
plt.grid(True)
plt.show()


print('')
print('')
print('ЗАДАНИЕ 4(VERY sussy!!!)')

# Данные для интервалов и частот из лабораторной работы №3
mid_points = np.array([18.5, 27.5, 36.5, 45.5, 54.5, 63.5, 72.5])  # середины интервалов
age_group_counts = np.array([4811, 9476, 7243, 7951, 1140, 1472, 330])  # частоты для интервалов

# Общая сумма всех наблюдений (объем выборки)
n = age_group_counts.sum()

# Общая средняя (вычисленная на основе всех данных)
overall_mean = np.sum(mid_points * age_group_counts) / n

# Межгрупповая дисперсия
between_group_variance = np.sum(age_group_counts * (mid_points - overall_mean) ** 2) / n

# Общая дисперсия (используем методику из лекции, разделяя межгрупповую и внутригрупповую)
total_variance = np.sum(age_group_counts * (mid_points - overall_mean) ** 2) / n

# Корреляционное отношение
eta_squared = between_group_variance / total_variance
eta = np.sqrt(eta_squared)

print("Корреляционное отношение (eta):", eta)
print("Квадрат корреляционного отношения (eta^2):", eta_squared)