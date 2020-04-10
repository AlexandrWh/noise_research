# Маленький проектик

## 1. Подмассивы

### Решение:
Рассмотрим префиксы входного массива **arr**, то есть суммы sum(a[:i]), i = 0,...,n-1 , где **n** - размер входного массива.
Поиск этих значений будет нам стоить О(n) времени. Для каждого такого значения нужно найти минимальное значение перед ним.
Это действие так же потребует O(n) времени, т.к. мы будем брать минимум из нового начения и текущего минимума.
И снова за за линейное время найдем пару, у которой разница самая большая.
Они и будет соответствовать наибольшему по сумме подмассиву исходного массива **arr**.
Подробности можно найти в файле **subarrays.py**.


## 2.1 Классификация мел-спектрограмм на предмет зашумленности

### Решение:
Даны мел-преобразования спектрограмм набора голосов. Т.к. мел-преобразование не позволяет нам без потерь вернуться к
изначальным сырым данным, то придется работать с этими признаками. Последовательный анализ этой задачи можно найти в
ноутбуке **EDA.ipynb**

Последним наиболее точным с точки зрения метрики **accuracy** оказался следующий подход из опробованных.
Т.к. спектрограммы имеют разный размер, алгоритм должен быть инваривантен размеру. Было рассмотрено 2 пути:
**1)** Сжать изображение до размера 80 на 80 и подать на вход сверточной сети
**2)** Тренировать ту же сеть со входом 80 на 80, только кормить ей вырезки из спектрограмм. А когда нужно получить
предсказание на инференсе, мы просканирует спектрограмму окном 80 на 80 с шагом, в качестве ответа возьмем усредненное
значение предиктов.

Нетрудно догадаться, что второй способ сложнее первого, однако он обладает юолбшей устойчивостью и немного опережает в
качестве. Был выбран второй подход. В среднем результат колеблется между **0.96** и **0.965** метрики **accuracy** для
валидации и тренировочного набора.

## 2.2 Шумодав
Из мел-пребразований зашумленных голосов нужно получить мед-преобразование "очищенного" голоса.
Задача решалась с помощью сверточной нейронной сети с **zero padding** для сохранения размера и дилатационных сверток
для повышения рецептивного поля нейрона. Результат вышел приятный как глазу, так и по цифрам. Метрика MSE достигла значения
0.043 на тренировочном и валидационном наборе.
