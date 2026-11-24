# Данные

- [`wholesale.csv`](wholesale.csv): [Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers), UCI (CC BY 4.0). Используется исходный `data.csv` из UCI, 440 покупателей. Цель `Channel`: 1 = Horeca (отели, рестораны, кафе), 2 = Retail. `Region` и шесть годовых категорий расходов доступны как признаки.
- [`sgemm.csv`](sgemm.csv): [SGEMM GPU kernel performance](https://archive.ics.uci.edu/dataset/440/sgemm+gpu+kernel+performance), Enrique G. Paredes и Rafael Ballester-Ripoll, UCI (CC BY 4.0). Используется случайная подвыборка из 3 000 из 241 600 уникальных конфигураций, чтобы запустить перебор и самописные ансамбли на CPU. Цель в ноутбуках: среднее время четырёх прогонов, мс. Четыре замера не входят в признаки; одна конфигурация целиком попадает в одну часть разбиения. Связанная работа: [Sobol Tensor Trains for Global Sensitivity Analysis](https://arxiv.org/abs/1712.00233).

Чтобы восстановить подвыборку, распакуйте `sgemm_product.csv` из [архива UCI](https://archive.ics.uci.edu/static/public/440/sgemm+gpu+kernel+performance.zip) и выполните в pandas:

```python
data = pd.read_csv('sgemm_product.csv')
data.sample(n=3000, random_state=20).sort_index().to_csv('data/sgemm.csv', index=False)
```
