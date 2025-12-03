import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder # для строковых значений из csv (это типо hot_encoder но для дерева лучше)
import numpy as np
# датасет искусственный поэтому логичной связи между некоторыми категориями нет
# оценку важности публикации получаем при составления датасета

# нужно чтобы энкодер не падал если находим неизвестный элемент и записывает -1
class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        y = pd.Series(y)
        known = set(self.classes_)
        return y.apply(
            lambda v: super(SafeLabelEncoder, self).transform([v])[0] if v in known else -1
        ).values



# счит базу данных главную, ту что мы получаем из открытых источников
df = pd.read_csv("project/drug_candidates_five.csv")

# нужно модель обучить на метриках для нахождения эффективности
def efectivity_model():
    """
    Принимает: 
    Возвращает: обученную модель предсказывать эфективность и её энкодеры
    Функция вычесляющая эфективность и печатающая точность предсказания 
    """
    dfe = pd.read_csv("project/synthetic_indication_target_eff2.csv")
    
    # разбиваем данные для обучения
    Y = dfe['efficacy_label']
    X = dfe.drop(columns= ['efficacy_label'])
    
    # преобразуем строковые значения из csv в индексный вид с помощью LabelEncoder
    le_ind = SafeLabelEncoder()
    le_target = SafeLabelEncoder()
    
    X["indication"] = le_ind.fit_transform(X["indication"])
    X["target"] = le_target.fit_transform(X["target"])
    
    # разбиваем train + val
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=30, criterion='entropy', random_state=42)
    model.fit(X_train, Y_train)
    
    # смотрим точность
    predict = model.predict(X_val)
    acuracy = accuracy_score(Y_val, predict) 
    print("точность Эффективности: ", acuracy)
    
    return model, le_ind, le_target


def build_efficiency_score():
    """
    Комбинирует предсказания модели эффективности (df['efective'])
    и текстовый признак df['has_positive_efficacy_phrase'] в один столбец.
    Результат: df['efficiency_score'] со значениями 0/1/2.
    """
    # нормируем модельный скор 0/1/2 -> 0..1
    model_norm = df["efective_model"] / 2.0

    # приводим текстовый к float (0.0 или 1.0)
    text_norm = df["has_positive_efficacy_phrase"]

    # веса можно потом подкрутить
    alpha = 0.5  # вес модели
    beta = 0.5   # вес текста

    combined = alpha * model_norm + beta * text_norm

    # режем на 3 класса: 0, 1, 2
    bins = [-0.1, 0.33, 0.66, 1.01]
    labels = [0, 1, 2]
    df["efficiency_score"] = pd.cut(combined, bins=bins, labels=labels).astype(int)
    print("Эффективность препаратов подсчитана")
    

def toxity_model():  
    """
    Разбивает данные на test, val для разных целей (potential_label, toxicity_label)
    Принимает: 
    Возвращает: обученную модель предсказывать токсичность и её энкодеры 
    Функция вычесляющая токсичность и печатающая точность предсказания 
    """
    dft = pd.read_csv("project/toxicity_training_dataset3.csv")
    
    # разбиваем наши цели предсказывания
    Y = dft["toxicity_label"]
    X = dft.drop(columns=["toxicity_label"])                    
    
    # преобразуем строковые значения из csv в индексный вид с помощью LabelEncoder
    le_ind = SafeLabelEncoder()
    le_target = SafeLabelEncoder()
    
    X["indication"] = le_ind.fit_transform(X["indication"])
    X["target"] = le_target.fit_transform(X["target"])
    
    # разбиваем train + val
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
        
    model = RandomForestClassifier(n_estimators=200, max_depth=30, criterion='entropy', random_state=42)
    model.fit(X_train, Y_train)
    
    # смотрим точность
    predict = model.predict(X_val)
    acuracy = accuracy_score(Y_val, predict) 
    print("точность Токсичности: ", acuracy)
    
    return model, le_ind, le_target
    
def build_toxicity_score():
    """
    Комбинирует вероятностное предсказание модели токсичности (0..1)
    и текстовый флаг токсичности (0/1) в один столбец toxicity_score (0/1/2).
    """

    # модельный скор уже 0..1
    model_norm = df["toxic_model"]

    # текстовый флаг
    text_norm = df["has_severe_toxicity_phrase"].astype(float)

    # веса
    alpha = 0.5
    beta = 0.5

    combined = alpha * model_norm + beta * text_norm

    # превращаем в 0,1,2
    bins = [-0.1, 0.33, 0.66, 1.01]
    labels = [0, 1, 2]

    df["toxicity_score"] = pd.cut(combined, bins=bins, labels=labels).astype(int)
    print("Токсичнасть препаратов подсчитана")
    

def build_uniqueness_score():
    """
    Вычисляет итоговый показатель уникальности (0..1) 
    В df добавляется только df['uniqueness_score'].
    """

    # Популярность target (
    target_pop = df["target"].map(df["target"].value_counts())
    target_pop_norm = target_pop / target_pop.max()

    # Популярность indication 
    ind_pop = df["indication"].map(df["indication"].value_counts())
    ind_pop_norm = ind_pop / ind_pop.max()

    # "Новизна" текста 
    publication_uniqueness = 1 - df["text_embed_score"]

    # Итоговая метрика (0..1)
    df["uniqueness_score"] = (
        0.4 * (1 - target_pop_norm) +
        0.4 * (1 - ind_pop_norm) +
        0.2 * publication_uniqueness
    )


def rating():
    """
    Вычисляет итоговый показатель потенциала (0..1)
    и добавляет df['score'] — категорию 0/1/2.
    """
    # итоговый балл (0..1)
    final_score = (
        0.5 * (df["efficiency_score"] / 2) +
        0.2 * (1 - df["toxicity_score"] / 2) +
        0.3 * df["uniqueness_score"]
    )
    # классификация
    df["score"] = pd.cut(
        final_score,
        bins=[-0.1, 0.33, 0.66, 1.01],
        labels=[0, 1, 2]
    ).astype(int)
    
    print("Потенциал препаратов подсчитан")
    

def move_data():
    """
    Функция изменяет df оставляя в нем обобщенные колонки эфективности, токсичности, уникальности, потенциала
    Общяая функция вызова функций 2 блока
    """
    # обучаем модели и получаем энкодеры
    model_e, le_ind_e, le_target_e = efectivity_model()
    model_t, le_ind_t, le_target_t = toxity_model()
    
    # готовим данные для эффективности
    X_e = df[['indication', 'target']].copy()
    X_e["indication"] = le_ind_e.transform(X_e["indication"])
    X_e["target"]     = le_target_e.transform(X_e["target"])

    # готовим данные для токсичности
    X_t = df[['indication', 'target', 'molecular_weight', 'logP']].copy()
    X_t["indication"] = le_ind_t.transform(X_t["indication"])
    X_t["target"]     = le_target_t.transform(X_t["target"])

    # предсказываем
    predict_e = model_e.predict(X_e)        # 0/1/2
    predict_t = model_t.predict(X_t)        # 0/1 (если RandomForestClassifier)

    # записываем в df
    df["efective_model"] = predict_e.astype(int)
    df["toxic_model"] = predict_t.astype(int)
    
    # добавляем в df обобщенные колонки эфектив., токсич., уникал., потенциал
    build_efficiency_score()
    build_toxicity_score()
    build_uniqueness_score()
    rating()
    
    # удаляем из df колонки эфект. и токс. не обобщенные
    df.drop(columns=["toxic_model", "efective_model", "has_severe_toxicity_phrase", "has_positive_efficacy_phrase"], inplace=True)

    


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 3 блок функции

def time_model():
    """
    Принимает: 
    Возвращает: обученную модель предсказывать обычное время разработки и её энкодеры
    Функция вычесляющая эфективность и печатающая точность предсказания 
    """
    df_time = pd.read_csv("project/time_training_dataset.csv")
    
    Y = df_time["traditional_time_years"]
    X = df_time.drop(columns=["traditional_time_years"])

    # преобразуем строковые значения из csv в индексный вид с помощью LabelEncoder
    le_ind = SafeLabelEncoder()
    le_target = SafeLabelEncoder()
    
    X["indication"] = le_ind.fit_transform(X["indication"])
    X["target"] = le_target.fit_transform(X["target"])
    
    # разбиваем train + val
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42)
    model.fit(X_train, Y_train)
    
    # смотрим погрешность
    predict = model.predict(X_val)
    mae = mean_absolute_error(Y_val, predict)
    print(f"Погрешность предсказания времени: {mae:.2f}")
    
    return model, le_ind, le_target


def ai_time():
    """
    Функция изменяет df, добавляя колонку traditional_time_years
    Вычисляет примерные сроки разработки, ускоренные AI.
    AI снижает сроки в 5–10 раз, но не меньше 1 года.
    """
    # берём случайный коэффициент ускорения: от 5 до 10
    accel = np.random.uniform(5, 10)

    # основная формула
    ai_t = df["traditional_time_years"] / accel

    # ограничиваем разумный минимум
    ai_t = ai_t.clip(lower=1.0)

    # округляем до сотых, чтобы выглядело красиво
    df["ai_time_years"] = ai_t.round(2)
    print("Время ускоренное AI подсчитано")


def cost_model():
    df_cost = pd.read_csv("dev_cost_training_dataset.csv")
    
    Y = df_cost["dev_cost_million_usd"]
    X = df.drop(columns=["dev_cost_million_usd"])
    
    le_ind = SafeLabelEncoder()
    le_target = SafeLabelEncoder()
    
    X["indication"] = le_ind.fit_transform(X["indication"])
    X["target"] = le_target.fit_transform(X["target"])
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42)
    model.fit(X_train, Y_train)
    
    # смотрим погрешность
    predict = model.predict(X_val)
    mae = mean_absolute_error(Y_val, predict)
    print(f"Погрешность предсказания стоимости: {mae:.2f}")
    
    return model, le_ind, le_target
    


def move_data_2():
    """
    Функция изменяет df, общяая функция вызова функций 3 блока
    """
    # записываем в df обычное время
    model_t, le_ind_t, le_target_t = time_model()
    
    X_t = df[['indication', 'target', 'molecular_weight', 'logP']].copy()
    X_t["indication"] = le_ind_t.transform(X_t["indication"])
    X_t["target"]     = le_target_t.transform(X_t["target"])
    
    predict_t = model_t.predict(X_t) 
    
    df["traditional_time_years"] = predict_t.astype(float)
    print("Время обычной разработки подсчитана")
    
    # записываем в df AI время
    ai_time()
    
    
    
    
    # записываем в df стоимость разработки
    model_c, le_ind_c, le_target_c = cost_model()
    
    X_c = df[["indication","target","molecular_weight","logP","market_size_million","competition_level","efficiency_score","toxicity_score","traditional_time_years"]].copy()
    X_c["indication"] = le_ind_t.transform(X_t["indication"])
    X_c["target"]     = le_target_t.transform(X_t["target"])
    
    predict_c = model_c.predict(X_c)
    
    cost = predict_t.astype(float)
    
    # округляем до сотых, чтобы выглядело красиво
    df["dev_cost_million_usd"] = cost.round(2)
    print("Время обычной разработки подсчитана")
    
    
    
    
    
    
    

def main():
  move_data()
  move_data_2()
  
  
  
  
  
if __name__ == "__main__":
    main()