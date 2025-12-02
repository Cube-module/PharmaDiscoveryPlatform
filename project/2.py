import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder # для строковых значений из csv (это типо hot_encoder но для дерева лучше)
import numpy as np
# датасет искусственный поэтому логичной связи между некоторыми категориями нет
# оценку важности публикации получаем при составления датасета


# счит базу данных главную, ту что мы получаем из открытых источников
df = pd.read_csv("project/drug_candidates_fourth.csv")

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
    le_ind = LabelEncoder()
    le_target = LabelEncoder()
    
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
    le_ind = LabelEncoder()
    le_target = LabelEncoder()
    
    X["indication"] = le_ind.fit_transform(X["indication"])
    X["target"] = le_target.fit_transform(X["target"])
    
    # разбиваем train + val
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
        
    model = RandomForestClassifier(n_estimators=200, max_depth=30, criterion='entropy', random_state=42)
    model.fit(X_train, Y_train)
    
    # смотрим точность
    predict = model.predict(X_val)
    acuracy = accuracy_score(Y_val, predict) 
    print("точность ТОксичности: ", acuracy)
    
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


def move_data():
    """
    Функция изменяет df оставляя в нем обобщенные колонки эфективности и токсичности
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


def main():
  move_data()
  
  
    
    
if __name__ == "__main__":
    main()