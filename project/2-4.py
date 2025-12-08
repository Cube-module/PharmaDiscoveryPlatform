import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder # для строковых значений из csv (это типо hot_encoder но для дерева лучше)
import numpy as np
import json
from dotenv import load_dotenv
from openai import OpenAI
import os
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
    df_cost = pd.read_csv("project/dev_cost_training_dataset.csv")
    
    Y = df_cost["dev_cost_million_usd"]
    X = df_cost.drop(columns=["dev_cost_million_usd"])
    
    le_ind = SafeLabelEncoder()
    le_target = SafeLabelEncoder()
    
    X["indication"] = le_ind.fit_transform(X["indication"])
    X["target"] = le_target.fit_transform(X["target"])
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.30, random_state=42)
    
    model = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42)
    model.fit(X_train, Y_train)
    
    # смотрим погрешность
    predict = model.predict(X_val)
    mae = mean_absolute_error(Y_val, predict)
    print(f"Погрешность предсказания стоимости: {mae:.2f}")
    
    return model, le_ind, le_target
    
    
def competition_scores():
    """
    Преобразуем данные о возможном заработке в оценку целевого рынка 0/1/2
    """
    # нормализация
    X = (
    (df["market_size_million"] - df["market_size_million"].min())
    / (df["market_size_million"].max() - df["market_size_million"].min())
)
    
    # классификация создаем новы столбец df
    df["competition_score"] = pd.cut(
        X,
        bins=[-0.1, 0.33, 0.66, 1.01],
        labels=[0, 1, 2]
    ).astype(int)
    
    print("Размер целевого рынка подсчитан")
    
    
def build_competition_level():
    """
    Оценка уровня конкуренции по комбинации indication+target.
    Значения: 0/1/2 — низкая / средняя / высокая конкуренция.
    Обратная логика с уникальностью п оценке
    """
    # создаём пару
    pair = df["indication"].astype(str) + "__" + df["target"].astype(str)

    # частота появления аналогичных препаратов
    freq = pair.map(pair.value_counts())

    # нормализация 0..1
    norm = (freq - freq.min()) / (freq.max() - freq.min())

    # категория 0/1/2
    df["competition_level"] = pd.cut(
        norm,
        bins=[-0.1, 0.33, 0.66, 1.01],
        labels=[0, 1, 2]
    ).astype(int)

    print("Уровень конкуренции подсчитан")
   
    
def build_market_share():
    """
    Строит прогноз доли рынка (0..1) на основе эффективности, токсичности,
    уникальности и уровня конкуренции.
    market_share = какая часть рынка достанется конкретному препарату
    """
    eff = df["efficiency_score"] / 2   # 0..1
    tox = df["toxicity_score"] / 2     # 0..1
    uniq = df["uniqueness_score"]      # 0..1
    comp = df["competition_score"] / 2 # 0..1

    # Базовая формула
    market_share = (
        0.2 * eff +
        0.2 * (1 - tox) +
        0.3 * uniq +
        0.2 * (1 - comp) +
        0.1
    )

    # обрезаем 0..1
    market_share = market_share.clip(0, 1)

    df["market_share"] = market_share.round(3)

    print("Доля рынка подсчитана")
    
    
def potential_profit():
    """
    Вычисляет:
    - df['potential_profit_million'] — потенциальную прибыль (млн)
    - df['profit_score']        — категорию прибыльности 0/1/2
    Формула: прибыль = размер рынка * доля рынка – стоимость разработки.
    """
    # 1. Доля рынка (0..1) как функция конкуренции, эффективности, токсичности и уникальности
    comp_norm = (2 - df["competition_level"]) / 2.0      # 1 = низкая конкуренция
    eff_norm  = df["efficiency_score"] / 2.0             # 0..1
    tox_norm  = 1 - df["toxicity_score"] / 2.0           # 1 = низкая токсичность
    uniq_norm = df["uniqueness_score"]                   # 0..1


    build_market_share()

    # 2. Потенциальная прибыль (в млн)
    df["potential_profit_million"] = (
        df["market_size_million"] * df["market_share"]
        - df["dev_cost_million"]
    ).round(2)

    # 3. Классификация прибыльности в 3 класса
    profits = df["potential_profit_million"]

    # чтобы пороги были не руками, а по распределению
    low, high = np.percentile(profits, [33, 66])

    df["profit_score"] = pd.cut(
        profits,
        bins=[-np.inf, low, high, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    print("Потенциальная прибыльность подсчитана")
    

def clinical_trial_probabilities():
    """
    Добавляет в df вероятности успеха клинических фаз I–III.
    Использует уже рассчитанные метрики: эффективность, токсичность,
    уникальность, размер рынка, уровень конкуренции.
    """
    X = df["market_size_million"]
    
    # нормализуем 0..1
    eff = df["efficiency_score"] / 2.0          # 0..1
    tox = 1 - df["toxicity_score"] / 2.0        # 1..0
    uniq = df["uniqueness_score"]               # 0..1
    market = ((X-X.min())/(X.max()-X.min())) / 2.0           # 0..1
    comp = df["competition_score"] / 2.0        # 0..1 
    
    # интегральный показатель успеха
    base = (
        0.4 * eff +
        0.3 * tox +
        0.2 * uniq +
        0.1 * market
        # можно + 0.05 * (1 - comp), но конкуренция влияет на бизнес, а не на клинический успех
    )
    
    # ограничиваем
    base = base.clip(0, 1)

    # вычисляем проценты успеха фаз
    df["phase1"] = (50 + 30 * base).round(1)   # 50–80%
    df["phase2"] = (20 + 25 * base).round(1)   # 20–45%
    df["phase3"] = (30 + 35 * base).round(1)   # 30–65%

    print("Вероятности клинических фаз I–III подсчитаны")


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
    
    # записываем в df оценку целевого рынка 
    competition_scores()
    
    # записываем в df уровень конкуренции
    build_competition_level()
    
    # записываем в df стоимость разработки
    model_c, le_ind_c, le_target_c = cost_model()
    
    X_c = df[["indication","target","molecular_weight","logP","market_size_million","competition_level","efficiency_score","toxicity_score","traditional_time_years"]].copy()
    X_c["indication"] = le_ind_c.transform(X_t["indication"])
    X_c["target"]     = le_target_c.transform(X_t["target"])
    
    predict_c = model_c.predict(X_c)
    
    cost = predict_c.astype(float)
    
    # округляем до сотых, чтобы выглядело красиво
    df["dev_cost_million"] = cost.round(2)
    print("Стоимость разработки подсчитана")
    
    # записываем в df потанциальную прибыль как оценку и в млн и долю рынка для препарата
    potential_profit()
    
    # записываем в df процент прохождения клинических испытаний фазы 1-3
    clinical_trial_probabilities()
    
    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 4 блок функции
    
def select_top_candidates(k):
    """
    Функция находит топ капли
    Возвращает: DataFr из лучших капель
    """
    df_copy = df.copy()
    
    X = df_copy["market_size_million"]
    market = (X-X.min())/(X.max()-X.min()) / 2.0 
    
    # 1. Посчитать приоритет
    df_copy["priority_score"] = (
        0.5 * df["efficiency_score"]
        - 0.3 * df["toxicity_score"]
        + 0.2 * df["uniqueness_score"]
        + 0.2 * market 
        - 0.1 * df["competition_level"]
    )

    # 2. Фильтр по score = 2 (High)
    df_copy = df_copy[df_copy["score"] == 2]

    # 2A. Если нет ни одной подходящей молекулы → ошибка
    if len(df_copy) == 0:
        raise ValueError("Нет ни одной молекулы с высоким научным потенциалом (score = 2).")

    # 3. Сортировка
    df_copy = df_copy.sort_values("priority_score", ascending=False)

    # 3A. Если кандидатов меньше, чем k → выводим предупреждение
    if len(df_copy) < k:
        print(f"Найдено только {len(df_copy)} подходящих молекул из {k} запрошенных.")
        print("Возвращаем все доступные молекулы.")
        return(len(df_copy))
    
    print(f"Топ {k} молекул найдены")
    return df_copy.head(k)



def build_candidate_profile(top_df):
    """
    Функция принимает DataFrame и преобразует данные в python словарь
    Возвращает: список словарей для каждомй топ молекулы
    """
    profiles = []  # сюда сложим профили всех молекул
    for _, row in top_df.iterrows():
        profile = {
            "drug_id": row["drug_id"],
            "indication": row["indication"],
            "target": row["target"],
            "efficiency": row["efficiency_score"],
            "toxicity": row["toxicity_score"],
            "uniqueness": row["uniqueness_score"],
            "scientific_patent": row["score"],
            "dev_cost_million": row["dev_cost_million"],
            "competition_level": row["competition_level"],
            "profit_million": row["potential_profit_million"],
            "profit_million_score": row["profit_score"],
            "time_traditional": row["traditional_time_years"],
            "time_ai": row["ai_time_years"],
            "phase1_prob": row["phase1"],
            "phase2_prob": row["phase2"],
            "phase3_prob": row["phase3"],
        }
        profiles.append(profile)
        
    return profiles


def build_companies_profile():
    """
    Возвращает список словарей для каждой компании
    """
    df_comp = pd.read_csv("project/companies.csv")
    companies = []

    for _, row in df_comp.iterrows():
        companies.append({
            "company_id": row["company_id"],
            "company_name": row["company_name"],
            "type": row["type"],
            "focus_area": row["focus_area"],
            "portfolio": row["portfolio"],
            "strategy": row["strategy"]
        })

    return companies


def find_company_by_name(companies, company_name):
    """
    Функция ищет в списке словарь с нужной компанией
    Принимает: список словарей и имя компаннии
    Возвращает: словарь определенной компании
    """
    for company in companies:
        if company["company_name"]== company_name:
            return company
        
    raise ValueError(f"Компания с именем '{company_name}' не найдена.")


def save_dossiers(dossier_text, profile):
    """
    Функция сохраняет досье капли в файл
    Принимает: досье(text), профиль молекулы(словарь)
    """
    # создаём папку 
    os.makedirs("dossiers", exist_ok=True)
    # находим id молекулы
    id = profile["drug_id"]
    # указываем для файла путь и имя
    filename = f"dossiers/dossier_{id}.txt"
    #  открываем файл и записываем туда текст
    with open(filename, "w", encoding="utf-8") as f:
        f.write(dossier_text)
 

def json_move(profile):
    """
    Функция преобразует словарь в json
    Возвращает json строку
    """
    json_text = json.dumps(profile, ensure_ascii=False, indent=2)
    
    return json_text


def build_dossier_prompt(profile_json, company_json):
    """
    Функция создающая промт для LLM
    Принимает: json нужной капли и компании
    Возвращает: созданный промт
    """
    return f"""
    Ты — эксперт по фармразработкам и деловым коммуникациям.

    Тебе дан кандидат в лекарство и фармкомпания, которой мы хотим предложить лицензирование.

    [Данные о молекуле (JSON)]
    {profile_json}

    [Данные о компании (JSON)]
    {company_json}

    Твоя задача — написать ПОЛНОЕ «досье для лицензирования» на русском языке.
    Структура ответа:

    1. Краткое резюме (2–3 предложения)
    - Что это за молекула и для какого показания (indication).
    - Почему она может быть интересна именно этой компании.

    2. Научное обоснование
    - Механизм действия (target, механизм, если он понятен из данных).
    - Почему молекула перспективна с точки зрения эффективности (используй efficiency_score, phase1/2/3_prob).
    - Кратко прокомментируй токсичность (toxicity_score).

    3. Рыночный потенциал
    - Оценка размера рынка (market_size_million).
    - Уровень конкуренции (competition_level, competition_score, если есть).
    - Потенциальная прибыль (profit_million) и её интерпретация.

    4. Синергия с портфелем компании
    - Сошлись на текущий портфель компании (portfolio, focus_areas).
    - Объясни, как данный кандидат дополняет существующие продукты.
    - Приведи 1–2 формулировки вида:
        «Ваш текущий препарат X лечит симптомы, а наш кандидат нацелен на причину заболевания».

    5. Операционные аспекты и сроки
    - Сравни традиционный срок разработки (time_traditional) и AI-ускоренный (time_ai).
    - Скажи, насколько быстрее можно выйти на рынок с использованием ИИ-подходов.

    6. Риски и неопределённости
    - Отметь основные научные и регуляторные риски.
    - Укажи, какие вопросы потребуют проверки в дальнейших исследованиях.

    Тон и акценты:
    - Если company["type"] = "biotech" или компания маленькая/инновационная — делай акцент на науке, новизне механизма и научной значимости.
    - Если company["type"] = "big_pharma" — делай акцент на масштабируемости, прибыли, сроках вывода и конкурентных преимуществах.
    - Используй умеренно формальный, профессиональный стиль.
    - Не повторяй дословно JSON, а используй его как факты для связного текста.

    Ответ верни только в виде оформленного текста досье, без JSON, без списков профиля и компании.
    """


def LLM(promt):
    """
    Функция работающая с LLM и сохраняет досье по каплям
    Принимает: промт
    Возвращает: ответ модели в виде текста
    Важно: добавте свой ключ от Groq иначе LLM не заработает
    """
    
    # открывает .env и превращает ключ в переменную среды
    load_dotenv("project/.env") 
    # присваеваем переменную среды (key_DeepSeek)
    Key = os.getenv("GROQ_API_KEY")
    
    client = OpenAI(
    api_key= Key,
    base_url="https://api.groq.com/openai/v1"
    )

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": promt}]
    )

    # получаем текст ответа
    text = resp.choices[0].message.content
    return text


def move_3(k):
    # находим табл топ кандидатов select_top_candidates(k) k=5
    top_df = select_top_candidates(k)
    
    # получ список со словорями для топ молекул
    profiles = build_candidate_profile(top_df)
    
    # получ список со словарями для компаний
    companies = build_companies_profile()
    
    # находим информацию о нуж компании и превращаем в json
    print("Введите название компании:")
    user_com = input()
    company = find_company_by_name(companies, user_com)
    company = json_move(company)
    
    # проходим по всем перспективным каплям
    for i in profiles:
        # получаем json одной капли
        capl = json_move(i)     
        
        # получаем промт
        promt = build_dossier_prompt(capl, company)
        # запускаем модель
        text = LLM(promt)
        
        save_dossiers(text, i)
        
    
    
def main():
    # работа 2 блока
    move_data()
    # работа 3 блока
    move_data_2()
    
    k=5
    move_3(k)

    

if __name__ == "__main__":
    main()