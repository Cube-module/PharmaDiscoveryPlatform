import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder # для строковых значений из csv (это типо hot_encoder но для дерева лучше)
import numpy as np


# счит базу данных
df = pd.read_csv("drug_candidates_fourth.csv")

def razbit(name):
    """
    Разбивает данные на test, val, train для разных целей (potential_label, toxicity_label)
    Принимает: name значение int 0/1 нужно для выбора разбиения под цель
    Возвращает: в зависимости от name (potential_label / toxicity_label) и X (dataframe)
    potential_label(0) -  X_train_1, X_test_1, X_val_1, Y_1_train, Y_1_test, Y_1_val
    toxicity_label(1) -  X_train_2, X_test_2, X_val_2, Y_2_train, Y_2_test, Y_2_val
    """
    
    # разбиваем наши цели предсказывания
    Y_1 = df["potential_label"]
    Y_2 = df["toxicity_label"]

    # удаляем из базы цели предсказ и лишние столбцы не нужные для предсказ
    X = df.drop(columns=["potential_label", "toxicity_label", "drug_id", 
                         "market_size_million", "competition_level",
                         "expected_profit_score", "success_probability",
                         "traditional_time_years", "ai_time_years",
                         "has_positive_efficacy_phrase", "has_severe_toxicity_phrase"])
    
    # преобразуем строковые значения из csv в индексный вид с помощью LabelEncoder
    le_ind = LabelEncoder()
    le_target = LabelEncoder()
    
    X["indication"] = le_ind.fit_transform(X["indication"])
    X["target"] = le_target.fit_transform(X["target"])
    
    # проверка на имя вызова (potential_label)
    if name == 0:
        # разбиение для Y_1
        X_train, X_test_1, Y_1_train, Y_1_test = train_test_split(X, Y_1, test_size=0.15, random_state=42)
        X_train_1, X_val_1, Y_1_train, Y_1_val = train_test_split(X_train, Y_1_train, test_size=0.15, random_state=42)
        
        return X_train_1, X_test_1, X_val_1, Y_1_train, Y_1_test, Y_1_val, X

    else:
        # разбиение для Y_2
        X_train, X_test_2, Y_2_train, Y_2_test = train_test_split(X, Y_2, test_size=0.15, random_state=42)
        X_train_2, X_val_2, Y_2_train, Y_2_val = train_test_split(X_train, Y_2_train, test_size=0.15, random_state=42)
        
        return X_train_2, X_test_2, X_val_2, Y_2_train, Y_2_test, Y_2_val
    

def accuracy():
    """
    Принимает: -
    Возвращает: предсказания моделей potential_label и toxicity_label предсказания по всему неразбитаму датасету
    Функция вычесляющая и печатающая точность предсказания
    """
    # подготавливаем и получаем данные
    X_train_1, X_test_1, X_val_1, Y_1_train, Y_1_test, Y_1_val, X = razbit(0)
    X_train_2, X_test_2, X_val_2, Y_2_train, Y_2_test, Y_2_val = razbit(1)
    
    # создаем модель для potential_label обучаем 
    model_1 = RandomForestClassifier(n_estimators=200, max_depth=30, criterion='entropy', random_state=42)
    model_1.fit(X_train_1, Y_1_train)
   
    # создаем модель для toxicity_label обучаем 
    model_2 = RandomForestClassifier(n_estimators=200, max_depth=30, criterion= 'entropy', random_state=42)
    model_2.fit(X_train_2, Y_2_train)
    
    # смотрим точность предсказания potential_label
    predict_1 = model_1.predict(X_val_1)
    val_accur_1 = accuracy_score(Y_1_val, predict_1)
    print("точность potential_label: ", val_accur_1)
    predict_1_all = model_1.predict(X)
    
    # смотрим точность предсказания toxicity_label
    predict_2 = model_2.predict(X_val_2)
    val_accur_2 = accuracy_score(Y_2_val, predict_2)
    print("точность toxicity_label: ", val_accur_2)
    predict_2_all = model_2.predict(X)

    return predict_1_all, predict_2_all

def main():

    # получаем предсказания моделей (numpy масив) и смотрим точность предсказаний
    predict_1, predict_2 = accuracy() 
    
    # собираем DataFrame для анализа привлекательности блок 3
    df["pred_potential"] = predict_1
    df["pred_toxicity"] = predict_2
    
    cols_block3 = [
    "drug_id",
    "indication",
    "pred_potential",      # предсказанный потенциал
    "pred_toxicity",       # предсказанная токсичность
    "market_size_million",
    "competition_level",
    "expected_profit_score",
    "success_probability",
    "traditional_time_years",
    "ai_time_years",
    ]
    
    df_b3 = df[cols_block3].copy() # собрали
    
    # считаем общий результат для кандидата
    df_b3["overall_priority_score"] = (
    (df_b3["pred_potential"] / 2) * 0.4 +          # чем выше потенциал, тем лучше
    (1 - df_b3["pred_toxicity"]) * 0.2 +          # нет токсичности → плюс
    df_b3["expected_profit_score"] * 0.4          # прибыльность с большим весом
    )
    
    # берем топ 5 кандидатов
    top5 = df_b3.sort_values("overall_priority_score", ascending=False).head(5)
    
    print(top5)
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()