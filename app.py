import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import io
import warnings
warnings.filterwarnings('ignore')

# Данные по размерам зерен из ГОСТ
GRAIN_DATA = {
    'G': [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'a_v': [1.000, 0.500, 0.250, 0.125, 0.0625, 0.0312, 0.0156, 0.00781, 0.00390, 
            0.00195, 0.00098, 0.00049, 0.000244, 0.000122, 0.000061, 0.000030, 0.000015, 0.000008],
    'd_av': [1.000, 0.707, 0.500, 0.353, 0.250, 0.177, 0.125, 0.088, 0.062, 
             0.044, 0.031, 0.022, 0.015, 0.011, 0.0079, 0.0056, 0.0039, 0.0027]
}

grain_df = pd.DataFrame(GRAIN_DATA)
grain_df['inv_sqrt_a_v'] = 1 / np.sqrt(grain_df['a_v'])
grain_df['ln_inv_sqrt_a_v'] = np.log(grain_df['inv_sqrt_a_v'])

class SigmaPhaseModel:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.r2 = None
        self.rmse = None
        self.mae = None
        
    def fit(self, X, y):
        model = LinearRegression()
        model.fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        
        # Расчет метрик
        y_pred = model.predict(X)
        self.r2 = r2_score(y, y_pred)
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))
        self.mae = mean_absolute_error(y, y_pred)
        
        return self
    
    def predict_temperature(self, d_sigma, time_hours, grain_size):
        """Предсказание температуры по модели"""
        if self.coef_ is None:
            raise ValueError("Модель не обучена!")
            
        # Получаем данные по зерну
        grain_info = grain_df[grain_df['G'] == grain_size]
        if len(grain_info) == 0:
            raise ValueError(f"Номер зерна {grain_size} не найден в базе данных")
            
        ln_inv_sqrt_a_v = grain_info['ln_inv_sqrt_a_v'].iloc[0]
        
        # Расчет по модели: ln(d_σ) = β₀ + β₁×ln(t) + β₂×(1/T) + β₃×ln(1/√a_v)
        # Преобразуем для получения температуры: 1/T = [ln(d_σ) - β₀ - β₁×ln(t) - β₃×ln(1/√a_v)] / β₂
        ln_d_sigma = np.log(d_sigma)
        ln_time = np.log(time_hours)
        
        numerator = ln_d_sigma - self.intercept_ - self.coef_[0] * ln_time - self.coef_[2] * ln_inv_sqrt_a_v
        inv_T = numerator / self.coef_[1]
        
        T_kelvin = 1 / inv_T
        T_celsius = T_kelvin - 273.15
        
        return T_celsius

def prepare_data(df, excluded_indices=[]):
    """Подготовка данных для регрессии"""
    df_clean = df.drop(excluded_indices).copy()
    
    # Добавляем данные по зернам
    df_clean = df_clean.merge(grain_df[['G', 'ln_inv_sqrt_a_v']], on='G', how='left')
    
    # Преобразуем переменные
    df_clean['ln_d'] = np.log(df_clean['d'])
    df_clean['ln_t'] = np.log(df_clean['t'])
    df_clean['inv_T'] = 1 / (df_clean['T'] + 273.15)  # T в Кельвинах
    
    # Создаем матрицу признаков
    X = df_clean[['ln_t', 'inv_T', 'ln_inv_sqrt_a_v']].values
    y = df_clean['ln_d'].values
    
    return X, y, df_clean

def main():
    st.set_page_config(page_title="Sigma Phase Analyzer", layout="wide")
    st.title("🔬 Анализатор сигма-фазы в стали 12Х18Н12Т")
    
    # Создаем вкладки
    tab1, tab2 = st.tabs(["📊 Анализ данных и калибровка модели", "🧮 Калькулятор температуры"])
    
    with tab1:
        st.header("Калибровка физической модели")
        
        # Загрузка данных
        st.subheader("1. Загрузка данных")
        uploaded_file = st.file_uploader("Загрузите Excel файл с данными", type=['xlsx'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                required_columns = ['G', 'T', 't', 'd']
                
                if all(col in df.columns for col in required_columns):
                    st.success("Данные успешно загружены!")
                    st.write("Предпросмотр данных:")
                    st.dataframe(df.head())
                    
                    # Показываем статистику
                    st.subheader("Статистика данных")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Количество измерений", len(df))
                    with col2:
                        st.metric("Диапазон температур", f"{df['T'].min()} - {df['T'].max()} °C")
                    with col3:
                        st.metric("Диапазон времени", f"{df['t'].min()} - {df['t'].max()} ч")
                    with col4:
                        st.metric("Номера зерен", ", ".join(map(str, sorted(df['G'].unique()))))
                    
                    # Выбор данных для исключения
                    st.subheader("2. Выбор данных для исключения")
                    st.write("Исключите выбросы для улучшения модели:")
                    
                    excluded_indices = []
                    cols = st.columns(3)
                    
                    for idx, row in df.iterrows():
                        col_idx = idx % 3
                        with cols[col_idx]:
                            if st.checkbox(f"Исключить: G={row['G']}, T={row['T']}°C, t={row['t']}ч", 
                                         key=f"exclude_{idx}"):
                                excluded_indices.append(idx)
                    
                    # Обучение модели
                    st.subheader("3. Обучение модели")
                    if len(df) - len(excluded_indices) >= 4:  # Минимум 4 точки для регрессии
                        try:
                            X, y, df_clean = prepare_data(df, excluded_indices)
                            model = SigmaPhaseModel()
                            model.fit(X, y)
                            
                            # Предсказания
                            y_pred = model.intercept_ + X @ model.coef_
                            df_clean['d_pred'] = np.exp(y_pred)
                            df_clean['T_pred'] = np.exp(model.intercept_ + 
                                                       model.coef_[0] * df_clean['ln_t'] + 
                                                       model.coef_[2] * df_clean['ln_inv_sqrt_a_v']) / \
                                               (model.coef_[1] * df_clean['inv_T'])
                            
                            # Показываем коэффициенты модели
                            st.subheader("Коэффициенты модели")
                            st.latex(r"ln(d) = \beta_0 + \beta_1 \cdot ln(t) + \beta_2 \cdot \frac{1}{T} + \beta_3 \cdot ln\left(\frac{1}{\sqrt{a_v}}\right)")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"β₀ (intercept) = {model.intercept_:.6f}")
                                st.write(f"β₁ (ln(t)) = {model.coef_[0]:.6f}")
                            with col2:
                                st.write(f"β₂ (1/T) = {model.coef_[1]:.6f}")
                                st.write(f"β₃ (ln(1/√a_v)) = {model.coef_[2]:.6f}")
                            
                            # Метрики качества
                            st.subheader("Метрики качества модели")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R²", f"{model.r2:.4f}")
                            with col2:
                                st.metric("RMSE", f"{model.rmse:.4f}")
                            with col3:
                                st.metric("MAE", f"{model.mae:.4f}")
                            
                            # Графики валидации
                            st.subheader("4. Валидация модели")
                            
                            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                            
                            # График 1: Предсказанные vs Фактические значения
                            ax1.scatter(np.exp(y), np.exp(y_pred), alpha=0.7)
                            ax1.plot([np.exp(y).min(), np.exp(y).max()], 
                                    [np.exp(y).min(), np.exp(y).max()], 'r--', alpha=0.8)
                            ax1.set_xlabel('Фактический диаметр (мкм²)')
                            ax1.set_ylabel('Предсказанный диаметр (мкм²)')
                            ax1.set_title('Предсказанные vs Фактические значения')
                            ax1.grid(True, alpha=0.3)
                            
                            # График 2: Остатки
                            residuals = np.exp(y) - np.exp(y_pred)
                            ax2.scatter(np.exp(y_pred), residuals, alpha=0.7)
                            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
                            ax2.set_xlabel('Предсказанный диаметр (мкм²)')
                            ax2.set_ylabel('Остатки')
                            ax2.set_title('Остатки модели')
                            ax2.grid(True, alpha=0.3)
                            
                            # График 3: Распределение ошибок
                            ax3.hist(residuals, bins=15, alpha=0.7, edgecolor='black')
                            ax3.set_xlabel('Ошибка предсказания')
                            ax3.set_ylabel('Частота')
                            ax3.set_title('Распределение ошибок')
                            ax3.grid(True, alpha=0.3)
                            
                            # График 4: Зависимость по температурам
                            temperatures = df_clean['T'].unique()
                            mean_errors = []
                            for temp in temperatures:
                                temp_mask = df_clean['T'] == temp
                                mean_errors.append(residuals[temp_mask].mean())
                            
                            ax4.bar(temperatures, mean_errors, alpha=0.7)
                            ax4.set_xlabel('Температура (°C)')
                            ax4.set_ylabel('Средняя ошибка')
                            ax4.set_title('Ошибка предсказания по температурам')
                            ax4.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Таблица с сравнением
                            st.subheader("Сравнение экспериментальных и расчетных значений")
                            comparison_df = df_clean[['G', 'T', 't', 'd', 'd_pred']].copy()
                            comparison_df['Ошибка, %'] = 100 * (comparison_df['d_pred'] - comparison_df['d']) / comparison_df['d']
                            st.dataframe(comparison_df.round(4))
                            
                            # Сохранение модели в сессии
                            st.session_state['trained_model'] = model
                            st.session_state['model_coef'] = model.coef_
                            st.session_state['model_intercept'] = model.intercept_
                            
                        except Exception as e:
                            st.error(f"Ошибка при обучении модели: {e}")
                    else:
                        st.warning("Недостаточно данных для обучения модели. Нужно минимум 4 измерения.")
                        
                else:
                    st.error(f"В файле должны быть столбцы: {required_columns}")
                    
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")
        else:
            st.info("Загрузите Excel файл с колонками: G, T, t, d")
    
    with tab2:
        st.header("Калькулятор температуры эксплуатации")
        
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            
            st.success("Модель готова к использованию!")
            st.write("Введите параметры для расчета температуры:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                grain_number = st.selectbox("Номер зерна (G)", options=grain_df['G'].tolist())
            with col2:
                time_hours = st.number_input("Время эксплуатации (ч)", min_value=1, value=5000)
            with col3:
                d_sigma = st.number_input("Эквивалентный диаметр сигма-фазы (мкм²)", 
                                        min_value=0.1, value=10.0, step=0.1)
            
            if st.button("Рассчитать температуру"):
                try:
                    temperature = model.predict_temperature(d_sigma, time_hours, grain_number)
                    
                    # Проверка диапазона работоспособности
                    if temperature < 550:
                        st.error(f"⚠️ Рассчитанная температура: {temperature:.1f} °C\n"
                                "Температура ниже 550°C - сигма-фаза не выделяется")
                    elif temperature > 900:
                        st.error(f"⚠️ Рассчитанная температура: {temperature:.1f} °C\n"
                                "Температура выше 900°C - сигма-фаза не выделяется")
                    elif 590 <= temperature <= 630:
                        st.success(f"✅ Оптимальный диапазон: {temperature:.1f} °C\n"
                                 "Модель работает с максимальной точностью")
                    else:
                        st.warning(f"📊 Рассчитанная температура: {temperature:.1f} °C\n"
                                 "Внимание: температура вне оптимального диапазона 590-630°C")
                    
                    # Дополнительная информация
                    with st.expander("Детали расчета"):
                        grain_info = grain_df[grain_df['G'] == grain_number].iloc[0]
                        st.write(f"Параметры зерна №{grain_number}:")
                        st.write(f"- Средняя площадь сечения: {grain_info['a_v']} мм²")
                        st.write(f"- Средний диаметр: {grain_info['d_av']} мм")
                        st.write(f"- ln(1/√a_v) = {grain_info['ln_inv_sqrt_a_v']:.4f}")
                        
                except Exception as e:
                    st.error(f"Ошибка при расчете: {e}")
        else:
            st.warning("Сначала обучите модель во вкладке 'Анализ данных'")

if __name__ == "__main__":
    main()
