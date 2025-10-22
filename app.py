import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

def create_validation_charts(df_clean, y, y_pred):
    """Создание графиков валидации с использованием Altair"""
    
    # Данные для графиков
    plot_data = pd.DataFrame({
        'actual': np.exp(y),
        'predicted': np.exp(y_pred),
        'residuals': np.exp(y) - np.exp(y_pred),
        'temperature': df_clean['T'],
        'grain_size': df_clean['G']
    })
    
    # График 1: Предсказанные vs Фактические значения
    chart1 = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X('actual:Q', title='Фактический диаметр (мкм²)'),
        y=alt.Y('predicted:Q', title='Предсказанный диаметр (мкм²)'),
        color='temperature:Q',
        tooltip=['actual', 'predicted', 'temperature', 'grain_size']
    ).properties(
        width=400,
        height=300,
        title='Предсказанные vs Фактические значения'
    )
    
    line = alt.Chart(pd.DataFrame({
        'actual': [plot_data['actual'].min(), plot_data['actual'].max()],
        'predicted': [plot_data['actual'].min(), plot_data['actual'].max()]
    })).mark_line(color='red', strokeDash=[5,5]).encode(
        x='actual:Q',
        y='predicted:Q'
    )
    
    chart1 = chart1 + line
    
    # График 2: Остатки
    chart2 = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X('predicted:Q', title='Предсказанный диаметр (мкм²)'),
        y=alt.Y('residuals:Q', title='Остатки'),
        color='temperature:Q',
        tooltip=['predicted', 'residuals', 'temperature']
    ).properties(
        width=400,
        height=300,
        title='Остатки модели'
    )
    
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q')
    chart2 = chart2 + zero_line
    
    # График 3: Распределение ошибок
    chart3 = alt.Chart(plot_data).mark_bar().encode(
        x=alt.X('residuals:Q', bin=alt.Bin(maxbins=15), title='Ошибка предсказания'),
        y=alt.Y('count()', title='Частота')
    ).properties(
        width=400,
        height=300,
        title='Распределение ошибок'
    )
    
    return chart1, chart2, chart3

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
                    
                    # Показываем данные
                    with st.expander("Просмотр данных"):
                        st.dataframe(df)
                    
                    # Выбор данных для исключения
                    st.subheader("2. Выбор данных для исключения")
                    st.write("Исключите выбросы для улучшения модели:")
                    
                    excluded_indices = []
                    
                    for idx, row in df.iterrows():
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                        with col1:
                            st.write(f"Измерение {idx+1}")
                        with col2:
                            st.write(f"G={row['G']}")
                        with col3:
                            st.write(f"T={row['T']}°C")
                        with col4:
                            st.write(f"t={row['t']}ч")
                        with col5:
                            if st.checkbox("Исключить", key=f"exclude_{idx}"):
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
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("R²", f"{model.r2:.4f}")
                            with col2:
                                st.metric("RMSE", f"{model.rmse:.4f}")
                            with col3:
                                st.metric("MAE", f"{model.mae:.4f}")
                            with col4:
                                st.metric("Количество точек", f"{len(df_clean)}")
                            
                            # Графики валидации
                            st.subheader("4. Валидация модели")
                            chart1, chart2, chart3 = create_validation_charts(df_clean, y, y_pred)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.altair_chart(chart1, use_container_width=True)
                                st.altair_chart(chart3, use_container_width=True)
                            with col2:
                                st.altair_chart(chart2, use_container_width=True)
                                
                                # График ошибок по температурам
                                error_by_temp = df_clean.groupby('T').apply(
                                    lambda x: (x['d'] - x['d_pred']).mean()
                                ).reset_index()
                                error_by_temp.columns = ['T', 'mean_error']
                                
                                chart4 = alt.Chart(error_by_temp).mark_bar().encode(
                                    x=alt.X('T:Q', title='Температура (°C)'),
                                    y=alt.Y('mean_error:Q', title='Средняя ошибка'),
                                    tooltip=['T', 'mean_error']
                                ).properties(
                                    height=300,
                                    title='Средняя ошибка по температурам'
                                )
                                st.altair_chart(chart4, use_container_width=True)
                            
                            # Таблица с сравнением
                            st.subheader("Сравнение экспериментальных и расчетных значений")
                            comparison_df = df_clean[['G', 'T', 't', 'd', 'd_pred']].copy()
                            comparison_df['Ошибка, %'] = 100 * (comparison_df['d_pred'] - comparison_df['d']) / comparison_df['d']
                            comparison_df['d'] = comparison_df['d'].round(4)
                            comparison_df['d_pred'] = comparison_df['d_pred'].round(4)
                            comparison_df['Ошибка, %'] = comparison_df['Ошибка, %'].round(2)
                            
                            st.dataframe(comparison_df)
                            
                            # Сохранение модели в сессии
                            st.session_state['trained_model'] = model
                            st.session_state['model_coef'] = model.coef_
                            st.session_state['model_intercept'] = model.intercept_
                            
                            # Кнопка для экспорта модели
                            st.subheader("5. Экспорт модели")
                            model_params = {
                                'intercept': model.intercept_,
                                'coef_ln_t': model.coef_[0],
                                'coef_inv_T': model.coef_[1],
                                'coef_ln_inv_sqrt_a_v': model.coef_[2]
                            }
                            
                            st.write("Параметры модели для использования в других программах:")
                            st.code(f"""
                            Модель: ln(d) = β₀ + β₁·ln(t) + β₂·(1/T) + β₃·ln(1/√a_v)
                            
                            β₀ = {model.intercept_:.8f}
                            β₁ = {model.coef_[0]:.8f}
                            β₂ = {model.coef_[1]:.8f}
                            β₃ = {model.coef_[2]:.8f}
                            
                            Формула для расчета температуры:
                            T = β₂ / [ln(d) - β₀ - β₁·ln(t) - β₃·ln(1/√a_v)] - 273.15
                            """)
                            
                        except Exception as e:
                            st.error(f"Ошибка при обучении модели: {str(e)}")
                    else:
                        st.warning("Недостаточно данных для обучения модели. Нужно минимум 4 измерения.")
                        
                else:
                    st.error(f"В файле должны быть столбцы: {required_columns}")
                    
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {str(e)}")
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
                        st.error(f"⚠️ Рассчитанная температура: {temperature:.1f} °C\n\n"
                                "**Внимание:** Температура ниже 550°C - сигма-фаза не выделяется")
                    elif temperature > 900:
                        st.error(f"⚠️ Рассчитанная температура: {temperature:.1f} °C\n\n"
                                "**Внимание:** Температура выше 900°C - сигма-фаза не выделяется")
                    elif 590 <= temperature <= 630:
                        st.success(f"✅ Оптимальный диапазон: {temperature:.1f} °C\n\n"
                                 "**Модель работает с максимальной точностью**")
                    else:
                        st.warning(f"📊 Рассчитанная температура: {temperature:.1f} °C\n\n"
                                 "**Внимание:** Температура вне оптимального диапазона 590-630°C")
                    
                    # Дополнительная информация
                    with st.expander("Детали расчета"):
                        grain_info = grain_df[grain_df['G'] == grain_number].iloc[0]
                        st.write(f"**Параметры зерна №{grain_number}:**")
                        st.write(f"- Средняя площадь сечения: {grain_info['a_v']} мм²")
                        st.write(f"- Средний диаметр: {grain_info['d_av']} мм")
                        st.write(f"- ln(1/√a_v) = {grain_info['ln_inv_sqrt_a_v']:.4f}")
                        
                        st.write("**Использованные параметры модели:**")
                        st.write(f"- β₀ = {model.intercept_:.6f}")
                        st.write(f"- β₁ = {model.coef_[0]:.6f}")
                        st.write(f"- β₂ = {model.coef_[1]:.6f}")
                        st.write(f"- β₃ = {model.coef_[2]:.6f}")
                        
                except Exception as e:
                    st.error(f"Ошибка при расчете: {str(e)}")
        else:
            st.warning("Сначала обучите модель во вкладке 'Анализ данных'")

if __name__ == "__main__":
    main()
