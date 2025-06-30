import numpy as np
import matplotlib
matplotlib.use('Agg')  # Для веб-приложения
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import io
import base64
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

class ChemicalReactorModel:
    """Класс для моделирования химических реакций в реакторе"""
    
    def __init__(self):
        self.R_const = 8.314  # Газовая постоянная
        self.T_base = 573.15  # Базовая температура (300°C)
        self.tau_end = 3  # Время реакции, с
        
        # Целевые концентрации продуктов
        self.target = np.array([
            0.0016, 0.0968, 0.0764, 0.0008, 
            0.0024, 0.0033, 0.0008, 0.1309
        ])  # [C3H6, O2, AR, AAc, CO, CO2, Ac, H2O]
        
        # Начальные условия
        self.conc0_base = np.array([
            0.0814, 0.1828, 0, 0, 0, 0, 0, 0.0480
        ])
        
        # Метки компонентов
        self.labels = [
            'C₃H₆ (Пропилен)', 'O₂ (Кислород)', 'AR (Акролеин)', 
            'AAc (Акриловая кислота)', 'CO', 'CO₂', 
            'Ac (Уксусная кислота)', 'H₂O'
        ]
        
        # Оптимальные параметры (будут определены при оптимизации)
        self.optimal_params = None
        
    def rates(self, conc: np.ndarray, k: np.ndarray) -> Tuple[float, ...]:
        """Расчет скоростей реакций"""
        A, B = conc[0], conc[1]
        if A <= 0 or B <= 0:
            return (0, 0, 0, 0, 0)
        
        r1 = k[0] * A * B
        r2 = k[1] * A * B**1.5
        r3 = k[2] * A * B**3
        r4 = k[3] * A * B**4.5
        r5 = k[4] * A * B**2.5
        
        return (r1, r2, r3, r4, r5)
    
    def deriv(self, t: float, conc: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Система дифференциальных уравнений"""
        A, B, AR, AAc, CO, CO2, Ac, H2O = conc
        
        if A <= 0 or B <= 0:
            return np.zeros(8)
        
        r1, r2, r3, r4, r5 = self.rates(conc, k)
        
        dA = -(r1 + r2 + r3 + r4 + r5)
        dB = -(1*r1 + 1.5*r2 + 3*r3 + 4.5*r4 + 2.5*r5)
        dAR = r1
        dAAc = r2
        dCO = 3*r3
        dCO2 = 3*r4 + 1*r5
        dAc = r5
        dH2O = r1 + r2 + 3*r3 + 3*r4 + r5
        
        return np.array([dA, dB, dAR, dAAc, dCO, dCO2, dAc, dH2O])
    
    def integrate_system(self, conc0: np.ndarray, k: np.ndarray, 
                        t_end: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Интегрирование системы ДУ"""
        if t_end is None:
            t_end = self.tau_end
            
        sol = solve_ivp(
            self.deriv, [0, t_end], conc0, args=(k,), 
            method='RK45', rtol=1e-6, atol=1e-8, dense_output=True
        )
        
        return sol.t, sol.y.T
    
    def calculate_k_values(self, A_vals: np.ndarray, E_vals: np.ndarray, 
                          T: float) -> np.ndarray:
        """Расчет констант скорости по уравнению Аррениуса"""
        return A_vals * np.exp(-E_vals / (self.R_const * T))
    
    def residuals_with_tikhonov(self, p: np.ndarray, lambda_reg: float = 0,
                               T: float = None, conc0: np.ndarray = None) -> np.ndarray:
        """Функция невязки с регуляризацией Тихонова"""
        if T is None:
            T = self.T_base
        if conc0 is None:
            conc0 = self.conc0_base
            
        A_vals = p[:5]
        E_vals = p[5:]
        k_vals = self.calculate_k_values(A_vals, E_vals, T)
        
        _, conc_history = self.integrate_system(conc0, k_vals, t_end=self.tau_end)
        final_conc = conc_history[-1]
        
        model = final_conc
        residual = model - self.target
        reg = lambda_reg * np.hstack([A_vals, E_vals])
        
        return np.hstack([residual, reg])
    
    def optimize_parameters(self, lambda_reg: float = 0, T: float = None,
                           conc0: np.ndarray = None) -> Dict[str, Any]:
        """Оптимизация параметров модели"""
        if T is None:
            T = self.T_base
        if conc0 is None:
            conc0 = self.conc0_base
            
        # Начальное приближение
        p0 = np.array([0.001]*5 + [29000, 50000, 50000, 25000, 25000])
        
        # Оптимизация
        res = least_squares(
            self.residuals_with_tikhonov, p0, 
            bounds=(0, np.inf), max_nfev=10000,
            args=(lambda_reg, T, conc0)
        )
        
        optimal_p = res.x
        A_vals_opt = optimal_p[:5]
        E_vals_opt = optimal_p[5:]
        k_vals_opt = self.calculate_k_values(A_vals_opt, E_vals_opt, T)
        
        return {
            'success': res.success,
            'A_vals': A_vals_opt,
            'E_vals': E_vals_opt,
            'k_vals': k_vals_opt,
            'residual': res.fun,
            'cost': res.cost
        }
    
    def pfr_model(self, T: float, o2_propene_ratio: float, 
                  residence_time: float = None) -> Dict[str, float]:
        """PFR модель реактора идеального вытеснения"""
        if residence_time is None:
            residence_time = self.tau_end
            
        # Пересчет начальных концентраций с учетом соотношения O2:пропилен
        total_propene = 0.0814
        total_flow = total_propene * (1 + o2_propene_ratio)
        
        conc0_pfr = self.conc0_base.copy()
        conc0_pfr[0] = total_propene  # пропилен
        conc0_pfr[1] = total_propene * o2_propene_ratio  # кислород
        
        # Используем оптимальные параметры или стандартные
        if self.optimal_params is None:
            self.optimal_params = self.optimize_parameters()
        
        k_vals = self.calculate_k_values(
            self.optimal_params['A_vals'], 
            self.optimal_params['E_vals'], 
            T
        )
        
        # Интегрирование
        times, conc_history = self.integrate_system(conc0_pfr, k_vals, residence_time)
        final_conc = conc_history[-1]
        
        # Расчет выходов
        aac_yield = (final_conc[3] / conc0_pfr[0]) * 100  # Выход акриловой кислоты
        ar_yield = (final_conc[2]/conc0_pfr[0])*100
        selectivity = final_conc[2] / (conc0_pfr[0] - final_conc[0]) * 100 if (conc0_pfr[0] - final_conc[0]) > 0 else 0
        conversion = (conc0_pfr[0] - final_conc[0]) / conc0_pfr[0] * 100
        
        return {
            'ar_yield': ar_yield,
            'aac_yield': aac_yield,
            'selectivity': selectivity,
            'conversion': conversion,
            'final_concentrations': final_conc,
            'concentration_history': conc_history,
            'times': times
        }
    
    def create_3d_surface(self, T_range: Tuple[float, float] = (500, 650),
                         ratio_range: Tuple[float, float] = (1.0, 4.0),
                         n_points: int = 20) -> go.Figure:
        """Создание 3D поверхности отклика"""
        T_values = np.linspace(T_range[0], T_range[1], n_points)
        ratio_values = np.linspace(ratio_range[0], ratio_range[1], n_points)
        
        T_grid, ratio_grid = np.meshgrid(T_values, ratio_values)
        yield_grid = np.zeros_like(T_grid)
        
        # Прогресс-бар для Streamlit
        if 'streamlit' in globals():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        total_points = n_points * n_points
        
        for i, T in enumerate(T_values):
            for j, ratio in enumerate(ratio_values):
                try:
                    result = self.pfr_model(T, ratio)
                    yield_grid[j, i] = result['ar_yield']
                except:
                    yield_grid[j, i] = 0
                
                # Обновление прогресса
                if 'streamlit' in globals():
                    progress = (i * n_points + j + 1) / total_points
                    progress_bar.progress(progress)
                    status_text.text(f'Расчет: T={T:.1f}K, O₂:C₃H₆={ratio:.2f}')
        
        if 'streamlit' in globals():
            progress_bar.empty()
            status_text.empty()
        
        # Создание 3D графика
        fig = go.Figure(data=[go.Surface(
            z=yield_grid,
            x=T_grid,
            y=ratio_grid,
            colorscale='Viridis',
            colorbar=dict(title="Выход Ar, %")
        )])
        
        fig.update_layout(
            title='Поверхность отклика: Выход акриловой кислоты',
            scene=dict(
                xaxis_title='Температура, K',
                yaxis_title='Соотношение O₂:C₃H₆',
                zaxis_title='Выход Ar, %',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_optimization_plots(self) -> List[go.Figure]:
        """Создание графиков оптимизации"""
        lambda_values = np.logspace(-10, -2, 10)
        A_results = []
        E_results = []
        residuals_norm = []
        solution_norm = []
        
        p_current = np.array([0.001]*5 + [29000, 50000, 50000, 25000, 25000])
        
        for lambda_reg in lambda_values:
            result = self.optimize_parameters(lambda_reg)
            A_results.append(result['A_vals'])
            E_results.append(result['E_vals'])
            residuals_norm.append(np.linalg.norm(result['residual'][:8]))
            solution_norm.append(np.linalg.norm(np.hstack([result['A_vals'], result['E_vals']])))
        
        A_results = np.array(A_results)
        E_results = np.array(E_results)
        
        figures = []
        
        # График зависимости параметров A от lambda
        fig1 = go.Figure()
        for i in range(5):
            fig1.add_trace(go.Scatter(
                x=lambda_values, y=A_results[:, i],
                mode='lines+markers', name=f'A_{i+1}'
            ))
        fig1.update_layout(
            title='Зависимость предэкспоненциальных факторов от λ',
            xaxis_title='λ (регуляризация)',
            yaxis_title='A',
            xaxis_type='log'
        )
        figures.append(fig1)
        
        # График зависимости параметров E от lambda
        fig2 = go.Figure()
        for i in range(5):
            fig2.add_trace(go.Scatter(
                x=lambda_values, y=E_results[:, i],
                mode='lines+markers', name=f'E_{i+1}'
            ))
        fig2.update_layout(
            title='Зависимость энергий активации от λ',
            xaxis_title='λ (регуляризация)',
            yaxis_title='E, Дж/моль',
            xaxis_type='log'
        )
        figures.append(fig2)
        
        # L-кривая
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=residuals_norm, y=solution_norm,
            mode='lines+markers', name='L-кривая'
        ))
        fig3.update_layout(
            title='L-кривая',
            xaxis_title='Норма невязки',
            yaxis_title='Норма решения'
        )
        figures.append(fig3)
        
        return figures

    def selectivity_vs_conversion(self, T: float, o2_ratio: float, max_time: float = 5.0, n_points: int = 50):
        times = np.linspace(0.1, max_time, n_points)
        conversions, sel_acrolein, sel_acid = [], [], []

        for t in times:
            res = self.pfr_model(T, o2_ratio, t)
            conv = res['conversion']
            A = self.conc0_base[0]
            AR, AAc = res['final_concentrations'][2], res['final_concentrations'][3]
            if conv > 0:
                conversions.append(conv)
                denom = A - AR - AAc
                sel_ar = AR / denom * 100 if denom > 0 else 0
                sel_aa = AAc / denom * 100 if denom > 0 else 0
                sel_acrolein.append(sel_ar)
                sel_acid.append(sel_aa)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=conversions, y=sel_acrolein, name="Селективность акролеина", mode="lines"))
        fig.add_trace(go.Scatter(x=conversions, y=sel_acid, name="Селективность акриловой кислоты", mode="lines"))
        fig.update_layout(xaxis_title="Конверсия пропилена, %", yaxis_title="Селективность, %")
        return fig

    def pfr_along_z(self, T: float, o2_ratio: float, total_length: float = 4.5, n_sections: int = 100):
        times = np.linspace(0, self.tau_end, n_sections)
        z_positions = np.linspace(0, total_length, n_sections)

        res = self.pfr_model(T, o2_ratio, self.tau_end)
        conc_history = res['concentration_history']

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=z_positions, y=[0]*len(z_positions), z=conc_history[:, 2],  # AR
            mode='lines', name="Акролеин", line=dict(color='red')
        ))
        fig.add_trace(go.Scatter3d(
            x=z_positions, y=[1]*len(z_positions), z=conc_history[:, 3],  # AAc
            mode='lines', name="Акриловая кислота", line=dict(color='blue')
        ))

        fig.update_layout(
            title='PFR: Распределение концентраций по длине реактора',
            scene=dict(
                xaxis_title='Длина, м',
                yaxis_title='Компонент (0 - AR, 1 - AAc)',
                zaxis_title='Концентрация, моль/м³'
            )
        )
        return fig

# Веб-интерфейс с Streamlit
def main():
    st.set_page_config(
        page_title="Модель химического реактора",
        page_icon="⚗️",
        layout="wide"
    )
    
    st.title("⚗️ Модель химического реактора для производства акролеина")
    st.markdown("---")
    
    # Инициализация модели
    if 'model' not in st.session_state:
        st.session_state.model = ChemicalReactorModel()
    
    model = st.session_state.model
    
    # Боковая панель для параметров
    st.sidebar.header("Параметры модели")
    
    # Параметры оптимизации
    st.sidebar.subheader("Оптимизация")
    lambda_reg = st.sidebar.slider("Регуляризация λ", 1e-10, 1e-2, 1e-6, format="%.2e")
    
    st.sidebar.subheader("Целевые концентрации")
    target_input = st.sidebar.text_area(
        "Введите 8 целевых концентраций через запятую (моль/м³):",
        value=", ".join(map(str, model.target))
    )
    try:
        user_target = np.array([float(x.strip()) for x in target_input.split(",")])
        if user_target.size == 8:
            model.target = user_target
        else:
            st.sidebar.error("Нужно ввести ровно 8 значений.")
    except Exception as e:
        st.sidebar.error(f"Ошибка в формате: {e}")

    if st.sidebar.button("Оптимизировать параметры"):
        with st.spinner("Оптимизация параметров..."):
            result = model.optimize_parameters(lambda_reg)
            model.optimal_params = result
            st.session_state.optimization_result = result
    
    # Параметры PFR модели
    st.sidebar.subheader("PFR модель")
    temperature = st.sidebar.slider("Температура, K", 450.0, 700.0, 573.15, 1.0)
    o2_ratio = st.sidebar.slider("Соотношение O₂:C₃H₆", 0.5, 5.0, 2.24, 0.1)
    residence_time = st.sidebar.slider("Время пребывания, с", 0.1, 10.0, 3.0, 0.1)
    
    # Основное содержимое
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Анализ модели", "🎯 Оптимизация", "🔬 PFR модель", "📈 3D поверхность"])
    
    with tab1:
        st.header("Анализ химической модели")
        
        if hasattr(st.session_state, 'optimization_result'):
            result = st.session_state.optimization_result
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Оптимальные параметры")
                params_df = pd.DataFrame({
                    'Параметр': [f'A_{i+1}' for i in range(5)] + [f'E_{i+1}' for i in range(5)],
                    'Значение': np.concatenate([result['A_vals'], result['E_vals']]),
                    'Единицы': ['с⁻¹·моль⁻¹·м³']*5 + ['Дж/моль']*5
                })
                st.dataframe(params_df)
            
            with col2:
                st.subheader("Качество оптимизации")
                st.metric("Успешность", "✅" if result['success'] else "❌")
                st.metric("Значение функции цели", f"{result['cost']:.2e}")
        
        else:
            st.info("Запустите оптимизацию параметров в боковой панели")
    
    with tab2:
        st.header("Анализ оптимизации")
        
        if st.button("Создать графики оптимизации"):
            with st.spinner("Создание графиков оптимизации..."):
                figures = model.create_optimization_plots()
                
                for i, fig in enumerate(figures):
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("PFR модель реактора")
        
        if st.button("Расчитать PFR модель"):
            with st.spinner("Расчет PFR модели..."):
                result = model.pfr_model(temperature, o2_ratio, residence_time)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Выход Ar", f"{result['ar_yield']:.2f}%")
                
                with col2:
                    st.metric("Селективность Ar", f"{result['selectivity']:.2f}%")
                
                with col3:
                    st.metric("Конверсия C3H6", f"{result['conversion']:.2f}%")
                
                # График изменения концентраций
                fig = go.Figure()
                
                for i, label in enumerate(model.labels):
                    fig.add_trace(go.Scatter(
                        x=result['times'],
                        y=result['concentration_history'][:, i],
                        mode='lines',
                        name=label
                    ))
                
                fig.update_layout(
                    title='Изменение концентраций во времени (PFR модель)',
                    xaxis_title='Время, с',
                    yaxis_title='Концентрация, моль/м³',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.header("3D поверхность отклика")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Параметры расчета")
            T_min = st.number_input("Мин. температура, K", 450.0, 600.0, 500.0)
            T_max = st.number_input("Макс. температура, K", 600.0, 700.0, 650.0)
            ratio_min = st.number_input("Мин. соотношение O₂:C₃H₆", 0.5, 2.0, 1.0)
            ratio_max = st.number_input("Макс. соотношение O₂:C₃H₆", 2.0, 5.0, 4.0)
            n_points = st.slider("Количество точек", 10, 30, 15)
        
        with col2:
            if st.button("Создать 3D поверхность", type="primary"):
                with st.spinner("Создание 3D поверхности..."):
                    fig_3d = model.create_3d_surface(
                        T_range=(T_min, T_max),
                        ratio_range=(ratio_min, ratio_max),
                        n_points=n_points
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Информация о модели
    with st.expander("ℹ️ Информация о модели"):
        st.markdown("""
        **Химические реакции:**
        1. C₃H₆ + O₂ → C₃H₄O + H₂O (образование акролеина)
        2. C₃H₆ + 1.5O₂ → C₃H₄O₂ + H₂O (образование акриловой кислоты)
        3. C₃H₆ + 3O₂ → 3CO + 3H₂O (глубокое окисление)
        4. C₃H₆ + 4.5O₂ → 3CO₂ + 3H₂O (полное сгорание)
        5. C₃H₆ + 2.5O₂ → C₂H₄O₂ + CO₂ + H₂O (образование уксусной кислоты)
        
        **Модель:** Реактор идеального вытеснения (PFR) с последующим анализом PFR
        
        **Оптимизация:** Метод наименьших квадратов с регуляризацией Тихонова
        """)

if __name__ == "__main__":
    main()
