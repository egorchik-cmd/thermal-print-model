"""
ИНТЕРАКТИВНОЕ ВЕБ-ПРИЛОЖЕНИЕ ДЛЯ ИССЛЕДОВАНИЯ МОДЕЛИ ТЕПЛОМАССОПЕРЕНОСА

Запуск:
  streamlit run app_teplomassoperenos.py

Откроется в браузере: http://localhost:8501
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# КОНФИГУРАЦИЯ STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Модель тепломассопереноса при термопечати",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Численное моделирование тепломассопереноса")
st.subheader("Переводная сублимационная термопечать на текстильные материалы")

# =============================================================================
# ПАРАМЕТРЫ МОДЕЛИ (КОНСТАНТЫ)
# =============================================================================

# Слои
lam1, rho1, cp1, d1 = 16.0, 7850.0, 500.0, 10.0e-3
lam2, rho2, cp2, d2 = 0.08, 700.0, 1300.0, 0.1e-3
lam3, rho3, cp3, d3 = 0.049, 1380.0, 1300.0, 0.3e-3
T0, alpha = 20.0, 10.0

# Диффузия
R_gas = 8.314
Ea = 121.0e3
D0 = 9.1339
Tg = 75.0
R_fiber = 7.5e-6

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

@st.cache_data
def arrhenius_D(T_celsius):
    T_K = np.atleast_1d(np.array(T_celsius, dtype=float)) + 273.15
    D = D0 * np.exp(-Ea / (R_gas * T_K))
    D[np.array(T_celsius) <= Tg] = 0.0
    return D if len(D) > 1 else float(D[0])

def harmonic_mean(a, b):
    result = np.where(a + b > 0, 2.0 * a * b / (a + b), 0.0)
    return result

def build_grid(Nx_per_mm=15):
    n1 = max(2, int(d1 * 1e3 * Nx_per_mm))
    n2 = max(2, int(d2 * 1e3 * Nx_per_mm))
    n3 = max(2, int(d3 * 1e3 * Nx_per_mm))
    x1 = np.linspace(0, d1, n1, endpoint=False)
    x2 = np.linspace(d1, d1+d2, n2, endpoint=False)
    x3 = np.linspace(d1+d2, d1+d2+d3, n3+1)
    x = np.concatenate([x1, x2, x3])
    lam_arr = np.empty(len(x))
    rho_arr = np.empty(len(x))
    cp_arr = np.empty(len(x))
    idx2 = n1
    idx3 = n1 + n2
    lam_arr[:idx2] = lam1; rho_arr[:idx2] = rho1; cp_arr[:idx2] = cp1
    lam_arr[idx2:idx3] = lam2; rho_arr[idx2:idx3] = rho2; cp_arr[idx2:idx3] = cp2
    lam_arr[idx3:] = lam3; rho_arr[idx3:] = rho3; cp_arr[idx3:] = cp3
    return x, lam_arr, rho_arr, cp_arr, idx2, idx3

def solve_heat(T_press, tau, lam_arr, rho_arr, cp_arr, x, safety=0.4):
    N = len(x)
    dx = x[1] - x[0]
    alpha_arr = lam_arr / (rho_arr * cp_arr)
    dt_cfl = 0.5 * dx**2 / np.max(alpha_arr)
    dt = safety * dt_cfl
    
    T = np.full(N, T0)
    T[0] = T_press
    
    t = 0.0
    T_history = [T.copy()]
    t_history = [0.0]
    save_interval = max(1, int(1.0 / dt))
    step = 0
    
    while t < tau:
        dt_step = min(dt, tau - t)
        T_new = T.copy()
        lam_e = harmonic_mean(lam_arr[1:-1], lam_arr[2:])
        lam_w = harmonic_mean(lam_arr[:-2], lam_arr[1:-1])
        flux = (lam_e * (T[2:] - T[1:-1]) - lam_w * (T[1:-1] - T[:-2])) / dx**2
        T_new[1:-1] = T[1:-1] + dt_step / (rho_arr[1:-1] * cp_arr[1:-1]) * flux
        lam_last = lam_arr[-1]
        T_new[-1] = (lam_last / dx * T[-2] + alpha * T0) / (lam_last / dx + alpha)
        T_new[0] = T_press
        T = T_new
        t += dt_step
        step += 1
        
        if step % save_interval == 0 or abs(t - tau) < 1e-9:
            T_history.append(T.copy())
            t_history.append(t)
    
    return T_history, t_history, T

def solve_diffusion(T_at_fiber, tau, R_fiber, Nr=40, safety=0.4):
    dr = R_fiber / (Nr - 1)
    r = np.linspace(0, R_fiber, Nr)
    T_max_mode = np.max(T_at_fiber)
    D_max = float(arrhenius_D(T_max_mode))
    
    if D_max > 0:
        dt_cfl_diff = 0.5 * dr**2 / D_max
        dt = safety * dt_cfl_diff
    else:
        dt = 0.1
    
    C = np.zeros(Nr)
    t = 0.0
    C_history = [C.copy()]
    t_history = [0.0]
    
    while t < tau:
        dt_step = min(dt, tau - t)
        t_frac = t / tau * (len(T_at_fiber) - 1)
        idx_lo = int(t_frac)
        idx_hi = min(idx_lo + 1, len(T_at_fiber) - 1)
        w = t_frac - idx_lo
        T_cur = (1 - w) * T_at_fiber[idx_lo] + w * T_at_fiber[idx_hi]
        D_cur = float(arrhenius_D(T_cur))
        
        if D_cur <= 0:
            t += dt_step
            continue
        
        C_new = C.copy()
        j = np.arange(1, Nr - 1)
        r_e = r[j] + 0.5 * dr
        r_w = r[j] - 0.5 * dr
        flux_e = D_cur * r_e * (C[j + 1] - C[j]) / dr
        flux_w = D_cur * r_w * (C[j] - C[j - 1]) / dr
        C_new[j] = C[j] + dt_step / (r[j] * dr) * (flux_e - flux_w)
        C_new[0] = C[0] + dt_step * 2.0 * D_cur * (C[1] - C[0]) / dr**2
        
        if T_cur > Tg:
            C_new[-1] = 1.0
        C_new[-1] = C_new[-2]
        
        C = np.clip(C_new, 0.0, 1.0)
        t += dt_step
        
        if len(t_history) == 0 or t - t_history[-1] >= 1.0:
            C_history.append(C.copy())
            t_history.append(t)
    
    return C_history, t_history, C

# =============================================================================
# БОКОВАЯ ПАНЕЛЬ (ВВОД ПАРАМЕТРОВ)
# =============================================================================

st.sidebar.markdown("## ⚙️ ПАРАМЕТРЫ РЕЖИМА")
st.sidebar.markdown("---")

# Переключатель между режимами диплома и кастомным режимом
mode_type = st.sidebar.radio(
    "Выбери тип режима:",
    ["📋 Три режима из диплома", "🔧 Свой режим"],
    help="Три режима из диплома уже подобраны оптимально"
)

if mode_type == "📋 Три режима из диплома":
    regime_name = st.sidebar.selectbox(
        "Выбери режим:",
        ["100°C / 60 с — Недостаточный нагрев",
         "200°C / 60 с — ОПТИМАЛЬНЫЙ режим",
         "220°C / 150 с — Интенсивный режим"],
        help="Эти режимы взяты из экспериментальных данных"
    )
    
    if "100°C" in regime_name:
        T_press = 100
        tau = 60
    elif "200°C" in regime_name:
        T_press = 200
        tau = 60
    else:
        T_press = 220
        tau = 150
else:
    T_press = st.sidebar.slider(
        "🌡️ Температура плиты T, °C",
        min_value=80,
        max_value=250,
        value=200,
        step=5,
        help="Диапазон 80-250°C. При T < Tg=75°C краса не проникает."
    )
    
    tau = st.sidebar.slider(
        "⏱️ Время выдержки τ, с",
        min_value=20,
        max_value=300,
        value=60,
        step=10,
        help="Диапазон 20-300 с"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 СВЕДЕНИЯ О ПАРАМЕТРАХ")
st.sidebar.write(f"""
**Выбранный режим:**
- Температура плиты: **{T_press}°C**
- Время выдержки: **{tau} с**

**Критические параметры:**
- Tg (стеклование ПЭТ): **{Tg}°C**
- D₀ (предэксп. множитель): **{D0:.4f} м²/с**
- Eₐ (энергия активации): **{Ea/1000:.0f} кДж/моль**
- Радиус волокна R: **{R_fiber*1e6:.1f} мкм**
""")

# =============================================================================
# ОСНОВНАЯ ОБЛАСТЬ: РАСЧЁТ И ВИЗУАЛИЗАЦИЯ
# =============================================================================

# Построение сетки
x, lam_arr, rho_arr, cp_arr, idx2, idx3 = build_grid()
idx_mid = idx3 + (len(x) - idx3) // 2

# Прогресс
progress_placeholder = st.empty()
progress_placeholder.info("⏳ Выполняю расчёты... (это займёт 3-5 сек)")

# Решение тепловой задачи
T_hist, t_hist, T_final = solve_heat(T_press, tau, lam_arr, rho_arr, cp_arr, x)
T_fabric_history = np.array([T[idx_mid] for T in T_hist])
T_max_fabric = T_fabric_history[-1]

# Решение диффузионной задачи
C_hist, t_c_hist, C_final = solve_diffusion(T_fabric_history, tau, R_fiber)
r_grid = np.linspace(0, R_fiber, len(C_final))
C_axis = C_final[0]

# Время выхода на полку (95% от T_press)
t_shelf = None
for i, T_val in enumerate(T_fabric_history):
    if T_val >= 0.95 * T_press:
        t_shelf = t_hist[i]
        break

# Глубина проникновения
pen_depth = 0.0
threshold_C = 0.01
for i in range(len(r_grid)):
    if C_final[i] >= threshold_C:
        pen_depth = R_fiber - r_grid[i]
        break

progress_placeholder.empty()

# =============================================================================
# ГЛАВНЫЕ РЕЗУЛЬТАТЫ (БОЛЬШИЕ ЧИСЛА)
# =============================================================================

st.markdown("---")
st.markdown("## 📈 РЕЗУЛЬТАТЫ РАСЧЁТА")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="T макс в ткани",
        value=f"{T_max_fabric:.1f}°C",
        delta=f"{T_max_fabric - Tg:.1f}°C выше Tg",
        delta_color="normal" if T_max_fabric > Tg else "inverse"
    )

with col2:
    st.metric(
        label="t выхода на полку",
        value=f"{t_shelf:.0f} с" if t_shelf else ">τ",
        help="Время, когда T достигла 95% от T_press"
    )

with col3:
    st.metric(
        label="Глубина проникновения",
        value=f"{pen_depth*1e6:.2f} мкм",
        help=f"От поверхности (R = {R_fiber*1e6:.1f} мкм)"
    )

with col4:
    st.metric(
        label="C/Cs на оси волокна",
        value=f"{C_axis:.4f}",
        help="Нормированная концентрация красителя на оси волокна"
    )

# =============================================================================
# ГРАФИКИ (2 СТОЛБЦА)
# =============================================================================

st.markdown("---")
st.markdown("## 📊 ГРАФИКИ ПРОЦЕССА")

col_T, col_C = st.columns(2)

# --- ГРАФИК 1: ТЕМПЕРАТУРА T(x,t) ---
with col_T:
    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    
    # Несколько временных срезов
    n_snap = min(6, len(T_hist))
    colors_snap = plt.cm.plasma(np.linspace(0.1, 0.9, n_snap))
    
    for k in range(n_snap):
        idx = int(k * (len(T_hist) - 1) / (n_snap - 1))
        t_val = t_hist[idx]
        ax_t.plot(x * 1e3, T_hist[idx], color=colors_snap[k], lw=2.5,
                  label=f"t = {t_val:.0f} с", alpha=0.8)
    
    # Оформление
    ax_t.axhline(Tg, color='green', ls='--', lw=2, alpha=0.8, label=f'Tg = {Tg}°C')
    ax_t.axvline(d1 * 1e3, color='grey', ls=':', lw=1.5, alpha=0.5)
    ax_t.axvline((d1 + d2) * 1e3, color='grey', ls=':', lw=1.5, alpha=0.5)
    
    # Добавить прямоугольники для слоёв
    ax_t.axvspan(0, d1*1e3, alpha=0.1, color='red', label='Плита')
    ax_t.axvspan(d1*1e3, (d1+d2)*1e3, alpha=0.1, color='yellow', label='Бумага')
    ax_t.axvspan((d1+d2)*1e3, (d1+d2+d3)*1e3, alpha=0.1, color='blue', label='Ткань')
    
    ax_t.set_xlabel('Координата x, мм', fontsize=11, fontweight='bold')
    ax_t.set_ylabel('Температура T, °C', fontsize=11, fontweight='bold')
    ax_t.set_title(f'Распространение тепла через пакет\n(режим {T_press}°C / {tau}с)', 
                   fontsize=12, fontweight='bold')
    ax_t.legend(fontsize=9, loc='best')
    ax_t.grid(True, alpha=0.3)
    ax_t.set_ylim([T0 - 5, T_press * 1.05])
    
    st.pyplot(fig_t, use_container_width=True)

# --- ГРАФИК 2: КОНЦЕНТРАЦИЯ C(r,t) ---
with col_C:
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    
    # Несколько временных срезов диффузии
    n_snap_c = min(6, len(C_hist))
    colors_snap_c = plt.cm.hot(np.linspace(0.2, 1, n_snap_c))
    
    for k in range(n_snap_c):
        idx = int(k * (len(C_hist) - 1) / (n_snap_c - 1))
        t_val = t_c_hist[idx]
        ax_c.plot(r_grid / R_fiber, C_hist[idx], color=colors_snap_c[k], lw=2.5,
                  marker='o', markersize=3, label=f"t = {t_val:.0f} с", alpha=0.8)
    
    # Заливка последнего профиля
    ax_c.fill_between(r_grid / R_fiber, 0, C_hist[-1], alpha=0.15, color='red')
    
    ax_c.set_xlabel('Радиус r / R', fontsize=11, fontweight='bold')
    ax_c.set_ylabel('Концентрация C / Cs', fontsize=11, fontweight='bold')
    ax_c.set_title(f'Проникновение красителя в волокно\n(режим {T_press}°C / {tau}с)',
                   fontsize=12, fontweight='bold')
    ax_c.set_xlim([0, 1])
    ax_c.set_ylim([-0.05, 1.1])
    ax_c.legend(fontsize=9, loc='best')
    ax_c.grid(True, alpha=0.3)
    
    # Аннотация
    if C_axis < 0.01:
        annotation = "❌ C ≈ 0: Краски нет"
    elif C_axis < 0.5:
        annotation = "⚠️ C < 0.5: Частичное проникновение"
    elif C_axis < 0.9:
        annotation = "✓ C ≈ 0.5–0.9: Хорошее проникновение"
    else:
        annotation = "✓✓ C ≈ 1: Полное насыщение"
    
    ax_c.text(0.5, 1.05, annotation, ha='center', fontsize=10, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    st.pyplot(fig_c, use_container_width=True)

# =============================================================================
# ТАБЛИЦА СРАВНЕНИЯ С ДИАГРАММОЙ
# =============================================================================

st.markdown("---")
st.markdown("## 📋 ДИАГНОСТИКА РЕЖИМА")

# Фазовая диаграмма
diag_col1, diag_col2 = st.columns([2, 1])

with diag_col1:
    # Маленькая фазовая диаграмма с отметкой
    fig_diag, ax_diag = plt.subplots(figsize=(10, 7))
    
    T_range_diag = np.linspace(80, 250, 12)
    tau_range_diag = np.linspace(20, 300, 12)
    C_grid = np.zeros((len(T_range_diag), len(tau_range_diag)))
    
    for i, T_d in enumerate(T_range_diag):
        for j, tau_d in enumerate(tau_range_diag):
            T_h_d, _, T_f_d = solve_heat(T_d, tau_d, lam_arr, rho_arr, cp_arr, x)
            T_fab_d = np.array([T[idx_mid] for T in T_h_d])
            _, _, C_f_d = solve_diffusion(T_fab_d, tau_d, R_fiber)
            C_grid[i, j] = C_f_d[0]
    
    # Контурный график
    im = ax_diag.contourf(tau_range_diag, T_range_diag, C_grid, levels=20, cmap='RdYlGn')
    cs = ax_diag.contour(tau_range_diag, T_range_diag, C_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                         colors='black', alpha=0.3, linewidths=0.5)
    ax_diag.clabel(cs, inline=True, fontsize=8)
    
    # Отметить текущий режим
    ax_diag.plot(tau, T_press, 'r*', markersize=30, markeredgecolor='darkred', markeredgewidth=2,
                 label=f'Текущий режим\n({T_press}°C, {tau}с)')
    
    # Отметить три режима диплома
    modes_diplom = [(60, 100), (60, 200), (150, 220)]
    for tau_d, T_d in modes_diplom:
        ax_diag.plot(tau_d, T_d, 'bs', markersize=8, alpha=0.7)
    
    plt.colorbar(im, ax=ax_diag, label='C/Cs на оси')
    ax_diag.set_xlabel('Время выдержки τ, с', fontsize=11, fontweight='bold')
    ax_diag.set_ylabel('Температура плиты T, °C', fontsize=11, fontweight='bold')
    ax_diag.set_title('Фазовая диаграмма: карта концентраций', fontsize=12, fontweight='bold')
    ax_diag.legend(fontsize=10, loc='upper right')
    ax_diag.grid(True, alpha=0.2)
    
    st.pyplot(fig_diag, use_container_width=True)

with diag_col2:
    st.markdown("### Интерпретация:")
    
    if C_axis < 0.01:
        st.error(f"""
        ❌ **Режим недостаточен**
        
        Краситель практически не проникает.
        
        **Причина:** T = {T_press}°C < {Tg}°C + 15°C
        
        D(T) ≈ 0 → диффузия замёрзнута
        """)
    elif 0.1 <= C_axis <= 0.8:
        st.success(f"""
        ✓ **Режим оптимален**
        
        Краситель проникает хорошо.
        
        **Характеристика:** Частичное насыщение
        
        C/Cs = {C_axis:.3f} — идеальное соотношение
        """)
    else:
        st.warning(f"""
        ⚠️ **Режим интенсивен**
        
        Краситель полностью проникает.
        
        **Риск:** Возможна деградация красителя
        
        T = {T_press}°C может быть слишком высокой
        """)

# =============================================================================
# НИЖНЯЯ ИНФОРМАЦИОННАЯ ПАНЕЛЬ
# =============================================================================

st.markdown("---")
st.markdown("## ℹ️ ФИЗИЧЕСКИЙ СМЫСЛ")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    ### 🌡️ Коэффициент диффузии
    
    D(T) зависит от температуры по закону Аррениуса:
    
    $$D(T) = D_0 \\cdot \\exp\\left(-\\frac{E_a}{RT}\\right)$$
    
    где:
    - D₀ = 9.13 м²/с
    - Eₐ = 121 кДж/моль
    - T в Кельвинах
    
    **При 100°C:** D ≈ 10⁻¹⁶ м²/с (ОЧЕНЬ мало!)
    **При 200°C:** D ≈ 4·10⁻¹³ м²/с (нормально)
    **При 220°C:** D ≈ 1.4·10⁻¹² м²/с (быстро)
    """)

with info_col2:
    st.markdown(f"""
    ### ⏱️ Характерное время диффузии
    
    τ_diff = R² / (6D)
    
    Это время, за которое краситель проникает на всю глубину волокна.
    
    **Для этого режима:**
    
    D(T) = {float(arrhenius_D(T_max_fabric)):.2e} м²/с
    
    τ_diff = {(R_fiber**2 / (6*max(float(arrhenius_D(T_max_fabric)), 1e-20))):.1f} с
    
    ✓ Если τ > τ_diff, краска проникает полностью
    ✗ Если τ < τ_diff, только частичное проникновение
    """)

with info_col3:
    st.markdown(f"""
    ### 📊 Условие устойчивости МКР
    
    Метод конечных разностей требует:
    
    **Для тепла (Fo):**
    Fo = α·dt/dx² < 0.5
    
    **Для диффузии (Fo):**
    Fo = D·dt/dr² < 0.5
    
    В коде используется safety = 0.4 для надёжности.
    
    ✓ Оба условия соблюдены
    """)

# =============================================================================
# НИЖНИЙ FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>

**Приложение для численного моделирования тепломассопереноса при переводной сублимационной термопечати**

Магистерская ВКР, направление 20.04.01 «Техносферная безопасность»

Модель реализована методом конечных разностей с явной схемой. 
Граничные условия соответствуют уравнениям из Главы 3 диплома.

</div>
""", unsafe_allow_html=True)
