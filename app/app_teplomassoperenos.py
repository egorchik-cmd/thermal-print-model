"""
ПРИЛОЖЕНИЕ: Численное моделирование тепломассопереноса при термопечати
ВКР, направление 20.04.01 «Техносферная безопасность»

Запуск: streamlit run термопечать.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Модель термопечати",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background: white; border-radius: 12px; padding: 10px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .stMetric label { font-size: 13px !important; color: #666 !important; }
    div[data-testid="metric-container"] {
        background: white; border-radius: 12px; padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .verdict-ok   { background:#e8f5e9; border-left:4px solid #4caf50;
                    padding:12px 16px; border-radius:4px; color:#2e7d32; }
    .verdict-bad  { background:#fce4ec; border-left:4px solid #e91e63;
                    padding:12px 16px; border-radius:4px; color:#880e4f; }
    .verdict-warn { background:#fff8e1; border-left:4px solid #ff9800;
                    padding:12px 16px; border-radius:4px; color:#e65100; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# ФИЗИЧЕСКИЕ КОНСТАНТЫ (Таблица 3.1 и уравнения 3.8–3.9)
# ─────────────────────────────────────────────────────────────────
# Слой 1: стальная плита (Михеев [25])
LAM1, RHO1, CP1, D1 = 16.0, 7850.0, 500.0, 10.0e-3

# Слой 2: бумага-носитель (Kalaoglu-Altan [24])
LAM2, RHO2, CP2, D2 = 0.08, 700.0, 1300.0, 0.1e-3

# Слой 3: ткань (эффективные свойства, Kalaoglu-Altan [24])
LAM3, RHO3, CP3, D3 = 0.049, 1380.0, 1300.0, 0.3e-3

# Граничные условия
T0    = 20.0    # начальная температура, °C
ALPHA = 10.0    # конвекция с обратной стороны ткани, Вт/(м²·К)

# Диффузия (Park 2020 [26], Burkinshaw 2024 [27])
R_GAS = 8.314
EA    = 121.0e3     # энергия активации, Дж/моль
D0    = 9.1339      # предэкспоненциальный множитель, м²/с
TG    = 75.0        # температура стеклования ПЭТ, °C

# Геометрия волокон (СЭМ-измерения, п. 3.1 [1])
R_PET  = 7.5e-6     # радиус ПЭТ-волокна, м (d = 15 мкм)
R_SILK = 6.3e-6     # радиус шёлкового волокна, м (d = 12.6 мкм)

# ─────────────────────────────────────────────────────────────────
# ФУНКЦИИ МОДЕЛИ
# ─────────────────────────────────────────────────────────────────

def D_arrhenius(T_c):
    """Коэффициент диффузии по Аррениусу (ур. 3.8–3.9)."""
    T_c = np.asarray(T_c, dtype=float)
    T_K = T_c + 273.15
    D = D0 * np.exp(-EA / (R_GAS * T_K))
    D = np.where(T_c <= TG, 0.0, D)
    return float(D) if D.ndim == 0 else D

def harmonic(a, b):
    """Гармоническое среднее (ур. 3.15)."""
    return np.where(a + b > 0, 2*a*b/(a+b), 0.0)

def build_grid(Nx=15):
    """Строит равномерную сетку через три слоя."""
    n1 = max(2, int(D1*1e3*Nx)); n2 = max(2, int(D2*1e3*Nx))
    n3 = max(2, int(D3*1e3*Nx))
    x = np.concatenate([
        np.linspace(0, D1, n1, endpoint=False),
        np.linspace(D1, D1+D2, n2, endpoint=False),
        np.linspace(D1+D2, D1+D2+D3, n3+1)
    ])
    N = len(x)
    lam = np.empty(N); rho = np.empty(N); cp = np.empty(N)
    i2, i3 = n1, n1+n2
    lam[:i2]=LAM1; rho[:i2]=RHO1; cp[:i2]=CP1
    lam[i2:i3]=LAM2; rho[i2:i3]=RHO2; cp[i2:i3]=CP2
    lam[i3:]=LAM3; rho[i3:]=RHO3; cp[i3:]=CP3
    return x, lam, rho, cp, i2, i3

@st.cache_data(show_spinner=False)
def solve_heat(T_press: float, tau: int, safety: float = 0.4):
    """Решает уравнение теплопроводности (3.1), явная схема (3.14)."""
    x, lam, rho, cp, i2, i3 = build_grid()
    dx = x[1] - x[0]
    dt_cfl = 0.5 * dx**2 / np.max(lam/(rho*cp))
    dt = safety * dt_cfl

    T = np.full(len(x), T0)
    T[0] = T_press

    # Сохраняем до ~20 срезов равномерно
    save_every = max(1, int(tau / (20 * dt)))
    snaps_T, snaps_t = [T.copy()], [0.0]
    t, step = 0.0, 0

    while t < tau:
        ds = min(dt, tau - t)
        Tn = T.copy()
        le = harmonic(lam[1:-1], lam[2:])
        lw = harmonic(lam[:-2], lam[1:-1])
        flux = (le*(T[2:]-T[1:-1]) - lw*(T[1:-1]-T[:-2])) / dx**2
        Tn[1:-1] = T[1:-1] + ds/(rho[1:-1]*cp[1:-1]) * flux
        # ГУ конвекция (3.4)
        Tn[-1] = (lam[-1]/dx*T[-2] + ALPHA*T0) / (lam[-1]/dx + ALPHA)
        Tn[0]  = T_press   # ГУ Дирихле (3.3)
        T = Tn
        t += ds; step += 1
        if step % save_every == 0 or abs(t - tau) < 1e-9:
            snaps_T.append(T.copy())
            snaps_t.append(t)

    return x, np.array(snaps_T), np.array(snaps_t), i3

@st.cache_data(show_spinner=False)
def solve_diffusion(T_fiber_arr: tuple, tau: int, R_fiber: float,
                    fiber_type: str = "PET", Nr: int = 40, safety: float = 0.4):
    """
    Решает уравнение диффузии Фика (3.7) в цилиндрических координатах.
    T_fiber_arr — кортеж из (значения температуры в ткани, шаги времени).
    """
    T_fab, t_fab = np.array(T_fiber_arr[0]), np.array(T_fiber_arr[1])
    dr   = R_fiber / (Nr - 1)
    r    = np.linspace(0, R_fiber, Nr)
    D_mx = float(D_arrhenius(np.max(T_fab)))
    dt   = (safety * dr**2 / D_mx) if D_mx > 0 else 1.0

    C = np.zeros(Nr)
    snaps_C, snaps_t = [C.copy()], [0.0]
    save_every = max(1, int(tau / (20 * dt)))
    t, step = 0.0, 0

    while t < tau:
        ds = min(dt, tau - t)
        # Текущая T в слое ткани (интерполяция)
        frac   = t / tau * (len(T_fab)-1)
        lo, hi = int(frac), min(int(frac)+1, len(T_fab)-1)
        w      = frac - lo
        T_cur  = (1-w)*T_fab[lo] + w*T_fab[hi]
        D_cur  = float(D_arrhenius(T_cur))

        if D_cur > 0:
            Cn = C.copy()
            j  = np.arange(1, Nr-1)
            re = r[j] + 0.5*dr;  rw = r[j] - 0.5*dr
            Cn[j] = C[j] + ds/(r[j]*dr) * (
                D_cur*re*(C[j+1]-C[j]) - D_cur*rw*(C[j]-C[j-1])
            ) / dr
            # Симметрия r=0 (3.12)
            Cn[0] = C[0] + ds * 2*D_cur*(C[1]-C[0]) / dr**2
            # ГУ на поверхности r=R
            if fiber_type == "PET":
                Cn[-1] = 1.0 if T_cur > TG else 0.0   # Дирихле (3.11)
            else:
                Cn[-1] = Cn[-2]                         # Неймана (3.13)
            C = np.clip(Cn, 0.0, 1.0)

        t += ds; step += 1
        if step % save_every == 0 or abs(t - tau) < 1e-9:
            snaps_C.append(C.copy())
            snaps_t.append(t)

    return r, np.array(snaps_C), np.array(snaps_t)

# ─────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────

def depth_at_half(C_profile, r_grid):
    """Глубина, где C = 0.5*Cs (от поверхности — то, что написано в главе 3 как R/2)."""
    for i in range(len(r_grid)-1, -1, -1):
        if C_profile[i] <= 0.5:
            return (r_grid[-1] - r_grid[i]) * 1e6   # мкм
    return 0.0

def t_shelf(T_history, t_history, T_press):
    """Время выхода температуры ткани на 95% от T_press."""
    for i, T in enumerate(T_history):
        if T >= 0.95 * T_press:
            return t_history[i]
    return None

# ─────────────────────────────────────────────────────────────────
# ЗАГОЛОВОК
# ─────────────────────────────────────────────────────────────────

st.title("🧵 Тепломассоперенос при термопечати")
st.caption(
    "Численная модель переводной сублимационной термопечати — "
    "уравнения теплопроводности (3.1) и диффузии Фика (3.7) | ВКР 20.04.01"
)
st.divider()

# ─────────────────────────────────────────────────────────────────
# БОКОВАЯ ПАНЕЛЬ — ВВОД ПАРАМЕТРОВ
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Параметры режима")

    mode = st.radio(
        "Режим ввода",
        ["Задать вручную", "Примеры из диплома"],
        help="В дипломе исследованы три показательных режима"
    )

    if mode == "Примеры из диплома":
        preset = st.selectbox(
            "Выбери пример",
            [
                "100°C / 60 с — Недостаточный нагрев",
                "200°C / 60 с — Оптимальный режим",
                "220°C / 150 с — Предельный режим",
            ]
        )
        if "100" in preset:   T_press, tau = 100, 60
        elif "200" in preset: T_press, tau = 200, 60
        else:                 T_press, tau = 220, 150
        st.info(f"Режим: **{T_press}°C / {tau} с**")
    else:
        T_press = st.slider("Температура плиты, °C", 80, 220, 200, 5,
                            help="Рабочий диапазон сублимационной термопечати")
        tau = st.slider("Время выдержки, с", 20, 300, 60, 10,
                        help="Типичный диапазон: 30–180 с")

    material = st.radio(
        "Ткань",
        ["Полиэстер (ПЭТ)", "Шёлк"],
        help="ПЭТ: ГУ Дирихле C=Cs на поверхности волокна  |  Шёлк: ГУ Неймана (непроницаем)"
    )
    fiber_type = "PET" if "ПЭТ" in material else "SILK"
    R_fiber = R_PET if fiber_type == "PET" else R_SILK

    st.divider()
    st.caption("**Параметры модели (константы)**")
    st.markdown(f"""
    | Параметр | Значение |
    |---|---|
    | Tg (ПЭТ) | {TG}°C |
    | Eₐ | {EA/1e3:.0f} кДж/моль |
    | D₀ | {D0:.4f} м²/с |
    | R волокна | {R_fiber*1e6:.1f} мкм |
    | D при {T_press}°C | {D_arrhenius(T_press):.2e} м²/с |
    """)

# ─────────────────────────────────────────────────────────────────
# РАСЧЁТ
# ─────────────────────────────────────────────────────────────────

with st.spinner("Выполняю расчёт..."):
    x, T_snaps, t_snaps, i3 = solve_heat(float(T_press), int(tau))

    # Температура в середине слоя ткани
    i_mid = i3 + (len(x) - i3) // 2
    T_fab_hist = T_snaps[:, i_mid]

    # Диффузионная задача
    r_grid, C_snaps, t_c_snaps = solve_diffusion(
        (tuple(T_fab_hist.tolist()), tuple(t_snaps.tolist())),
        int(tau), float(R_fiber), fiber_type
    )

# Ключевые числа
T_max    = float(T_fab_hist[-1])
C_final  = C_snaps[-1]
C_axis   = float(C_final[0])
depth_50 = depth_at_half(C_final, r_grid)   # глубина до C=0.5 (как в тексте гл. 3)
t_sh     = t_shelf(T_fab_hist, t_snaps, T_press)
D_at_T   = D_arrhenius(T_max)
tau_diff = (R_fiber**2) / (6*D_at_T) if D_at_T > 0 else float("inf")

# ─────────────────────────────────────────────────────────────────
# МЕТРИКИ
# ─────────────────────────────────────────────────────────────────

st.subheader("📊 Ключевые результаты расчёта")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("🌡️ T макс в ткани",
          f"{T_max:.1f} °C",
          f"{T_max - TG:+.1f}°C к Tg",
          delta_color="normal" if T_max > TG else "inverse")

c2.metric("⏱️ t выхода на полку",
          f"{t_sh:.0f} с" if t_sh else f"> {tau} с",
          help="Время достижения 95% от T_press в ткани")

c3.metric("🎨 C/Cs на оси волокна",
          f"{C_axis:.4f}",
          help="0 = краски нет, 1 = полное насыщение")

c4.metric("📐 Глубина (до C=0.5)",
          f"{depth_50:.2f} мкм",
          f"R = {R_fiber*1e6:.1f} мкм",
          help="Расстояние от поверхности волокна до точки C = 0.5·Cs")

c5.metric("⚡ D при T_макс",
          f"{D_at_T:.1e} м²/с",
          f"τ_diff = {tau_diff:.1f} с" if np.isfinite(tau_diff) else "D = 0",
          help="Коэффициент диффузии Аррениуса при T_макс")

st.divider()

# ─────────────────────────────────────────────────────────────────
# ВЕРДИКТ О РЕЖИМЕ
# ─────────────────────────────────────────────────────────────────

if C_axis < 0.01:
    st.markdown(
        f'<div class="verdict-bad">❌ <strong>Недостаточный нагрев.</strong> '
        f'T_ткани = {T_max:.0f}°C — краситель не диффундирует (D ≈ 0). '
        f'Нет цветопереноса.</div>', unsafe_allow_html=True
    )
elif C_axis < 0.5:
    st.markdown(
        f'<div class="verdict-ok">✅ <strong>Оптимальный режим.</strong> '
        f'C/Cs на оси = {C_axis:.3f} — частичное равномерное проникновение красителя. '
        f'Ожидается чёткий насыщенный цвет на ткани.</div>', unsafe_allow_html=True
    )
elif C_axis < 0.95:
    st.markdown(
        f'<div class="verdict-warn">⚠️ <strong>Интенсивный режим.</strong> '
        f'C/Cs = {C_axis:.3f} — глубокое проникновение. '
        f'Возможно расплывание контура изображения.</div>', unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div class="verdict-warn">⚠️ <strong>Предельный режим.</strong> '
        f'Полное насыщение сечения волокна (C ≈ Cs по всему радиусу). '
        f'Высокий риск потери чёткости и деградации красителя.</div>', unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────────────────────────
# ГЛАВНЫЕ ГРАФИКИ
# ─────────────────────────────────────────────────────────────────

st.subheader("📈 Графики процесса")
tab1, tab2, tab3 = st.tabs([
    "🌡️ Температурное поле T(x,t)",
    "🎨 Проникновение красителя C(r,t)",
    "🔬 Схема процесса"
])

# --- ВКЛАДКА 1: ТЕМПЕРАТУРА ---
with tab1:
    st.caption(
        "Как температура распространяется сквозь пакет «плита–бумага–ткань» во времени. "
        "Горизонтальная линия Tg = 75°C — порог, ниже которого краситель не диффундирует."
    )

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_facecolor("#fafafa")
    fig1.patch.set_facecolor("white")

    # Закрашиваем слои
    ax1.axvspan(0,               D1*1e3,         alpha=0.07, color="#e53935", label="Плита (сталь)")
    ax1.axvspan(D1*1e3,         (D1+D2)*1e3,    alpha=0.12, color="#f9a825", label="Бумага-носитель")
    ax1.axvspan((D1+D2)*1e3,    (D1+D2+D3)*1e3, alpha=0.10, color="#1e88e5", label="Ткань")

    # Вертикальные границы
    for xv in [D1*1e3, (D1+D2)*1e3]:
        ax1.axvline(xv, color="#999", ls="--", lw=1.0, alpha=0.6)

    # Линия Tg
    ax1.axhline(TG, color="#43a047", ls="-.", lw=1.5, alpha=0.8, label=f"Tg = {TG}°C (порог диффузии)")

    # Температурные профили (6 срезов)
    n_total = len(t_snaps)
    idx_list = np.unique(np.round(np.linspace(0, n_total-1, 6)).astype(int))
    cmap = plt.cm.plasma(np.linspace(0.15, 0.92, len(idx_list)))

    for k, idx in enumerate(idx_list):
        lbl = f"t = {t_snaps[idx]:.0f} с"
        lw  = 2.5 if idx == idx_list[-1] else 1.5
        ax1.plot(x*1e3, T_snaps[idx], color=cmap[k], lw=lw, label=lbl)

    # Аннотация T_max в ткани
    x_fabric_mid = x[i_mid] * 1e3
    ax1.annotate(
        f"T_макс в ткани\n= {T_max:.1f}°C",
        xy=(x_fabric_mid, T_max),
        xytext=(x_fabric_mid - 1.5, T_max - 25),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=9, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
    )

    ax1.set_xlabel("Координата x, мм  (поверхность плиты → через бумагу → ткань)", fontsize=11)
    ax1.set_ylabel("Температура T, °C", fontsize=11)
    ax1.set_title(f"Нагрев пакета при {T_press}°C / {tau} с", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, (D1+D2+D3)*1e3)
    ax1.set_ylim(T0 - 5, T_press * 1.08)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls="--")

    # Подписи слоёв под графиком
    for label, xc in [
        ("Плита", D1/2), ("Бумага", D1 + D2/2), ("Ткань", D1+D2+D3/2)
    ]:
        ax1.text(xc*1e3, T0 + 3, label, ha="center", fontsize=9,
                 color="#555", style="italic")

    st.pyplot(fig1, use_container_width=True)

    st.info(
        f"**Что видно:** волна тепла от плиты ({T_press}°C) идёт слева направо "
        f"через бумагу в ткань. "
        f"Наиболее тонкая кривая — начало процесса (t=0), наиболее яркая — конец (t={tau}с). "
        f"Ткань прогревается до **{T_max:.1f}°C** — "
        + ("**выше Tg: диффузия красителя запущена.**" if T_max > TG
           else "**ниже Tg: диффузия невозможна.**")
    )

# --- ВКЛАДКА 2: КОНЦЕНТРАЦИЯ ---
with tab2:
    if fiber_type == "SILK":
        st.warning(
            "**Шёлковое волокно:** ГУ Неймана на поверхности (уравнение 3.13) — "
            "дисперсный краситель не проникает внутрь белковой матрицы фиброина. "
            "Концентрация C = 0 по всему сечению во всех режимах."
        )
    else:
        st.caption(
            "Как краситель проникает внутрь волокна ПЭТ во времени. "
            "Левый край (r/R = 0) — ось волокна, правый (r/R = 1) — поверхность. "
            "Поверхность держится при C = Cs (граничное условие 3.11)."
        )

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.patch.set_facecolor("white")

    # Левый: профиль C(r) в момент τ
    ax_l = axes2[0]
    ax_l.set_facecolor("#fafafa")

    # Несколько временных срезов
    n_c = len(t_c_snaps)
    idx_c_list = np.unique(np.round(np.linspace(0, n_c-1, 5)).astype(int))
    cmap2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(idx_c_list)))

    for k, idx in enumerate(idx_c_list):
        lw = 3.0 if idx == idx_c_list[-1] else 1.2
        al = 1.0 if idx == idx_c_list[-1] else 0.5
        ax_l.plot(r_grid/R_fiber, C_snaps[idx], color=cmap2[k], lw=lw, alpha=al,
                  label=f"t = {t_c_snaps[idx]:.0f} с")

    ax_l.fill_between(r_grid/R_fiber, 0, C_snaps[-1], alpha=0.12, color="#e53935")

    # Отметить C_axis и глубину
    ax_l.axhline(0.5, color="#555", ls=":", lw=1.2, alpha=0.7, label="C = 0.5·Cs")
    ax_l.plot(0, C_axis, "r*", ms=12, label=f"C на оси = {C_axis:.4f}")

    ax_l.set_xlabel("r / R    (ось волокна ← → поверхность)", fontsize=11)
    ax_l.set_ylabel("Нормированная концентрация  C / Cs", fontsize=11)
    ax_l.set_title(f"Профиль C(r,t)  |  {T_press}°C / {tau} с", fontsize=12, fontweight="bold")
    ax_l.set_xlim(0, 1); ax_l.set_ylim(-0.05, 1.12)
    ax_l.legend(fontsize=9); ax_l.grid(True, alpha=0.3, ls="--")

    # Правый: тепловая карта C(r,t) — двумерная картина эволюции
    ax_r = axes2[1]
    if n_c > 3:
        C_mat = C_snaps          # shape: (время, радиус)
        t_ax  = t_c_snaps
        r_ax  = r_grid / R_fiber
        im = ax_r.pcolormesh(r_ax, t_ax, C_mat, cmap="RdYlGn",
                             vmin=0, vmax=1, shading="auto")
        fig2.colorbar(im, ax=ax_r, label="C / Cs")
        ax_r.set_xlabel("r / R    (ось ← → поверхность)", fontsize=11)
        ax_r.set_ylabel("Время t, с", fontsize=11)
        ax_r.set_title("Тепловая карта концентрации  C(r, t)", fontsize=12, fontweight="bold")
        ax_r.grid(False)

    st.pyplot(fig2, use_container_width=True)

    # Интерпретация
    if C_axis < 0.01:
        st.info("**Вывод:** Краситель не проникает — D = 0 (T < Tg). Профиль C(r) = 0.")
    elif C_axis < 0.5:
        st.info(
            f"**Вывод:** Краситель проник на глубину **≈ {depth_50:.1f} мкм** "
            f"(до уровня C = 0.5·Cs от поверхности). "
            f"Характерное время диффузии τ_diff = R²/6D = {tau_diff:.0f} с. "
            f"За τ = {tau} с заполнено ~{tau/tau_diff*100:.0f}% от полного насыщения."
        )
    else:
        st.info(
            f"**Вывод:** Сечение волокна насыщено на {C_axis*100:.0f}%. "
            f"τ_diff = {tau_diff:.0f} с — выдержка τ = {tau} с значительно превышает "
            f"характерное время диффузии."
        )

# --- ВКЛАДКА 3: СХЕМА ---
with tab3:
    st.caption("Визуальная схема физики процесса — что именно считает модель.")

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    fig3.patch.set_facecolor("white")
    ax3.set_facecolor("white")
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 6)
    ax3.axis("off")

    # Слой 1: плита
    ax3.add_patch(plt.Rectangle((0.5, 1), 2, 4, color="#ef9a9a", alpha=0.8, zorder=2))
    ax3.text(1.5, 5.3, "ПЛИТА", ha="center", fontsize=11, fontweight="bold", color="#b71c1c")
    ax3.text(1.5, 4.5, f"λ = {LAM1} Вт/(м·К)", ha="center", fontsize=9, color="#555")
    ax3.text(1.5, 4.0, f"δ = {D1*1e3:.0f} мм", ha="center", fontsize=9, color="#555")
    ax3.text(1.5, 3.3, f"T = {T_press}°C", ha="center", fontsize=11,
             fontweight="bold", color="#b71c1c",
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Слой 2: бумага
    ax3.add_patch(plt.Rectangle((2.5, 1), 0.5, 4, color="#fff176", alpha=0.9, zorder=2))
    ax3.text(2.75, 5.3, "БУМАГА", ha="center", fontsize=9, fontweight="bold", color="#f57f17")
    ax3.text(2.75, 1.4, f"δ = {D2*1e3:.1f} мм", ha="center", fontsize=8, color="#555")

    # Слой 3: ткань
    ax3.add_patch(plt.Rectangle((3.0, 1), 1.5, 4, color="#90caf9", alpha=0.7, zorder=2))
    ax3.text(3.75, 5.3, "ТКАНЬ", ha="center", fontsize=11, fontweight="bold", color="#0d47a1")
    ax3.text(3.75, 4.5, f"λ = {LAM3} Вт/(м·К)", ha="center", fontsize=9, color="#555")
    ax3.text(3.75, 4.0, f"δ = {D3*1e3:.1f} мм", ha="center", fontsize=9, color="#555")
    ax3.text(3.75, 3.3, f"T_макс = {T_max:.0f}°C", ha="center", fontsize=10,
             fontweight="bold", color="#0d47a1",
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Стрелка теплового потока
    ax3.annotate("", xy=(3.8, 3), xytext=(1.8, 3),
                 arrowprops=dict(arrowstyle="->", lw=2, color="#e53935"))
    ax3.text(2.8, 3.2, "Тепловой поток\nур. (3.1)", ha="center", fontsize=9, color="#e53935")

    # Волокно
    theta = np.linspace(0, 2*np.pi, 100)
    r_vis = 0.6
    cx, cy = 7.0, 3.0
    ax3.plot(cx + r_vis*np.cos(theta), cy + r_vis*np.sin(theta), "b-", lw=2, zorder=4)
    ax3.fill(cx + r_vis*np.cos(theta), cy + r_vis*np.sin(theta),
             color="#e3f2fd", alpha=0.6, zorder=3)

    # Профиль концентрации внутри волокна
    r_plot = np.linspace(0, r_vis, 30)
    C_plot = np.interp(r_plot/r_vis, r_grid/R_fiber, C_final)
    for rr, cc in zip(r_plot, C_plot):
        ang = np.linspace(0, 2*np.pi, 60)
        x_ring = cx + rr*np.cos(ang)
        y_ring = cy + rr*np.sin(ang)
        col = plt.cm.RdYlGn(cc)
        ax3.plot(x_ring, y_ring, color=col, lw=1.5, alpha=0.7, zorder=4)

    ax3.text(cx, cy+0.1, f"C/Cs={C_axis:.3f}", ha="center", fontsize=8,
             fontweight="bold", color="#1a237e", zorder=5)
    ax3.text(cx, cy+r_vis+0.3, f"Волокно ПЭТ\nR = {R_fiber*1e6:.1f} мкм",
             ha="center", fontsize=10, fontweight="bold", color="#0d47a1")
    ax3.text(cx, cy-r_vis-0.4, "C(r,t) — ур. (3.7)", ha="center",
             fontsize=9, color="#555", style="italic")

    # Стрелка: ткань → волокно
    ax3.annotate("", xy=(cx - r_vis - 0.05, cy), xytext=(4.6, cy),
                 arrowprops=dict(arrowstyle="->", lw=2, color="#7b1fa2"))
    ax3.text(5.5, cy+0.3, "Диффузия\nкрасителя", ha="center", fontsize=9, color="#7b1fa2")

    # Легенда цветовой шкалы
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig3.colorbar(sm, ax=ax3, fraction=0.015, pad=0.01, location="right")
    cbar.set_label("C / Cs", fontsize=9)

    ax3.text(5.0, 0.4, "← x →  Ось теплопроводности (1D)", ha="center",
             fontsize=10, color="#888", style="italic")
    ax3.set_title(
        f"Физика процесса: режим {T_press}°C / {tau} с | "
        f"Материал: {'ПЭТ' if fiber_type == 'PET' else 'Шёлк'}",
        fontsize=12, fontweight="bold", pad=10
    )

    st.pyplot(fig3, use_container_width=True)

    st.info(
        "**Что видно на схеме:** слева — горячая плита даёт тепло, "
        "волна тепла идёт через бумагу в ткань (уравнение 3.1). "
        "В ткани находятся волокна (справа) — в них краситель диффундирует "
        "от поверхности к оси (уравнение 3.7). "
        "Цветовая шкала показывает концентрацию: красный = 0, зелёный = Cs."
    )

# ─────────────────────────────────────────────────────────────────
# НИЖНЯЯ ПАНЕЛЬ: ФИЗИКА В ЦИФРАХ
# ─────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📐 Физические зависимости")

col_d, col_tau = st.columns(2)

with col_d:
    st.markdown("**D(T) — коэффициент диффузии по Аррениусу (ур. 3.8–3.9)**")
    T_range = np.linspace(70, 230, 200)
    D_range = D_arrhenius(T_range)

    fig_d, ax_d = plt.subplots(figsize=(6, 3.5))
    ax_d.set_facecolor("#fafafa")
    fig_d.patch.set_facecolor("white")
    ax_d.semilogy(T_range, D_range, "b-", lw=2.5)
    ax_d.axvline(TG, color="#43a047", ls="--", lw=1.5, label=f"Tg = {TG}°C")
    ax_d.axvline(T_press, color="#e53935", ls="-.", lw=1.5, label=f"T_press = {T_press}°C")
    ax_d.plot(T_max, D_at_T, "r*", ms=12, label=f"D(T_макс) = {D_at_T:.1e}")
    ax_d.set_xlabel("Температура, °C"); ax_d.set_ylabel("D, м²/с")
    ax_d.set_title("Аррениус: экспоненциальная зависимость")
    ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3, which="both")
    ax_d.set_xlim(70, 230)
    st.pyplot(fig_d, use_container_width=True)
    st.caption(
        "Обратите внимание: D растёт на **7 порядков** при нагреве с 75 до 220°C. "
        "Это объясняет, почему 10 градусов разницы меняют результат кардинально."
    )

with col_tau:
    st.markdown("**τ_diff(T) — характерное время диффузии через волокно**")
    T_r2 = np.linspace(80, 230, 200)
    D_r2 = D_arrhenius(T_r2)
    td_r2 = np.where(D_r2 > 0, R_fiber**2 / (6*D_r2), np.nan)

    fig_t2, ax_t2 = plt.subplots(figsize=(6, 3.5))
    ax_t2.set_facecolor("#fafafa")
    fig_t2.patch.set_facecolor("white")
    ax_t2.semilogy(T_r2, td_r2, "r-", lw=2.5)
    ax_t2.axvline(T_press, color="#e53935", ls="-.", lw=1.5, label=f"T_press = {T_press}°C")
    ax_t2.axhline(tau, color="#1e88e5", ls="--", lw=1.5, label=f"τ_выдержки = {tau} с")
    if np.isfinite(tau_diff):
        ax_t2.plot(T_max, tau_diff, "r*", ms=12, label=f"τ_diff = {tau_diff:.0f} с")
    ax_t2.set_xlabel("Температура, °C"); ax_t2.set_ylabel("τ_diff, с")
    ax_t2.set_title("Как быстро краситель проникает в волокно?")
    ax_t2.legend(fontsize=8); ax_t2.grid(True, alpha=0.3, which="both")
    ax_t2.set_xlim(80, 230); ax_t2.set_ylim(0.01, 1e10)
    st.pyplot(fig_t2, use_container_width=True)
    st.caption(
        "Если τ_diff ≪ τ_выдержки (линия ниже синей) — волокно насытится полностью. "
        "Если τ_diff ≫ τ_выдержки — краситель практически не проникнет."
    )

# ─────────────────────────────────────────────────────────────────
# ПОДПИСЬ
# ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Модель: уравнение теплопроводности (3.1) + диффузия Фика (3.7) | "
    "МКР, явная схема, условия CFL (3.16–3.17) | "
    "ВКР 20.04.01 «Техносферная безопасность»"
)
