import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Параметры системы
# Условие 1: Заданы все параметры системы
m1 = 1.0    # Масса первого стержня (кг)
m2 = 0.6    # Масса второго стержня (кг)
l1 = 1.0    # Длина первого стержня (м)
l2 = 1.0    # Длина второго стержня (м)
a = 0.4     # Расстояние до точек крепления пружины (м)
c = 200.0   # Жёсткость пружины (Н/м)
phi0 = np.pi / 4  # Угол, при котором пружина не растянута (рад)
g = 9.81    # Ускорение свободного падения (м/с²)

# Условие 2: Заданы начальные условия
phi1_0 = 0          # Начальный угол φ1 = 0 рад
phi2_0 = 0.7        # Начальный угол φ2 = 0.7 рад
dphi1_dt_0 = 10     # Начальная угловая скорость φ1 = 10 с⁻¹
dphi2_dt_0 = 0      # Начальная угловая скорость φ2 = 0 с⁻¹

# Время моделирования
t = np.linspace(0, 10, 1000)

# Начальное состояние
initial_state = [phi1_0, phi2_0, dphi1_dt_0, dphi2_dt_0]

# Уравнения движения системы
def system_eqs(state, t):
    phi1, phi2, dphi1_dt, dphi2_dt = state
    I1 = m1 * l1**2 / 3  # Моменты инерции стержней
    I2 = m2 * l2**2 / 3
    
    # Условие 10: Уравнения Лагранжа II рода:
    # d/dt(∂T/∂φ'₁) - ∂T/∂φ₁ = Q₁
    # d/dt(∂T/∂φ'₂) - ∂T/∂φ₂ = Q₂
    spring_term = (a**2) * c * (np.sin(phi2 - phi1) - 2 * np.sin(phi0/2) * np.cos((phi2 - phi1)/2))
    gravity_term1 = (m1 * g * l1 / 2) * np.cos(phi1)
    gravity_term2 = (m2 * g * l2 / 2) * np.cos(phi2)
    d2phi1_dt2 = (spring_term - gravity_term1) / I1
    d2phi2_dt2 = (-spring_term - gravity_term2) / I2
    return [dphi1_dt, dphi2_dt, d2phi1_dt2, d2phi2_dt2]

# Решение системы
solution = odeint(system_eqs, initial_state, t)
phi1_sol = solution[:, 0]
phi2_sol = solution[:, 1]
dphi1_dt_sol = solution[:, 2]
dphi2_dt_sol = solution[:, 3]

# Функция для расчёта реакций
def calculate_reactions(t, solution):
    phi1 = solution[:, 0]
    phi2 = solution[:, 1]
    dphi1_dt = solution[:, 2]
    dphi2_dt = solution[:, 3]
    d2phi1_dt2 = np.gradient(dphi1_dt, t)
    d2phi2_dt2 = np.gradient(dphi2_dt, t)
    Rx = -0.5 * (m1 * l1 * (d2phi1_dt2 * np.sin(phi1) + dphi1_dt**2 * np.cos(phi1)) +
                 m2 * l2 * (d2phi2_dt2 * np.sin(phi2) + dphi2_dt**2 * np.cos(phi2)))
    Ry = 0.5 * (m1 * l1 * (d2phi1_dt2 * np.cos(phi1) - dphi1_dt**2 * np.sin(phi1)) +
                m2 * l2 * (d2phi2_dt2 * np.cos(phi2) - dphi2_dt**2 * np.sin(phi2))) + (m1 + m2) * g
    return Rx, Ry

# Вычисление реакций
Rx, Ry = calculate_reactions(t, solution)

# Графики углов и реакций
fig_plots, (ax_angles, ax_reactions) = plt.subplots(2, 1, figsize=(10, 8))

# Условие 3: Построить графики зависимости углов от времени
ax_angles.plot(t, phi1_sol, label='φ1')
ax_angles.plot(t, phi2_sol, label='φ2')
ax_angles.set_xlabel('Время (с)')
ax_angles.set_ylabel('Угол (рад)')
ax_angles.set_title('Угловые положения')
ax_angles.grid(True)
ax_angles.legend()

# Условие 4: Построить графики реакций в точке O
ax_reactions.plot(t, Rx, label='Rx')
ax_reactions.plot(t, Ry, label='Ry')
ax_reactions.set_xlabel('Время (с)')
ax_reactions.set_ylabel('Сила реакции (Н)')
ax_reactions.set_title('Силы реакции')
ax_reactions.grid(True)
ax_reactions.legend()

plt.tight_layout()

# Условие 9: Выражения для кинетической и потенциальной энергии системы
def calculate_energy(phi1, phi2, dphi1_dt, dphi2_dt):
    # Кинетическая энергия
    T1 = (m1 * l1**2 / 6) * dphi1_dt**2  # Кинетическая энергия первого стержня
    T2 = (m2 * l2**2 / 6) * dphi2_dt**2  # Кинетическая энергия второго стержня
    T = T1 + T2
    
    # Потенциальная энергия
    V_gravity1 = -(m1 * g * l1 / 2) * np.sin(phi1)  # Потенциальная энергия тяжести первого стержня
    V_gravity2 = -(m2 * g * l2 / 2) * np.sin(phi2)  # Потенциальная энергия тяжести второго стержня
    V_spring = (a**2 * c / 2) * (2 - 2*np.cos((phi2 - phi1)/2 - phi0/2))  # Потенциальная энергия пружины
    V = V_gravity1 + V_gravity2 + V_spring
    
    # Обобщенные силы (производные от потенциальной энергии по обобщенным координатам)
    Q1 = -m1*g*l1/2*np.cos(phi1) + a**2*c*(np.sin(phi2-phi1) - 2*np.sin(phi0/2)*np.cos((phi2-phi1)/2))
    Q2 = -m2*g*l2/2*np.cos(phi2) - a**2*c*(np.sin(phi2-phi1) - 2*np.sin(phi0/2)*np.cos((phi2-phi1)/2))
    
    return T, V, Q1, Q2

# Условие 10: Уравнения Лагранжа II рода и интеграл энергии
# Уравнения движения системы 
def system_eqs(state, t):
    phi1, phi2, dphi1_dt, dphi2_dt = state
    I1 = m1 * l1**2 / 3  #
    I2 = m2 * l2**2 / 3
    
    # Уравнения Лагранжа II рода:
    # d/dt(∂T/∂φ'₁) - ∂T/∂φ₁ = Q₁
    # d/dt(∂T/∂φ'₂) - ∂T/∂φ₂ = Q₂
    spring_term = (a**2) * c * (np.sin(phi2 - phi1) - 2 * np.sin(phi0/2) * np.cos((phi2 - phi1)/2))
    gravity_term1 = (m1 * g * l1 / 2) * np.cos(phi1)
    gravity_term2 = (m2 * g * l2 / 2) * np.cos(phi2)
    d2phi1_dt2 = (spring_term - gravity_term1) / I1
    d2phi2_dt2 = (-spring_term - gravity_term2) / I2
    return [dphi1_dt, dphi2_dt, d2phi1_dt2, d2phi2_dt2]

# Расчет энергий и построение графика интеграла энергии
T, V, Q1, Q2 = calculate_energy(solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3])
E = T + V  # Интеграл энергии системы

# График интеграла энергии
fig_energy = plt.figure(figsize=(10, 6))
ax_energy = fig_energy.add_subplot(111)
ax_energy.plot(t, T, label='Кинетическая энергия', color='blue')
ax_energy.plot(t, V, label='Потенциальная энергия', color='red')
ax_energy.plot(t, E, label='Полная энергия (интеграл)', color='green')
ax_energy.set_xlabel('Время (с)')
ax_energy.set_ylabel('Энергия (Дж)')
ax_energy.set_title('Интеграл энергии системы')
ax_energy.grid(True)
ax_energy.legend()

# Анимация
fig_anim = plt.figure(figsize=(8, 8))
ax_anim = fig_anim.add_subplot(111)
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(-2, 2)
ax_anim.grid(True)
ax_anim.set_aspect('equal')
ax_anim.set_title('Анимация системы')
line_OA, = ax_anim.plot([], [], 'b-', lw=2, label='Стержень OA')
line_OB, = ax_anim.plot([], [], 'r-', lw=2, label='Стержень OB')
spring, = ax_anim.plot([], [], 'g--', lw=1, label='Пружина')
point_O = Circle((0, 0), 0.05, fc='k')
ax_anim.add_patch(point_O)
ax_anim.legend()

def update(frame):
    phi1 = phi1_sol[frame]
    phi2 = phi2_sol[frame]
    xA, yA = l1 * np.cos(phi1), l1 * np.sin(phi1)
    xB, yB = l2 * np.cos(phi2), l2 * np.sin(phi2)
    xD, yD = a * np.cos(phi1), a * np.sin(phi1)
    xE, yE = a * np.cos(phi2), a * np.sin(phi2)
    line_OA.set_data([0, xA], [0, yA])
    line_OB.set_data([0, xB], [0, yB])
    spring.set_data([xD, xE], [yD, yE])
    return line_OA, line_OB, spring

anim = FuncAnimation(fig_anim, update, frames=len(t), interval=20, blit=True)

# 5. Поиск наименьшей угловой скорости
def find_min_velocity():
 
    I1 = m1 * l1**2 / 3
    
    # Изменение потенциальной энергии тяжести
    delta_V_gravity = (m1 * g * l1 / 2) * (1 - np.sin(np.pi/2))
    
    # Изменение потенциальной энергии пружины
    delta_V_spring = (a**2 * c / 2) * (2 - 2*np.cos(-np.pi/2 - phi0/2))
    
    # Минимальная необходимая начальная кинетическая энергия
    T_min = delta_V_gravity + delta_V_spring
    
    # Находим минимальную угловую скорость
    omega_min = np.sqrt(2 * T_min / I1)
    
    print("\n5. Минимальная угловая скорость:")
    print(f"ω_min = {omega_min:.3f} рад/с")
    return omega_min

# 6. Главные векторы и моменты сил инерции
def calculate_inertia_forces(t_point):
    idx = np.abs(t - t_point).argmin()
    
    phi1 = solution[idx, 0]
    phi2 = solution[idx, 1]
    dphi1_dt = solution[idx, 2]
    dphi2_dt = solution[idx, 3]
    
    # Вычисляем ускорения через уравнения движения
    state = [phi1, phi2, dphi1_dt, dphi2_dt]
    _, _, d2phi1_dt2, d2phi2_dt2 = system_eqs(state, t_point)
    
    # Главные векторы сил инерции
    F_in1_x = -m1 * l1/2 * (d2phi1_dt2 * np.sin(phi1) + dphi1_dt**2 * np.cos(phi1))
    F_in1_y = -m1 * l1/2 * (d2phi1_dt2 * np.cos(phi1) - dphi1_dt**2 * np.sin(phi1))
    F_in2_x = -m2 * l2/2 * (d2phi2_dt2 * np.sin(phi2) + dphi2_dt**2 * np.cos(phi2))
    F_in2_y = -m2 * l2/2 * (d2phi2_dt2 * np.cos(phi2) - dphi2_dt**2 * np.sin(phi2))
    
    # Главные моменты сил инерции
    M_in1 = -(m1 * l1**2 / 3) * d2phi1_dt2
    M_in2 = -(m2 * l2**2 / 3) * d2phi2_dt2
    
    print(f"\n6. Силы инерции при t = {t_point:.2f} с:")
    print(f"Стержень OA: F = ({F_in1_x:.2f}, {F_in1_y:.2f}) Н, M = {M_in1:.2f} Н·м")
    print(f"Стержень OB: F = ({F_in2_x:.2f}, {F_in2_y:.2f}) Н, M = {M_in2:.2f} Н·м")
    
    return (F_in1_x, F_in1_y), (F_in2_x, F_in2_y), M_in1, M_in2

# Условие 7: Проверка результатов с помощью принципа Даламбера
def verify_dalembert(t_point, forces_and_moments):
    idx = np.abs(t - t_point).argmin()
    (F_in1_x, F_in1_y), (F_in2_x, F_in2_y), M_in1, M_in2 = forces_and_moments
    
    # Проверка равновесия всех сил и моментов
    sum_F_x = F_in1_x + F_in2_x + Rx[idx]
    sum_F_y = F_in1_y + F_in2_y + Ry[idx] - (m1 + m2) * g
    
    print(f"\n7. Проверка принципа Даламбера при t = {t_point:.2f} с:")
    print(f"Сумма сил по X: {sum_F_x:.2e} Н")
    print(f"Сумма сил по Y: {sum_F_y:.2e} Н")

# Условие 11: Исследование положений равновесия и устойчивости
def analyze_equilibrium():
    print("\n11. Анализ положений равновесия:")
    
    # Для малых колебаний около положения равновесия:
    I2 = m2 * l2**2 / 3
    k_eff = a**2 * c
    omega = np.sqrt(k_eff / I2)  # Частота малых колебаний
    T = 2 * np.pi / omega        # Период малых колебаний
    
    print(f"Частота малых колебаний: {omega:.3f} рад/с")
    print(f"Период малых колебаний: {T:.3f} с")
    
    return omega, T

# Выполнение всех пунктов
omega_min = find_min_velocity()
forces_and_moments = calculate_inertia_forces(1.0)  # Вызываем только один раз для t = 1.0 с
verify_dalembert(1.0, forces_and_moments)  # Передаем уже вычисленные силы
omega, T = analyze_equilibrium()

# 8. Дифференциальное уравнение движения стержня OA
print("\n8. Дифференциальное уравнение движения стержня OA:")
print("I₁φ̈₁ = a²c[sin(φ₂-φ₁) - 2sin(φ₀/2)cos((φ₂-φ₁)/2)] - (m₁gl₁/2)cos(φ₁)")
print("где:")
print(f"I₁ = {m1*l1**2/3:.3f} кг·м²")
print(f"a²c = {a**2*c:.3f} Н·м")
print(f"m₁gl₁/2 = {m1*g*l1/2:.3f} Н·м")

plt.show()