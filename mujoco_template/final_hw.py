import numpy as np
import pinocchio as pin
from simulator import Simulator
from pathlib import Path
import os, csv

# ==============================
# Пути и модель
# ==============================
current_dir = os.path.dirname(os.path.abspath(__file__))
UR5_XML_PATH = "/Users/daeron/forc/hw/mujoco_template/robots/universal_robots_ur5e/ur5e.xml"

pin_model = pin.buildModelFromMJCF(UR5_XML_PATH)
pin_data = pin_model.createData()

# ==============================
# Параметры
# ==============================
SIM_TIME = 10.0
FPS = 30

KP_ID, KD_ID = 100, 20
LAMBDA_SMC = 10
K_SMC = 20
PHI_LIST = [0.2, 0.5, 1.0]

Q_DES = np.array([0, -np.pi/4, np.pi/2, -np.pi/4, np.pi/2, 0])
DQ_DES = np.zeros(6)
DDQ_DES = np.zeros(6)

Path("logs/videos").mkdir(parents=True, exist_ok=True)

# ==============================
# Контроллеры
# ==============================
def inverse_dynamics_controller(q, dq):
    e  = Q_DES - q
    de = DQ_DES - dq
    pin.computeAllTerms(pin_model, pin_data, q, dq)
    return pin_data.M @ (DDQ_DES + KP_ID*e + KD_ID*de) + pin_data.nle

def sat(s, phi):
    return np.clip(s/phi, -1, 1)

def sliding_mode_controller(q, dq, phi):
    e  = Q_DES - q
    de = DQ_DES - dq
    s  = de + LAMBDA_SMC*e
    pin.computeAllTerms(pin_model, pin_data, q, dq)
    tau = pin_data.M @ (DDQ_DES + LAMBDA_SMC*de) + pin_data.nle + K_SMC * sat(s, phi)
    return tau, s

# ==============================
# Симуляция с записью видео
# ==============================
def run_sim(controller_type, phi=None):
    # создаём новый инстанс симуляции
    sim = Simulator(
        xml_path=UR5_XML_PATH,
        enable_task_space=False,
        show_viewer=False,
        record_video=True,
        fps=FPS,
        width=640,
        height=480
    )

    # уникальное имя видео
    if controller_type == 'ID':
        sim.video_path = "logs/videos/ID.mp4"
    else:
        sim.video_path = f"logs/videos/SMC_phi_{phi}.mp4"

    # динамика
    sim.set_joint_damping(np.array([0.5,0.5,0.5,0.1,0.1,0.1]))
    sim.set_joint_friction(np.array([1.5,0.5,0.5,0.1,0.1,0.1]))
    sim.modify_body_properties("end_effector", mass=4)

    # логирование
    log = {"time":[], "q":[], "q_des":[], "s":[]}

    def control(q, dq, t):
        if controller_type == "ID":
            tau = inverse_dynamics_controller(q, dq)
            s_val = np.zeros(6)
        else:
            tau, s_val = sliding_mode_controller(q, dq, phi)

        log["time"].append(t)
        log["q"].append(q.copy())
        log["q_des"].append(Q_DES.copy())
        log["s"].append(s_val.copy())
        return tau

    sim.set_controller(control)
    sim.run(SIM_TIME)

    for k in log:
        log[k] = np.array(log[k])
    return log

# ==============================
# Анализ и сохранение
# ==============================
def analyze_and_save():
    results = []

    # ID контроллер
    logs_id = run_sim("ID")
    rmse_id = np.mean(np.linalg.norm(logs_id["q"] - logs_id["q_des"], axis=1))
    results.append({"Controller":"ID","phi":"N/A","RMSE":rmse_id,"Chattering":0})

    # SMC с разным φ
    for phi in PHI_LIST:
        logs = run_sim("SMC", phi)
        rmse = np.mean(np.linalg.norm(logs["q"] - logs["q_des"], axis=1))
        chattering = np.mean(np.linalg.norm(logs["s"], axis=1))
        results.append({"Controller":"SMC","phi":phi,"RMSE":rmse,"Chattering":chattering})

    with open("tradeoff_results.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Controller","phi","RMSE","Chattering"])
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Completed ===")
    for r in results:
        print(r)

if __name__ == "__main__":
    analyze_and_save()