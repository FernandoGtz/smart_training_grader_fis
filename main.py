import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. CARGA DE CONFIGURACIÓN Y DATOS
def cargar_json(ruta_archivo):
    with open(ruta_archivo, 'r') as file:
        return json.load(file)

# 2. PROCESAMIENTO DE DATOS MATEMÁTICOS
def calcular_brzycki(peso, reps):
    if reps <= 1:
        return peso
    # Falta: Implementar la fórmula matemática de Brzycki
    pass


def procesar_datos(registros, config):
    datos_procesados = []
    limite_min = config["limites_universo"]["delta_1rm_min"]
    limite_max = config["limites_universo"]["delta_1rm_max"]

    # Falta: Bucle para iterar sobre registros['actual']
    # Falta: Buscar el registro correspondiente en registros['historico']
    # Falta: Manejar el caso nulo (ejercicio actual no está en el histórico)
    # Falta: Calcular 1RM para ambos y obtener la variación porcentual
    # Falta: Aplicar clipping con limite_min y limite_max

    return datos_procesados

# 3. CONSTRUCCIÓN DEL SISTEMA DE INFERENCIA DIFUSA (FIS)
def construir_fis(config):
    # A. Definición de Universos (Variables)
    # Antecedentes FIS 1
    delta_1rm = ctrl.Antecedent(np.arange(-15, 16, 1), 'delta_1rm')
    rpe = ctrl.Antecedent(np.arange(1, 11, 1), 'rpe')
    # Consecuente FIS 1
    rm = ctrl.Consequent(np.arange(0, 101, 1), 'rm')

    # Falta: Instanciar antecedentes y consecuentes para FIS 2 (ct, fa -> ce)
    # Falta: Instanciar consecuente para FIS 3 (ics)

    # B. Asignación de Funciones de Pertenencia (Sintaxis de ejemplo)
    # trapmf requiere [inicio, pico1, pico2, fin]
    delta_1rm['Retroceso'] = fuzz.trapmf(delta_1rm.universe, [-15, -15, -3, 0])
    # trimf requiere [inicio, pico, fin]
    # Falta: Asignar 'Mantenimiento' (trimf) y 'Progreso' (trapmf)

    # gaussmf requiere [centro, sigma]
    rpe['Suboptimo'] = fuzz.gaussmf(rpe.universe, 3, 1.5)
    # Falta: Asignar 'Optimo' y 'Al Limite' para RPE
    # Falta: Asignar funciones para CT, FA, RM, CE, ICS

    # C. Generación Dinámica de Reglas
    reglas_rm_obj = []
    for regla in config['matriz_reglas_rm']:
        # Sintaxis para crear regla: ctrl.Rule(antecedente1 & antecedente2, consecuente)
        # Falta: Lógica para leer el string del JSON y conectarlo con las variables difusas
        pass

    # Falta: Generar reglas_ce_obj y reglas_final_obj

    # D. Instanciación de Controladores
    # rm_ctrl = ctrl.ControlSystem(reglas_rm_obj)
    # rm_sim = ctrl.ControlSystemSimulation(rm_ctrl)

    # Falta: Instanciar simuladores para CE e ICS

    return  # Retornar los simuladores para usarlos en el cálculo final


# ---------------------------------------------------------
# 4. EJECUCIÓN PRINCIPAL
# ---------------------------------------------------------
if __name__ == "__main__":
    # Cargar JSONs
    # Llamar procesar_datos
    # Llamar construir_fis
    # Bucle para pasar los datos procesados a los simuladores (.input['variable'] = valor)
    # Computar (.compute()) y extraer la salida
    # Aplicar agregación ponderada según Tier
    pass