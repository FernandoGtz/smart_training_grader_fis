import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. CARGA DE CONFIGURACION Y DATOS
def cargar_json(ruta_archivo):
    with open(ruta_archivo, 'r') as file:
        return json.load(file)

# 2. PROCESAMIENTO DE DATOS MATEMATICOS
def calcular_1rm(peso, reps):
    if reps <= 1:
        return peso
    return peso * (1 + 0.033 * reps) # Formula de Epley (1985), adecuada para menos de 15 reps cercanas al fallo muscular

def procesar_datos(registros, config, rutina):
    datos_procesados = []

    # Crear diccionario de busqueda rapida del historico
    historico_dict = {reg['id_ejercicio']: reg for reg in registros['historico']}
    rutina_dict = {ej['id_ejercicio']: ej for ej in rutina}

    # Iterar sobre registros actuales
    for reg_actual in registros['actual']:
        id_ej = reg_actual['id_ejercicio']

        # Buscar en historico
        if id_ej not in historico_dict:
            # Manejar caso nulo
            continue

        reg_historico = historico_dict[id_ej]
        rutina_info = rutina_dict[id_ej]
        tier_ejercicio = rutina_info['tier']
        musculo_ejercicio = rutina_info['musculo']

        # Calcular 1RM para actual e historico
        rm_actual = calcular_1rm(reg_actual['peso'], reg_actual['reps'])
        rm_historico = calcular_1rm(reg_historico['peso'], reg_historico['reps'])

        # Calcular variacion porcentual
        delta_1rm = ((rm_actual - rm_historico) / rm_historico) * 100

        # Aplicar clipping con limites
        limite_min = config.get('limite_min', -15)
        limite_max = config.get('limite_max', 15)
        delta_1rm = np.clip(delta_1rm, limite_min, limite_max)

        # Construccion de diccionario con datos procesados
        datos_procesados.append({
            'id_ejercicio': id_ej,
            'nombre': rutina_info['nombre'],
            'musculo': musculo_ejercicio,
            'tier': tier_ejercicio,
            'delta_1rm': delta_1rm,
            'rpe': reg_actual['rpe'],
            'ct': reg_actual['ct'],
            'fa': reg_actual['fa'],
            'rm_actual': rm_actual,
            'rm_historico': rm_historico
        })
    return datos_procesados

# 3. CONSTRUCCION DEL SISTEMA DE INFERENCIA DIFUSO (FIS)
def construir_fis(config):
    # Definición de Rangos de las Variables (Universo Discurso)
    # -- FIS 1 --
    # Antecedentes
    delta_1rm = ctrl.Antecedent(np.arange(-15, 16, 1), 'delta_1rm')
    rpe = ctrl.Antecedent(np.arange(0, 11, 1), 'rpe')
    # Consecuente
    rm = ctrl.Consequent(np.arange(0, 101, 1), 'rm')

    # -- FIS 2 --
    # Antecedentes
    ct = ctrl.Antecedent(np.arange(0.0, 5.5, 0.5), 'ct')
    fa = ctrl.Antecedent(np.arange(0, 11, 1), 'fa')
    # Consecuente
    ce = ctrl.Consequent(np.arange(0, 101, 1), 'ce')

    # -- FIS 3 --
    # Antecedentes
    rm_fis3 = ctrl.Antecedent(np.arange(0, 101, 1), 'rm_fis3')
    ce_fis3 = ctrl.Antecedent(np.arange(0, 101, 1), 'ce_fis3')
    # Consecuente
    ics = ctrl.Consequent(np.arange(0, 101, 1), 'ics')

    # Asignación de Funciones de Pertenencia
    # FIS 1 - Delta 1RM
    delta_1rm['Retroceso'] = fuzz.trapmf(delta_1rm.universe, [-15, -15, -3, 0])
    delta_1rm['Mantenimiento'] = fuzz.trimf(delta_1rm.universe, [-3, 0, 3])
    delta_1rm['Progreso'] = fuzz.trapmf(delta_1rm.universe, [0, 3, 15, 15])

    # FIS 1 - RPE
    rpe['Suboptimo'] = fuzz.gaussmf(rpe.universe, 3, 1.5)
    rpe['Optimo'] = fuzz.gaussmf(rpe.universe, 8, 1.5)
    rpe['Limite'] = fuzz.gaussmf(rpe.universe, 10, 1.0)

    # FIS 1 - RM
    rm['Deficiente'] = fuzz.trapmf(rm.universe, [0, 0, 20, 40])
    rm['Aceptable'] = fuzz.trimf(rm.universe, [30, 50, 70])
    rm['Sobresaliente'] = fuzz.trapmf(rm.universe, [60, 80, 100, 100])

    # FIS 2 - CT (Capacidad Tecnica)
    ct['Comprometida'] = fuzz.trapmf(ct.universe, [0.5, 0.5, 1.5, 2.5])
    ct['Aceptable'] = fuzz.trimf(ct.universe, [2, 3.5, 4.5])
    ct['Impecable'] = fuzz.trapmf(ct.universe, [4, 4.75, 5.5, 5.5])

    # FIS 2 - FA (Fatiga)
    fa['Baja'] = fuzz.trapmf(fa.universe, [0, 0, 2, 5])
    fa['Manejable'] = fuzz.trimf(fa.universe, [3, 6, 9])
    fa['Critica'] = fuzz.trapmf(fa.universe, [7, 9, 11, 11])

    # FIS 2 - CE (Calidad de Ejecucion)
    ce['Deficiente'] = fuzz.trapmf(ce.universe, [0, 0, 20, 40])
    ce['Estandar'] = fuzz.trimf(ce.universe, [30, 50, 70])
    ce['Optima'] = fuzz.trapmf(ce.universe, [60, 80, 100, 100])

    # FIS 3 - RM (Rendimiento Muscular)
    rm_fis3['Deficiente'] = fuzz.trapmf(rm_fis3.universe, [0, 0, 20, 40])
    rm_fis3['Aceptable'] = fuzz.trimf(rm_fis3.universe, [30, 50, 70])
    rm_fis3['Sobresaliente'] = fuzz.trapmf(rm_fis3.universe, [60, 80, 100, 100])

    # FIS 3 - CE (Calidad de Ejecucion)
    ce_fis3['Deficiente'] = fuzz.trapmf(ce_fis3.universe, [0, 0, 20, 40])
    ce_fis3['Estandar'] = fuzz.trimf(ce_fis3.universe, [30, 50, 70])
    ce_fis3['Optima'] = fuzz.trapmf(ce_fis3.universe, [60, 80, 100, 100])

    # FIS 3 - ICS (Indice de Calidad de Sesion)
    ics['Pobre'] = fuzz.trapmf(ics.universe, [0, 0, 20, 40])
    ics['Productivo'] = fuzz.trimf(ics.universe, [30, 50, 70])
    ics['AltoRendimiento'] = fuzz.trapmf(ics.universe, [60, 80, 100, 100])

    # Generacion de Reglas
    reglas_rm_obj = []
    for regla in config['matriz_reglas_rm']:
        delta_term = delta_1rm[regla['delta_1rm']]
        rpe_term = rpe[regla['rpe']]
        rm_term = rm[regla['rm']]
        regla_obj = ctrl.Rule(delta_term & rpe_term, rm_term)
        reglas_rm_obj.append(regla_obj)

    reglas_ce_obj = []
    for regla in config['matriz_reglas_ce']:
        ct_term = ct[regla['ct']]
        fa_term = fa[regla['fa']]
        ce_term = ce[regla['ce']]
        regla_obj = ctrl.Rule(ct_term & fa_term, ce_term)
        reglas_ce_obj.append(regla_obj)

    reglas_final_obj = []
    for regla in config['matriz_reglas_final']:
        rm_term = rm_fis3[regla['rm']]
        ce_term = ce_fis3[regla['ce']]
        ics_term = ics[regla['ics']]
        regla_obj = ctrl.Rule(rm_term & ce_term, ics_term)
        reglas_final_obj.append(regla_obj)

    # Instanciacion de Controladores
    rm_ctrl = ctrl.ControlSystem(reglas_rm_obj)
    rm_sim = ctrl.ControlSystemSimulation(rm_ctrl)

    ce_ctrl = ctrl.ControlSystem(reglas_ce_obj)
    ce_sim = ctrl.ControlSystemSimulation(ce_ctrl)

    ics_ctrl = ctrl.ControlSystem(reglas_final_obj)
    ics_sim = ctrl.ControlSystemSimulation(ics_ctrl)

    return {
        'rm_sim': rm_sim,
        'ce_sim': ce_sim,
        'ics_sim': ics_sim
    }


# 4. EJECUCION PRINCIPAL
if __name__ == "__main__":
    # Cargar JSONs
    config = cargar_json('config_perfil_hipertrofia.json')
    registros = cargar_json('registros_sesion.json')
    rutina = cargar_json('rutina_Ex_fullbody.json')

    # Procesar datos
    datos_procesados = procesar_datos(registros, config, rutina)

    # Construir FIS
    simuladores = construir_fis(config)
    rm_sim = simuladores['rm_sim']
    ce_sim = simuladores['ce_sim']
    ics_sim = simuladores['ics_sim']

    # Resultados finales
    resultados = []
    resultados_por_musculo = {}

    # Bucle para pasar los datos procesados a los simuladores
    for dato in datos_procesados:
        # FIS 1: Calcular RM (Rendimiento Muscular)
        rm_sim.input['delta_1rm'] = np.clip(dato['delta_1rm'], -15, 15)
        rm_sim.input['rpe'] = np.clip(dato['rpe'], 0, 10)
        rm_sim.compute()
        rm_output = rm_sim.output['rm']

        # FIS 2: Calcular CE (Calidad de Ejecucion)
        ce_sim.input['ct'] = np.clip(dato['ct'], 0.5, 5.5)
        ce_sim.input['fa'] = np.clip(dato['fa'], 0, 10)
        ce_sim.compute()
        ce_output = ce_sim.output['ce']

        # FIS 3: Calcular ICS (Indice de Calidad de Sesion)
        ics_sim.input['rm_fis3'] = np.clip(rm_output, 0, 100)
        ics_sim.input['ce_fis3'] = np.clip(ce_output, 0, 100)
        ics_sim.compute()
        ics_output = ics_sim.output['ics']

        # Obtener ponderacion segun Tier
        tier = dato['tier']
        ponderacion = float(config['ponderaciones_tier'][str(tier)])

        resultado_ejercicio = {
            'id_ejercicio': dato['id_ejercicio'],
            'nombre': dato['nombre'],
            'musculo': dato['musculo'],
            'tier': tier,
            'delta_1rm': round(dato['delta_1rm'], 2),
            'rpe': dato['rpe'],
            'ct': dato['ct'],
            'fa': dato['fa'],
            'rm_score': round(rm_output, 2),
            'ce_score': round(ce_output, 2),
            'ics_score': round(ics_output, 2),
            'ponderacion': ponderacion,
            'ics_ponderado': round(ics_output * ponderacion, 2)
        }
        resultados.append(resultado_ejercicio)

        # Agregar a diccionario por grupo muscular
        musculo = dato['musculo']
        if musculo not in resultados_por_musculo:
            resultados_por_musculo[musculo] = {
                'ejercicios': [],
                'ics_suma_ponderada': 0,
                'ponderacion_total': 0
            }

        resultados_por_musculo[musculo]['ejercicios'].append(resultado_ejercicio)
        resultados_por_musculo[musculo]['ics_suma_ponderada'] += ics_output * ponderacion
        resultados_por_musculo[musculo]['ponderacion_total'] += ponderacion

    # Calcular ICS promedio ponderado por grupo muscular
    ics_por_musculo = {}
    for musculo, datos in resultados_por_musculo.items():
        if datos['ponderacion_total'] > 0:
            ics_por_musculo[musculo] = round(datos['ics_suma_ponderada'] / datos['ponderacion_total'], 2)

    # Calcular calificacion final (sumatoria de productos ICS*ponderacion / sumatoria de ponderaciones)
    ics_suma_total = sum([r['ics_ponderado'] for r in resultados])
    ponderacion_suma_total = sum([r['ponderacion'] for r in resultados])
    calificacion_final = round(ics_suma_total / ponderacion_suma_total, 2) if ponderacion_suma_total > 0 else 0

    # Mostrar resultados detallados
    print("\n" + "="*80)
    print("EVALUACIÓN DE SESIÓN DE ENTRENAMIENTO - FULL BODY")
    print("="*80)

    for resultado in resultados:
        print(f"\n[{resultado['musculo'].upper()}] {resultado['nombre']} ({resultado['id_ejercicio']})")
        print(f"  Tier: {resultado['tier']} | Ponderación: {resultado['ponderacion']}")
        print(f"  Delta 1RM: {resultado['delta_1rm']}% | RPE: {resultado['rpe']}")
        print(f"  CT: {resultado['ct']} | FA: {resultado['fa']}")
        print(f"  Score RM: {resultado['rm_score']} | Score CE: {resultado['ce_score']}")
        print(f"  Score ICS: {resultado['ics_score']} | ICS Ponderado: {resultado['ics_ponderado']}")

    # Mostrar resultados por grupo muscular
    print("\n" + "="*80)
    print("CALIFICACIONES POR GRUPO MUSCULAR")
    print("="*80)

    for musculo in sorted(ics_por_musculo.keys()):
        ics_musculo = ics_por_musculo[musculo]
        print(f"\n{musculo.upper()}: {ics_musculo}")
        # Mostrar ejercicios del grupo muscular
        for ej in resultados_por_musculo[musculo]['ejercicios']:
            print(f"  - {ej['nombre']}: ICS {ej['ics_score']} (Ponderado: {ej['ics_ponderado']})")

    # Mostrar calificacion final
    print("\n" + "="*80)
    print("CALIFICACIÓN FINAL DE LA SESIÓN")
    print("="*80)
    print(f"\nICS Final Ponderado: {calificacion_final}")
    print(f"(Calculado como: Σ(ICS × Ponderación) / Σ(Ponderaciones))")
    print(f"Numerador: {ics_suma_total} | Denominador: {ponderacion_suma_total}")
    print("="*80 + "\n")

