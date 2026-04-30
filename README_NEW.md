# Smart Training Grader FIS

## Descripción General

Sistema inteligente de evaluación de sesiones de entrenamiento mediante **Sistemas de Inferencia Difusa (FIS - Fuzzy Inference Systems)**. El sistema procesa datos de ejercicios de entrenamiento y genera calificaciones automáticas basadas en lógica difusa.

## Arquitectura del Sistema

El sistema está compuesto por **3 Sistemas de Inferencia Difusa (FIS)** que trabajan en cascada:

### FIS 1: Evaluación de Rendimiento Muscular (RM)
**Entrada:**
- `delta_1rm`: Variación porcentual del 1RM respecto a la sesión anterior (-15% a +15%)
- `rpe`: Esfuerzo percibido (0-10)

**Salida:**
- `rm`: Score de Rendimiento Muscular (0-100)

**Reglas:** 9 reglas difusas basadas en matriz de configuración

### FIS 2: Evaluación de Calidad de Ejecución (CE)
**Entrada:**
- `ct`: Capacidad Técnica (0.5-5.5)
- `fa`: Fatiga percibida (0-10)

**Salida:**
- `ce`: Score de Calidad de Ejecución (0-100)

**Reglas:** 9 reglas difusas basadas en matriz de configuración

### FIS 3: Índice de Calidad de Sesión (ICS)
**Entrada:**
- `rm_fis3`: Score de RM del FIS 1 (0-100)
- `ce_fis3`: Score de CE del FIS 2 (0-100)

**Salida:**
- `ics`: Índice de Calidad de Sesión (0-100)

**Reglas:** 9 reglas difusas basadas en matriz de configuración

## Flujo de Datos

```
registros_sesion.json
        ↓
  Procesar Datos (Brzycki)
        ↓
  FIS 1: delta_1rm + rpe → rm
        ↓
  FIS 2: ct + fa → ce
        ↓
  FIS 3: rm + ce → ics
        ↓
resultados_sesion.json
```

## Componentes Principales

### 1. `calcular_brzycki(peso, reps)`
Fórmula: `1RM = peso × (36 / (37 - reps))`

### 2. `procesar_datos(registros, config)`
- Calcula 1RM histórico y actual
- Computa variación porcentual (delta_1rm)
- Aplica clipping a límites

### 3. `construir_fis(config)`
- Define 3 FIS con universos, funciones de pertenencia y reglas
- Retorna simuladores lista para uso

### 4. Bucle Principal
Procesa cada ejercicio a través de los 3 FIS en cascada

## Ejecución

```bash
python main.py
```

## Archivos

- **config_perfil_hipertrofia.json**: Matrices de reglas y ponderaciones
- **registros_sesion.json**: Datos de entrada
- **resultados_sesion.json**: Resultados de salida
- **main.py**: Código principal completo

## Resultados

Genera JSON con:
- Scores individuales por ejercicio (rm_score, ce_score, ics_score)
- ICS Promedio Ponderado de la sesión
- Delta 1RM y parámetros de entrada

---
**Versión:** 1.0 - Completado
**Los 3 FIS están totalmente operacionales**

