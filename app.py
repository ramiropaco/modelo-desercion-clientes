import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#streamlit run app.py
# Configurar la página de streamlit
st.set_page_config(page_title="Predictor de Deserción de Clientes", layout="wide")
st.title("Predicción de Deserción de Clientes")

# Crear un modelo simple que se entrenará sobre la marcha
def create_model():
    model = Sequential([
        Dense(16, activation='relu', input_shape=(11,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Generador de datos sintéticos de ejemplo (en lugar de leer desde CSV)
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)

    # Generar características sintéticas
    credit_scores = np.random.randint(300, 850, n_samples)
    genders = np.random.randint(0, 2, n_samples)  # 0: Hombre, 1: Mujer
    ages = np.random.randint(18, 95, n_samples)
    tenures = np.random.randint(0, 11, n_samples)
    balances = np.random.uniform(0, 250000, n_samples)
    products = np.random.randint(1, 5, n_samples)
    credit_cards = np.random.randint(0, 2, n_samples)
    active_members = np.random.randint(0, 2, n_samples)
    salaries = np.random.uniform(10000, 200000, n_samples)

    # Geografía codificada con un solo valor (3 países)
    geo_france = np.random.randint(0, 2, n_samples)
    geo_germany = np.random.randint(0, 2, n_samples)
    geo_spain = np.random.randint(0, 2, n_samples)

    # Crear matriz de características
    X = np.column_stack([
        credit_scores, genders, ages, tenures, balances,
        products, credit_cards, active_members, salaries,
        geo_france, geo_germany
    ])

    # Generar objetivo sintético (deserción)
    # Mayor probabilidad de deserción para:
    # - Mayor edad
    # - Menor antigüedad
    # - Menor saldo de cuenta
    # - Miembros no activos
    desercion_prob = (
        0.1 +
        0.3 * (ages / 95) +
        0.2 * (1 - tenures / 10) +
        0.2 * (1 - balances / 250000) +
        0.2 * (1 - active_members)
    )

    y = (np.random.random(n_samples) < desercion_prob).astype(int)

    return X, y

def normalize_input(input_data):
    """Normalizar los datos de entrada a un rango similar a los datos de entrenamiento"""
    # Escalado min-max simple basado en rangos esperados
    normalized = np.zeros(11)

    normalized[0] = (input_data[0] - 300) / (850 - 300)  # Puntaje de Crédito
    normalized[1] = input_data[1]  # Género (ya es 0 o 1)
    normalized[2] = (input_data[2] - 18) / (95 - 18)  # Edad
    normalized[3] = input_data[3] / 10  # Antigüedad
    normalized[4] = input_data[4] / 250000  # Saldo
    normalized[5] = (input_data[5] - 1) / 3  # Número de Productos
    normalized[6] = input_data[6]  # Tiene Tarjeta de Crédito
    normalized[7] = input_data[7]  # Es Miembro Activo
    normalized[8] = input_data[8] / 200000  # Salario
    normalized[9] = input_data[9]  # Geografía Francia
    normalized[10] = input_data[10]  # Geografía Alemania

    return normalized.reshape(1, -1)  # Redimensionar para la entrada del modelo

# Crear barra lateral para el entrenamiento del modelo (si el usuario quiere reentrenar)
with st.sidebar:
    st.header("Entrenamiento del Modelo")
    st.write("Entrena el modelo con datos sintéticos")

    if st.button("Entrenar Modelo"):
        with st.spinner("Entrenando el modelo con datos sintéticos..."):
            # Generar datos sintéticos
            X_train, y_train = generate_synthetic_data(n_samples=5000)

            # Crear y entrenar el modelo
            model = create_model()
            model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                verbose=0
            )

            # Guardar el modelo en el estado de la sesión
            st.session_state.model = model
            st.success("¡Modelo entrenado exitosamente!")

    # Añadir información sobre la aplicación
    st.markdown("---")
    st.write("""
    ### Acerca de
    Esta aplicación proporciona una predicción simplificada de la deserción de clientes.
    El modelo se entrena con datos sintéticos que simulan el comportamiento del cliente.
    """)

# Sección principal para la predicción
st.header("Ingrese la Información del Cliente")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geografía', ['Francia', 'Alemania', 'España'])
    gender = st.selectbox('Género', ['Hombre', 'Mujer'])
    age = st.slider('Edad', 18, 95, 35)
    tenure = st.slider('Antigüedad (años)', 0, 10, 3)
    balance = st.number_input('Saldo ($)', min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)

with col2:
    credit_score = st.slider('Puntaje de Crédito', 300, 850, 650)
    estimated_salary = st.number_input('Salario Estimado ($)', min_value=10000.0, max_value=200000.0, value=50000.0, step=1000.0)
    num_of_products = st.slider('Número de Productos', 1, 4, 1)
    has_cr_card = st.selectbox('¿Tiene Tarjeta de Crédito?', ['No', 'Sí'])
    is_active_member = st.selectbox('¿Es Miembro Activo?', ['No', 'Sí'])

# Procesar los datos de entrada cuando el usuario hace clic en "Predecir Deserción"
if st.button("Predecir Probabilidad de Deserción"):
    # Verificar si el modelo existe en el estado de la sesión, de lo contrario crear uno
    if 'model' not in st.session_state:
        with st.spinner("Entrenando un modelo predeterminado..."):
            X_train, y_train = generate_synthetic_data(n_samples=2000)
            model = create_model()
            model.fit(
                X_train, y_train,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            st.session_state.model = model

    # Convertir las entradas al formato apropiado
    geo_france = 1 if geography == 'Francia' else 0
    geo_germany = 1 if geography == 'Alemania' else 0
    gender_val = 1 if gender == 'Mujer' else 0
    has_cc_val = 1 if has_cr_card == 'Sí' else 0
    is_active_val = 1 if is_active_member == 'Sí' else 0

    # Crear la matriz de entrada
    input_data = [
        credit_score, gender_val, age, tenure, balance,
        num_of_products, has_cc_val, is_active_val, estimated_salary,
        geo_france, geo_germany
    ]

    # Normalizar los datos
    input_normalized = normalize_input(input_data)

    # Hacer la predicción
    prediction = st.session_state.model.predict(input_normalized)[0][0]

    # Mostrar los resultados
    st.header("Resultados de la Predicción")

    col1, col2 = st.columns([1, 2])

    with col1:
        if prediction >= 0.5:
            st.error(f"Riesgo de Deserción: {prediction:.2%}")
            st.write("⚠️ ¡Es probable que este cliente abandone!")
        else:
            st.success(f"Riesgo de Deserción: {prediction:.2%}")
            st.write("✅ ¡Es probable que este cliente se quede!")

    with col2:
        # Mostrar los factores que influyeron en la predicción
        st.subheader("Factores Clave")

        factors = []
        if age > 60:
            factors.append("Una mayor edad incrementa el riesgo de deserción")
        if tenure < 3:
            factors.append("Una menor antigüedad incrementa el riesgo de deserción")
        if balance < 10000:
            factors.append("Un menor saldo en la cuenta incrementa el riesgo de deserción")
        if is_active_member == 'No':
            factors.append("Los miembros inactivos tienen un mayor riesgo de deserción")
        if num_of_products > 2:
            factors.append("Los clientes con muchos productos pueden sentirse abrumados")

        if factors:
            for factor in factors:
                st.write("• " + factor)
        else:
            st.write("No se identificaron factores de riesgo importantes")

# Añadir sección de explicación
with st.expander("¿Cómo funciona esto?"):
    st.write("""
    ### Acerca del Modelo

    Esta aplicación utiliza una red neuronal simple con la siguiente arquitectura:
    - Capa de entrada con 11 neuronas (para las características del cliente)
    - Capa oculta con 16 neuronas y activación ReLU
    - Capa oculta con 8 neuronas y activación ReLU
    - Capa de salida con activación sigmoide (probabilidad de deserción)

    El modelo se entrena con datos sintéticos que simulan patrones de comportamiento del cliente.
    En un entorno de producción, esto se reemplazaría con datos históricos reales.

    ### Características Utilizadas
    - **Puntaje de Crédito**: Calificación crediticia del cliente
    - **Género**: Hombre o Mujer
    - **Edad**: Edad del cliente en años
    - **Antigüedad**: Años con la compañía
    - **Saldo**: Saldo de la cuenta
    - **Número de Productos**: Cuántos productos bancarios utiliza
    - **Tarjeta de Crédito**: Si tiene tarjeta de crédito
    - **Miembro Activo**: Si utiliza los servicios activamente
    - **Salario Estimado**: Ingresos estimados del cliente
    - **Geografía**: País del cliente (codificado con un solo valor)
    """)