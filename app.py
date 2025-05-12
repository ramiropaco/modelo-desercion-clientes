import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set up the streamlit page
st.set_page_config(page_title="Predictor de Fuga de Clientes", layout="wide")
st.title("Predictor de Fuga de Clientes")

# Create a model using scikit-learn instead of TensorFlow
def create_model():
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,
        random_state=42
    )
    return model

# Sample synthetic data generator (instead of reading from CSV)
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate synthetic features
    credit_scores = np.random.randint(300, 850, n_samples)
    genders = np.random.randint(0, 2, n_samples)  # 0: Male, 1: Female
    ages = np.random.randint(18, 95, n_samples)
    tenures = np.random.randint(0, 11, n_samples)
    balances = np.random.uniform(0, 250000, n_samples)
    products = np.random.randint(1, 5, n_samples)
    credit_cards = np.random.randint(0, 2, n_samples)
    active_members = np.random.randint(0, 2, n_samples)
    salaries = np.random.uniform(10000, 200000, n_samples)
    
    # Geography one-hot encoded (3 countries)
    geo_france = np.random.randint(0, 2, n_samples)
    geo_germany = np.random.randint(0, 2, n_samples)
    
    # Create feature matrix
    X = np.column_stack([
        credit_scores, genders, ages, tenures, balances, 
        products, credit_cards, active_members, salaries,
        geo_france, geo_germany
    ])
    
    # Generate synthetic target (churn)
    # Higher probability of churn for:
    # - Higher age
    # - Lower tenure
    # - Lower account balance
    # - Not active members
    churn_prob = (
        0.1 + 
        0.3 * (ages / 95) +
        0.2 * (1 - tenures / 10) +
        0.2 * (1 - balances / 250000) +
        0.2 * (1 - active_members)
    )
    
    y = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Create DataFrame for better handling
    columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
               'NumProducts', 'HasCreditCard', 'IsActiveMember', 'EstimatedSalary',
               'Geography_France', 'Geography_Germany']
    X_df = pd.DataFrame(X, columns=columns)
    
    return X_df, y

def preprocess_input(input_data):
    """Scale input data for prediction"""
    scaler = StandardScaler()
    # We need to fit the scaler with some data first
    X_sample, _ = generate_synthetic_data(n_samples=1000)
    scaler.fit(X_sample)
    
    # Scale the input
    return scaler.transform(input_data.values.reshape(1, -1))

# Create sidebar for model training (if user wants to retrain)
with st.sidebar:
    st.header("Entrenamiento del Modelo")
    st.write("Entrene el modelo con datos sintéticos")
    
    if st.button("Entrenar Modelo"):
        with st.spinner("Entrenando modelo con datos sintéticos..."):
            # Generate synthetic data
            X_train, y_train = generate_synthetic_data(n_samples=5000)
            
            # Create and train model
            model = create_model()
            model.fit(X_train, y_train)
            
            # Store the model in session state
            st.session_state.model = model
            st.session_state.importance = pd.DataFrame({
                'Característica': X_train.columns,
                'Importancia': model.feature_importances_
            }).sort_values('Importancia', ascending=False)
            
            st.success("¡Modelo entrenado exitosamente!")
    
    # Add some information about the app
    st.markdown("---")
    st.write("""
    ### Acerca de
    Esta aplicación proporciona una predicción simplificada de abandono de clientes.
    El modelo se entrena con datos sintéticos que simulan el comportamiento del cliente.
    """)

# Main section for prediction
st.header("Ingrese la Información del Cliente")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('País', ['Francia', 'Alemania', 'España'])
    gender = st.selectbox('Género', ['Masculino', 'Femenino'])
    age = st.slider('Edad', 18, 95, 35)
    tenure = st.slider('Antigüedad (años)', 0, 10, 3)
    balance = st.number_input('Saldo ($)', min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)

with col2:
    credit_score = st.slider('Puntuación Crediticia', 300, 850, 650)
    estimated_salary = st.number_input('Salario Estimado ($)', min_value=10000.0, max_value=200000.0, value=50000.0, step=1000.0)
    num_of_products = st.slider('Número de Productos', 1, 4, 1)
    has_cr_card = st.selectbox('Tiene Tarjeta de Crédito', ['No', 'Sí'])
    is_active_member = st.selectbox('Es un Miembro Activo', ['No', 'Sí'])

# Process input data when user clicks predict
if st.button("Predecir Probabilidad de Abandono"):
    # Check if model exists in session state, otherwise create one
    if 'model' not in st.session_state:
        with st.spinner("Entrenando un modelo predeterminado..."):
            X_train, y_train = generate_synthetic_data(n_samples=2000)
            model = create_model()
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.session_state.importance = pd.DataFrame({
                'Característica': X_train.columns,
                'Importancia': model.feature_importances_
            }).sort_values('Importancia', ascending=False)
    
    # Convert inputs to appropriate format
    geo_france = 1 if geography == 'Francia' else 0
    geo_germany = 1 if geography == 'Alemania' else 0
    gender_val = 1 if gender == 'Femenino' else 0
    has_cc_val = 1 if has_cr_card == 'Sí' else 0
    is_active_val = 1 if is_active_member == 'Sí' else 0
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore': [credit_score], 
        'Gender': [gender_val], 
        'Age': [age], 
        'Tenure': [tenure], 
        'Balance': [balance],
        'NumProducts': [num_of_products], 
        'HasCreditCard': [has_cc_val], 
        'IsActiveMember': [is_active_val], 
        'EstimatedSalary': [estimated_salary],
        'Geography_France': [geo_france], 
        'Geography_Germany': [geo_germany]
    })
    
    # Make prediction
    prediction_proba = st.session_state.model.predict_proba(input_data)[0][1]
    
    # Display results
    st.header("Resultados de la Predicción")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if prediction_proba >= 0.5:
            st.error(f"Riesgo de Abandono: {prediction_proba:.2%}")
            st.write("⚠️ ¡Este cliente probablemente abandonará!")
        else:
            st.success(f"Riesgo de Abandono: {prediction_proba:.2%}")
            st.write("✅ ¡Este cliente probablemente se quedará!")
    
    with col2:
        # Show the factors that influenced the prediction
        st.subheader("Factores Clave")
        
        factors = []
        if age > 60:
            factors.append("La edad avanzada aumenta el riesgo de abandono")
        if tenure < 3:
            factors.append("Menor antigüedad aumenta el riesgo de abandono")
        if balance < 10000:
            factors.append("Menor saldo de cuenta aumenta el riesgo de abandono")
        if is_active_member == 'No':
            factors.append("Los miembros inactivos tienen mayor riesgo de abandono")
        if num_of_products > 2:
            factors.append("Los clientes con muchos productos pueden sentirse abrumados")
            
        if factors:
            for factor in factors:
                st.write("• " + factor)
        else:
            st.write("No se identificaron factores de riesgo importantes")
    
    # Show feature importance
    if 'importance' in st.session_state:
        st.subheader("Importancia de las Características")
        st.bar_chart(st.session_state.importance.set_index('Característica'))

# Add explanation section
with st.expander("¿Cómo funciona esto?"):
    st.write("""
    ### Sobre el Modelo
    
    Esta aplicación utiliza un clasificador Random Forest con:
    - 100 árboles de decisión
    - Profundidad máxima de 5 para evitar sobreajuste
    - El modelo toma en cuenta 11 características para predecir el abandono de clientes
    
    El modelo se entrena con datos sintéticos que simulan patrones de comportamiento del cliente.
    En un entorno de producción, esto se reemplazaría con datos históricos reales.
    
    ### Características Utilizadas
    - **Puntuación Crediticia**: Calificación crediticia del cliente
    - **Género**: Masculino o Femenino
    - **Edad**: Edad del cliente en años
    - **Antigüedad**: Años con la empresa
    - **Saldo**: Saldo de la cuenta
    - **Número de Productos**: Cuántos productos bancarios utilizan
    - **Tarjeta de Crédito**: Si tienen una tarjeta de crédito
    - **Miembro Activo**: Si están utilizando activamente los servicios
    - **Salario Estimado**: Ingresos estimados del cliente
    - **País**: País del cliente (codificado one-hot)
    """)
