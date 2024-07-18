# Imports

import joblib
import os
import pandas as pd
import streamlit as st

###############################################################################

# Cargar los modelos y datos

data = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'MET_Limpio.csv'))
image = os.path.join(os.getcwd(), '..', 'img', 'banner.png')

preprocessor = joblib.load(os.path.join(os.getcwd(), '..', 'modelos', 'preprocessor_completo.pkl'))
selector = joblib.load(os.path.join(os.getcwd(), '..', 'modelos', 'selector_completo.pkl'))
modelo_xgb = joblib.load(os.path.join(os.getcwd(), '..', 'modelos', 'modelo_xgb_completo.pkl'))

###############################################################################

# Funciones

def preprocess_input(art):
    for key in art:
        if isinstance(art[key], str):
            art[key] = art[key].lower()
    return art

def predict_highlight(art):
    
    art = preprocess_input(art)
    art_df = pd.DataFrame([art])
    
    preprocessor_art = preprocessor.transform(art_df)
    
    selector_art = selector.transform(preprocessor_art)
    
    prediction = modelo_xgb.predict(selector_art)
    probability = modelo_xgb.predict_proba(selector_art)
    
    prediction_result = "Sí" if prediction[0] else "No"
    highlight_probability = probability[0][1] * 100

    return prediction_result, highlight_probability

###############################################################################

# Aplicación

# Título de la aplicación
st.title('Museo de Arte Metropolitano')

# Barra lateral para la navegación
st.sidebar.title('Navegación')
selection = st.sidebar.radio('Ir a', ['El Museo', 'Clasificador de Obras de Arte'])

# Definición de cada página
def page1():
    st.header('El Museo')
    st.write('El Museo de Arte Metropolitano de Nueva York, comúnmente conocido como el MET, es uno de los museos de arte más grandes y prestigiosos del mundo.')
    st.write('Fundado en 1870, el MET alberga una colección de más de dos millones de obras de arte que abarcan 5,000 años de historia. \
             Sus colecciones incluyen piezas de todas las culturas y épocas, desde antigüedades egipcias hasta arte contemporáneo.')
    st.write('Con su sede principal en la icónica Quinta Avenida y una segunda ubicación en el Met Cloisters en Fort Tryon Park, \
             el museo ofrece una experiencia cultural inigualable, atrayendo a millones de visitantes de todo el mundo cada año.')
    st.image(image, caption = 'Museo de Arte Metropolitano de Nueva York', use_column_width = True)

def page2():
    st.header('Clasificador de Obras de Arte')
    st.write('Ingrese los detalles de la obra:')

    # Inputs del usuario en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input('Título de la obra')
        artist = st.text_input('Artista')
        date = st.number_input('Fecha', min_value = 0, max_value = 9999)
        date_era = st.selectbox('Era', ['d.C.', 'a.C.'])
        if date_era == 'a.C.':
            date = -date
        culture = st.text_input('Cultura')

    with col2:
        period = st.text_input('Periodo')
        object_name = st.text_input('Tipo de objeto')
        medium = st.text_input('Material')
        dimensions = st.text_input('Dimensiones')
        country = st.text_input('País')

    # Crear un diccionario para la información
    input_user = {
        'Is Timeline Work' : False,
        'Is Public Domain' : False,
        'Title' : title,
        'Artist' : artist,
        'Date' : date,
        'Culture' : culture,
        'Period' : period,
        'Object Name' : object_name,
        'Medium' : medium,
        'Dimensions' : dimensions,
        'Country' : country,
        'Acquisition Year' : 0,
        'Credit Line' : ""
    }

    # Inicializar variables de resultado
    prediction_result = None
    highlight_probability = None

    # Botón para predecir
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button('Predecir'):
            prediction_result, highlight_probability = predict_highlight(input_user)

    # Mostrar resultado
    if prediction_result is not None and highlight_probability is not None:
        st.subheader('Resultado de la Predicción')
        st.success(f'¿Sería una obra de arte destacada?: {prediction_result}')
        st.info(f'Probabilidad de ser destacada: {highlight_probability:.2f}%')

# Selección de la página a mostrar
if selection == 'El Museo':
    page1()
elif selection == 'Clasificador de Obras de Arte':
    page2()