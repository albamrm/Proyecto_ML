
![Museo de Arte Metropolitano de Nueva York](img/banner.png)

# Clasificación de Obras de Arte del MET

En este proyecto, vamos a intentar clasificar, mediante un modelo de machine learning, obras de arte dentro de la colección del Museo de Arte Metropolitano de Nueva York.

El proceso que hemos seguido para obtener nuestro modelo de machine learning es:

1. Obtención de datos:
Los datos se han sacado del repositorio de [GitHub](https://github.com/metmuseum/openaccess) donde tienen de manera abierta subida su colección de obras.

2. Tratamiento del dataset:
El dataset con el que entrenaremos nuestro modelo cuenta con las siguientes columnas:
    - Is Highlight (bool) 
    - Is Timeline Work (bool) 
    - Is Public Domain (bool)
    - Title (object)
    - Artist (object)
    - Date (int64)
    - Culture (object)
    - Period (object)
    - Object Name (object)
    - Medium (object)
    - Dimensions (object)
    - Country (object)
    - Acquisition Year (int64)
    - Credit Line (object)
               
3. Entrenamiento y evaluación del modelo:
A pesar de utilizar la validación cruzada para ver que modelo era el que mejor se adaptaba a nuestro fin, la cual nos recomendaba utilizar el modelo Extreme Gradient Boosting Classifier, nosotros hemos entrenado y optimizado varios modelos para ver cuál daba mejor resultado de todos.
            
4. Clasificaciones y Recomendaciones:
Para clasificar las obras de arte, hemos realizado una aplicación con streamlit, donde introduciendo diversos parámetros te dirá si es una obra destacada o no.

Para más información sobre el proyecto, está a disposición la memoria, la cual analiza y redirige a todos los pasos realizados.