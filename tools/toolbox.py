# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from scipy.stats import f_oneway, mannwhitneyu, pearsonr, shapiro, ttest_ind
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE, SelectFromModel, SelectKBest, SequentialFeatureSelector
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, precision_score, recall_score

###############################################################################

# Función | plot_categorical_distribution

def plot_categorical_distribution(df, categorical_columns, relative = False, show_values = False):

    """
    Muestra la distribución de columnas categóricas en un DataFrame.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    categorical_columns (list): Lista de nombres de columnas categóricas a graficar.
    relative (bool, opcional): Si es True, muestra la frecuencia relativa. Por defecto es False.
    show_values (bool, opcional): Si es True, muestra los valores en las barras. Por defecto es False.

    Returns:
    None
    """

    # Calcula el número de filas necesarias para los gráficos
    num_columns = len(categorical_columns)
    num_rows = (num_columns // 2) + (num_columns % 2)

    # Crea los subplots
    fig, axes = plt.subplots(num_rows, 2, figsize = (15, 5 * num_rows))
    axes = axes.flatten()

    # Genera gráficos de barras para cada columna categórica
    for i, col in enumerate(categorical_columns):
        ax = axes[i]
        if relative:
            total = df[col].value_counts().sum()
            series = df[col].value_counts().apply(lambda x: x / total)
            sns.barplot(x = series.index, y = series, ax = ax, palette = 'viridis', hue = series.index, legend = False)
            ax.set_ylabel('Frecuencia Relativa')
        else:
            series = df[col].value_counts()
            sns.barplot(x = series.index, y = series, ax = ax, palette = 'viridis', hue = series.index, legend = False)
            ax.set_ylabel('Frecuencia')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis = 'x', rotation = 45)

        if show_values:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                            ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')

    # Oculta ejes vacíos
    for j in range(i + 1, num_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

###############################################################################

# Función | plot_categorical_relationship

def plot_categorical_relationship(df, cat_col1, cat_col2, relative_freq = False, show_values = False, size_group = 5):

    """
    Muestra la relación entre dos columnas categóricas en un DataFrame.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    cat_col1 (str): Nombre de la primera columna categórica.
    cat_col2 (str): Nombre de la segunda columna categórica.
    relative_freq (bool, opcional): Si es True, muestra la frecuencia relativa. Por defecto es False.
    show_values (bool, opcional): Si es True, muestra los valores en las barras. Por defecto es False.
    size_group (int, opcional): Tamaño del grupo para agrupar los datos. Por defecto es 5.

    Returns:
    None
    """

    # Prepara los datos
    count_data = df.groupby([cat_col1, cat_col2]).size().reset_index(name = 'count')
    total_counts = df[cat_col1].value_counts()
    
    # Convierte a frecuencias relativas si se solicita
    if relative_freq:
        count_data['count'] = count_data.apply(lambda x : x['count'] / total_counts[x[cat_col1]], axis = 1)

    # Si hay más de size_group categorías en cat_col1, las divide en grupos de size_group
    unique_categories = df[cat_col1].unique()
    if len(unique_categories) > size_group:
        num_plots = int(np.ceil(len(unique_categories) / size_group))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * size_group:(i + 1) * size_group]
            data_subset = count_data[count_data[cat_col1].isin(categories_subset)]

            # Crea el gráfico
            plt.figure(figsize = (10, 6))
            ax = sns.barplot(x = cat_col1, y = 'count', hue = cat_col2, data = data_subset, order = categories_subset)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {cat_col1} y {cat_col2} - Grupo {i + 1}')
            plt.xlabel(cat_col1)
            plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
            plt.xticks(rotation = 45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha = 'center', va = 'center', fontsize = 10, color = 'black', xytext = (0, size_group),
                                textcoords = 'offset points')

            # Muestra el gráfico
            plt.show()

    else:
        # Crea el gráfico para menos de size_group categorías
        plt.figure(figsize = (10, 6))
        ax = sns.barplot(x = cat_col1, y = 'count', hue = cat_col2, data = count_data)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {cat_col1} y {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel('Frecuencia' if relative_freq else 'Conteo')
        plt.xticks(rotation = 45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', fontsize = 10, color = 'black', xytext = (0, size_group),
                            textcoords = 'offset points')

        # Muestra el gráfico
        plt.show()

###############################################################################

# Función | plot_categorical_numerical_relationship

def plot_categorical_numerical_relationship(df, categorical_col, numerical_col, show_values = False, measure = 'mean'):

    """
    Muestra la relación entre una columna categórica y una columna numérica utilizando gráficos de barras.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    categorical_col (str): Nombre de la columna categórica.
    numerical_col (str): Nombre de la columna numérica.
    show_values (bool, opcional): Si es True, muestra los valores en las barras. Por defecto es False.
    measure (str, opcional): Medida de tendencia central a usar ('mean' o 'median'). Por defecto es 'mean'.

    Returns:
    None
    """

    # Calcula la medida de tendencia central (mean o median)
    if measure == 'median':
        grouped_data = df.groupby(categorical_col)[numerical_col].median()
    else:
        # Por defecto, usa la media
        grouped_data = df.groupby(categorical_col)[numerical_col].mean()

    # Ordena los valores
    grouped_data = grouped_data.sort_values(ascending = False)

    # Si hay más de 5 categorías, las divide en grupos de 5
    if grouped_data.shape[0] > 5:
        unique_categories = grouped_data.index.unique()
        num_plots = int(np.ceil(len(unique_categories) / 5))

        for i in range(num_plots):
            # Selecciona un subconjunto de categorías para cada gráfico
            categories_subset = unique_categories[i * 5:(i + 1) * 5]
            data_subset = grouped_data.loc[categories_subset]

            # Crea el gráfico
            plt.figure(figsize = (10, 6))
            ax = sns.barplot(x = data_subset.index, y = data_subset.values)

            # Añade títulos y etiquetas
            plt.title(f'Relación entre {categorical_col} y {numerical_col} - Grupo {i + 1}')
            plt.xlabel(categorical_col)
            plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
            plt.xticks(rotation = 45)

            # Mostrar valores en el gráfico
            if show_values:
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha = 'center', va = 'center', fontsize = 10, color = 'black', xytext = (0, 5),
                                textcoords = 'offset points')

            # Muestra el gráfico
            plt.show()

    else:
        # Crea el gráfico para menos de 5 categorías
        plt.figure(figsize = (10, 6))
        ax = sns.barplot(x = grouped_data.index, y = grouped_data.values)

        # Añade títulos y etiquetas
        plt.title(f'Relación entre {categorical_col} y {numerical_col}')
        plt.xlabel(categorical_col)
        plt.ylabel(f'{measure.capitalize()} de {numerical_col}')
        plt.xticks(rotation = 45)

        # Mostrar valores en el gráfico
        if show_values:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'center', fontsize = 10, color = 'black', xytext = (0, 5),
                            textcoords = 'offset points')

        # Muestra el gráfico
        plt.show()

###############################################################################

# Función | plot_combined_graphs

def plot_combined_graphs(df, columns, whisker_width = 1.5, bins = None):

    """
    Muestra gráficos combinados de histograma y KDE junto con boxplots para una lista de columnas numéricas.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    columns (list): Lista de nombres de columnas a graficar.
    whisker_width (float, opcional): Ancho de los bigotes en el boxplot. Por defecto es 1.5.
    bins (int, opcional): Número de bins para el histograma. Por defecto es None, que usa 'auto'.

    Returns:
    None
    """

    num_cols = len(columns)
    if num_cols:
        
        fig, axes = plt.subplots(num_cols, 2, figsize = (12, 5 * num_cols))
        print(axes.shape)

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                # Histograma y KDE
                sns.histplot(df[column], kde = True, ax = axes[i,0] if num_cols > 1 else axes[0], bins = "auto" if not bins else bins)
                if num_cols > 1:
                    axes[i,0].set_title(f'Histograma y KDE de {column}')
                else:
                    axes[0].set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x = df[column], ax = axes[i,1] if num_cols > 1 else axes[1], whis = whisker_width)
                if num_cols > 1:
                    axes[i,1].set_title(f'Boxplot de {column}')
                else:
                    axes[1].set_title(f'Boxplot de {column}')

        plt.tight_layout()
        plt.show()

###############################################################################

# Función | plot_grouped_boxplots

def plot_grouped_boxplots(df, cat_col, num_col):

    """
    Muestra diagramas de caja (boxplots) agrupados por una columna categórica.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    cat_col (str): Nombre de la columna categórica para agrupar.
    num_col (str): Nombre de la columna numérica para los boxplots.

    Returns:
    None
    """

    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    group_size = 5

    # Genera gráficos por grupos de categorías
    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize = (10, 6))
        sns.boxplot(x = cat_col, y = num_col, data = subset_df)
        plt.title(f'Boxplots de {num_col} para {cat_col} (Grupo {i//group_size + 1})')
        plt.xticks(rotation = 45)
        plt.show()

###############################################################################

# Función | plot_grouped_histograms

def plot_grouped_histograms(df, cat_col, num_col, group_size = 5):

    """
    Muestra histogramas agrupados por una columna categórica.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    cat_col (str): Nombre de la columna categórica para agrupar.
    num_col (str): Nombre de la columna numérica para los histogramas.
    group_size (int, opcional): Tamaño del grupo para agrupar los datos. Por defecto es 5.

    Returns:
    None
    """

    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)

    # Genera histogramas por grupos de categorías
    for i in range(0, num_cats, group_size):
        subset_cats = unique_cats[i:i+group_size]
        subset_df = df[df[cat_col].isin(subset_cats)]
        
        plt.figure(figsize = (10, 6))
        for cat in subset_cats:
            sns.histplot(subset_df[subset_df[cat_col] == cat][num_col], kde = True, label = str(cat))
        
        plt.title(f'Histogramas de {num_col} para {cat_col} (Grupo {i//group_size + 1})')
        plt.xlabel(num_col)
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.show()

###############################################################################

# Función | scatter_plot_with_correlation

def scatter_plot_with_correlation(df, x_col, y_col, point_size = 50, show_correlation = False):

    """
    Crea un diagrama de dispersión entre dos columnas y opcionalmente muestra la correlación.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    x_col (str): Nombre de la columna para el eje X.
    y_col (str): Nombre de la columna para el eje Y.
    point_size (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    show_correlation (bool, opcional): Si es True, muestra la correlación en el gráfico. Por defecto es False.

    Returns:
    None
    """

    plt.figure(figsize = (10, 6))
    sns.scatterplot(data = df, x = x_col, y = y_col, s = point_size)

    if show_correlation:
        correlation = df[[x_col, y_col]].corr().iloc[0, 1]
        plt.title(f'Diagrama de Dispersión con Correlación: {correlation:.2f}')
    else:
        plt.title('Diagrama de Dispersión')

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()

###############################################################################

# Función | bubble_plot

def bubble_plot(df, x_col, y_col, size_col, scale = 1000):

    """
    Crea un scatter plot usando dos columnas para los ejes X e Y, y una tercera columna para determinar el tamaño de los puntos.

    Args:
    df (pd.DataFrame): DataFrame de pandas.
    x_col (str): Nombre de la columna para el eje X.
    y_col (str): Nombre de la columna para el eje Y.
    size_col (str): Nombre de la columna para determinar el tamaño de los puntos.
    scale (int, opcional): Escala para ajustar el tamaño de las burbujas. Por defecto es 1000.

    Returns:
    None
    """

    # Asegúrate de que los valores de tamaño sean positivos
    sizes = (df[size_col] - df[size_col].min() + 1) / scale

    plt.scatter(df[x_col], df[y_col], s = sizes)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Burbujas de {x_col} vs {y_col} con Tamaño basado en {size_col}')
    plt.show()

###############################################################################

# Función | describe_df

def describe_df(df):

    """
    Muestra diferentes tipos de datos de un DataFrame.

    Args:
    df (pd.DataFrame): DataFrame que queremos describir.

    Returns:
    pd.DataFrame: Devuelve un DataFrame con información sobre el tipo de datos, el porcentaje de valores nulos,
    la cantidad de valores únicos y el porcentaje de cardinalidad, de todas las variables de este.
    """

    # Obtener tipos de columnas
    types = df.dtypes

    # Calcular porcentaje de valores nulos
    missing_percentage = (df.isnull().mean() * 100).round(2)

    # Obtener valores únicos
    unique_values = df.nunique()

    # Obtener porcentaje de cardinalidad
    cardinality_percentage = ((unique_values / len(df)) * 100).round(2)

    # Crear un DataFrame con la información recopilada
    result_describe = pd.DataFrame({
        "Tipos": types,
        "% Faltante": missing_percentage,
        "Valores Únicos": unique_values,
        "% Cardinalidad": cardinality_percentage
    })

    return result_describe.T

###############################################################################

# Función | classify_variables

def classify_variables(df, category_threshold, continuous_threshold):

    """
    Clasifica las variables del DataFrame en tipos sugeridos.

    Args:
    df (pd.DataFrame): DataFrame cuyas variables queremos clasificar.
    category_threshold (int): Umbral para considerar una variable como categórica.
    continuous_threshold (float): Umbral para considerar una variable como numérica continua.

    Returns:
    pd.DataFrame: Devuelve un DataFrame con dos columnas: "variable_name" y "suggested_type".
    """

    # Crear una lista con la tipificación sugerida para cada variable
    classification_list = []

    # Iterar sobre cada columna del DataFrame
    for column in df.columns:

        # Obtener la cardinalidad
        cardinality = len(df[column].unique())

        # Determinar el tipo
        if cardinality == 2:
            classification_list.append("Binaria")
        elif cardinality < category_threshold:
            classification_list.append("Categórica")
        else:
            cardinality_percentage = (cardinality / len(df)) * 100
            if cardinality_percentage >= continuous_threshold:
                classification_list.append("Numérico Continuo")
            else:
                classification_list.append("Numérico Discreto")

    # Agregar el resultado a un nuevo DataFrame con los resultados
    result_classification = pd.DataFrame({"nombre_variable": df.columns.tolist(), "tipo_sugerido": classification_list})

    return result_classification

###############################################################################

# Función | get_num_features_for_regression

def get_num_features_for_regression(df, target_col, corr_threshold, pvalue = None):

    """
    Devuelve una lista con las columnas numéricas del DataFrame cuya correlación con la columna designada por "target_col" 
    sea superior en valor absoluto al valor dado por "corr_threshold". Si "pvalue" no es None, solo devolverá las columnas 
    que también superen el test de hipótesis con significación mayor o igual a 1-pvalue.

    Args:
    df (pd.DataFrame): DataFrame de Pandas.
    target_col (str): Nombre de la columna objetivo del DataFrame.
    corr_threshold (float): Valor de correlación para seleccionar las columnas.
    pvalue (float, opcional): Valor p para el test de hipótesis. Por defecto es None.

    Returns:
    list: Lista de las columnas que cumplen con los criterios especificados.
    """

    # Comprobaciones
    if not isinstance(df, pd.DataFrame):
        print("Error: No has introducido un DataFrame válido de pandas.")
        return None
    
    # Comprobar si target_col está en el DataFrame
    if target_col not in df.columns:
        print(f"Error: La columna {target_col} no está en el DataFrame.")
        return None
    
    # Comprobar si target_col es numérico
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna {target_col} no es numérica.")
        return None
    
    # Comprobar si corr_threshold está entre 0 y 1
    if type(corr_threshold) != float and type(corr_threshold) != int:
        print("Error: El parámetro corr_threshold", corr_threshold, " no es un número.")
    if not 0 <= corr_threshold <= 1:
        print("Error: El corr_threshold debe estar entre 0 y 1.")
        return None
    
    # Comprobar que pvalue es un entero o un float, y está en el rango [0,1]
    if pvalue is not None:
        if type(pvalue) != float and type(pvalue) != int:
            print("Error: El parámetro pvalue", pvalue, " no es un número.")
            return None
        elif not (0 <= pvalue <= 1):
            print("Error: El parámetro pvalue", pvalue, " está fuera del rango [0,1].")
            return None
        
    # Se usa la función classify_variables para identificar las variables numéricas
    var_class = classify_variables(df, 5, 9)
    num_cols = var_class[(var_class["tipo_sugerido"] == "Numérico Continuo") | (var_class["tipo_sugerido"] == "Numérico Discreto")]["nombre_variable"].tolist()

    # Comprobación de que hay alguna columna numérica para relacionar
    if len(num_cols) == 0:
        print("Error: No hay ninguna columna numérica o discreta a analizar que cumpla con los requisitos establecidos en los umbrales.")
    else:
        # Se realizan las correlaciones y se eligen las que superen el umbral
        correlations = df[num_cols].corr()[target_col]
        filtered_columns = correlations[abs(correlations) > corr_threshold].index.tolist()
        if target_col in filtered_columns:
            filtered_columns.remove(target_col)
    
        # Comprobación de que si se introduce un pvalue pase los tests de hipótesis (Pearson)
        if pvalue is not None:
            final_columns = []
            for col in filtered_columns:
                specific_pvalue = pearsonr(df[col], df[target_col])[1]
                if pvalue < (1 - specific_pvalue):
                    final_columns.append(col)
            filtered_columns = final_columns.copy()

    if len(filtered_columns) == 0:
        print("No hay columna numérica que cumpla con las especificaciones de umbral de correlación y/o pvalue.")
        return None

    return filtered_columns

###############################################################################

# Función | plot_num_features_for_regression

def plot_num_features_for_regression(df, target_col = "", columns = [], corr_threshold = 0, pvalue = None):

    """
    Genera un pairplot del DataFrame considerando la columna designada por "target_col" y aquellas 
    incluidas en "columns" que cumplan que su correlación con "target_col" es superior en valor absoluto a "corr_threshold", 
    y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de significación estadística. 
    La función devolverá los valores de "columns" que cumplan con las condiciones anteriores.

    Args:
    df (pd.DataFrame): DataFrame de Pandas.
    target_col (str): Nombre de la columna objetivo del DataFrame.
    columns (list): Lista de nombres de columnas a considerar. Por defecto es una lista vacía.
    corr_threshold (float): Valor de correlación para seleccionar las columnas. Por defecto es 0.
    pvalue (float, opcional): Valor p para el test de hipótesis. Por defecto es None.

    Returns:
    list: Lista de las columnas correlacionadas.
    """

    # Comprobaciones
    
    # Si la lista de columnas está vacía, asignar todas las variables numéricas del DataFrame
    if not columns:
        columns = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    
    # Si se proporciona una lista de columnas, comprobar si están en el DataFrame
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: La columna {col} no está en el DataFrame.")
                return None
    
    filtered_columns = get_num_features_for_regression(df, target_col, corr_threshold, pvalue)
    
    re_filtered_columns = []
    for col in filtered_columns:
        if col in columns:
            re_filtered_columns.append(col)

    # Divide la lista de columnas filtradas en grupos de máximo cinco columnas
    grouped_columns = [re_filtered_columns[i:i+4] for i in range(0, len(re_filtered_columns), 4)]
    
    # Generar pairplots para cada grupo de columnas
    for group in grouped_columns:
        sns.pairplot(df[[target_col] + group])
        plt.show()
    
    # Devolver la lista de columnas filtradas
    return re_filtered_columns

###############################################################################

# Función | get_cat_features_for_regression

def get_cat_features_for_regression(df, target_col, pvalue = 0.05):

    """
    Devuelve una lista con las columnas categóricas del DataFrame que superen el test de hipótesis para la regresión con "target_col".

    Args:
    df (pd.DataFrame): DataFrame a analizar.
    target_col (str): Nombre de la columna objetivo del DataFrame.
    pvalue (float): Valor de significación a superar.

    Returns:
    list: Lista de las columnas categóricas que superan el test de hipótesis.
    """
    
    # Comprobar que df es un DataFrame
    if not (isinstance(df, pd.DataFrame)):
        print("Error: El parámetro df", df, " no es un DataFrame.")
        return None
    
    # Comprobar que pvalue es un entero o un float, y está en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, " no es un número.")
        return None
    elif not (0 <= pvalue <= 1):
        print("Error: El parámetro pvalue", pvalue, " está fuera del rango [0,1].")
        return None
        
    # Comprobar que target_col es una variable del DataFrame
    if not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del DataFrame.")
        return None  
      
    # Comprobar que target_col es una variable numérica continua
    var_class = classify_variables(df, 5, 9)

    if not (var_class.loc[var_class["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérico Continuo") or (var_class.loc[var_class["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérico Discreto"):
        print("Error: El parámetro target_col ", target_col , " no es una columna numérica del DataFrame bajo los criterios de umbrales establecidos.")
        return None

    # Hacer una lista con las columnas categóricas o binarias
    cat_cols = var_class[(var_class["tipo_sugerido"] == "Categórica") | (var_class["tipo_sugerido"] == "Binaria")]["nombre_variable"].tolist()
    if len(cat_cols) == 0:
        return None
         
    # Inicializamos la lista de salida
    selected_cols = []

    # Por cada columna categórica o binaria 
    for value in cat_cols:
        groups = df[value].unique()  # Obtener los valores únicos de la columna categórica
        if len(groups) == 2:
            group_a = df.loc[df[value] == groups[0]][target_col]
            group_b = df.loc[df[value] == groups[1]][target_col]
            _, p = shapiro(group_a)  # Usamos la prueba de normalidad de Shapiro-Wilk para saber si siguen una distribución normal o no
            _, p2 = shapiro(group_b)
            if p < 0.05 and p2 < 0.05:
                stat, p_val = ttest_ind(group_a, group_b)  # Aplicamos el t-Student si siguen una distribución normal
            else:
                u_stat, p_val = mannwhitneyu(group_a, group_b)  # Aplicamos el test U de Mann si no la siguen
        else:
            cat_values = [df[df[value] == group][target_col] for group in groups]  # obtenemos los grupos y los incluimos en una lista
            f_val, p_val = stats.f_oneway(*cat_values)  # Aplicamos el test ANOVA
        if p_val < pvalue:
            selected_cols.append(value)  # Si supera el test correspondiente añadimos la variable a la lista de salida
        
    if len(selected_cols) == 0:
        print("No hay columna categórica o binaria que cumpla con las especificaciones.")
        return None
       
    return selected_cols

###############################################################################

# Función | plot_cat_features_for_regression

def plot_cat_features_for_regression(df, target_col = "", columns = [], pvalue = 0.05):

    """
    Genera histogramas agrupados para visualizar la relación entre las columnas categóricas de un DataFrame y una columna objetivo, 
    filtrando aquellas columnas que pasan una prueba de hipótesis según un nivel de significación especificado.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo para la regresión. Valor por defecto es una cadena vacía.
    columns (list): Lista de nombres de columnas a considerar. Por defecto es una lista vacía.
    pvalue (float): Nivel de significación para la prueba de hipótesis. Por defecto es 0.05.

    Returns:
    list: Lista de nombres de columnas que cumplen con el criterio de significación especificado.
    """

    # Comprobar que df es un DataFrame
    if not (isinstance(df, pd.DataFrame)):
        print("Error: El parámetro df", df, " no es un DataFrame.")
        return None
    
    # Comprobar que pvalue es un entero o un float, y está en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, " no es un número.")
        return None
    elif not (0 <= pvalue <= 1):
        print("Error: El parámetro pvalue", pvalue, " está fuera del rango [0,1].")
        return None
      
    # Comprobar que target_col es una variable numérica continua
    var_class = classify_variables(df, 5, 9)

    # Si no hay target_col, pedir al usuario la introducción de una
    if target_col == "":
        print("Por favor, introduce una columna objetivo con la que realizar el análisis.")
        return "plot_cat_features_for_regression(df, target_col= ___, ...)"

    # Comprobar que target_col es una variable del DataFrame
    if not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del DataFrame.")
        return None  

    if not (var_class.loc[var_class["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérico Continuo") or (var_class.loc[var_class["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérico Discreto"):
        print("Error: El parámetro target_col ", target_col , " no es una columna numérica del DataFrame bajo los criterios de umbrales establecidos.")
        return None

    # Si la lista de columnas está vacía, asignar todas las variables categóricas del DataFrame
    if not columns:
        columns = var_class[var_class["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()
    
    # Si se proporciona una lista de columnas, comprobar si están en el DataFrame
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: La columna {col} no está en el DataFrame.")
                return None    

    df_columns = df[columns]
    df_columns[target_col] = df[target_col]       
    
    filtered_columns = get_cat_features_for_regression(df_columns, target_col, pvalue)

    # Generar los histogramas agrupados para cada columna filtrada
    for col in filtered_columns:        
        plot_grouped_histograms(df, cat_col = col, num_col = target_col, group_size = len(df[col].unique()))
    
    # Devolver la lista de columnas filtradas
    return filtered_columns

###############################################################################

# Función | eval_model

def eval_model(target, predictions, problem_type, metrics):

    """
    Evalua un modelo de Machine Learning utilizando diferentes métricas para problemas de regresión o clasificación.

    Args:
    target (array-like): Valores del target.
    predictions (array-like): Valores predichos por el modelo.
    problem_type (str): Tipo de problema, puede ser "regresion" o "clasificacion".
    metrics (list): Lista de métricas a calcular. 
                    Para problemas de regresión: "RMSE", "MAE", "MAPE", "GRAPH".
                    Para problemas de clasificación: "ACCURACY", "PRECISION", "RECALL", "CLASS_REPORT", "MATRIX", "MATRIX_RECALL", "MATRIX_PRED", "PRECISION_X", "RECALL_X".

    Returns:
    tuple: Tupla con los resultados de las métricas especificadas.
    """

    results = []

    # Regresión
    if problem_type == "regresion":

        for metric in metrics:
            
            if metric == "RMSE":
                rmse = np.sqrt(mean_squared_error(target, predictions))
                print(f"RMSE: {rmse}")
                results.append(rmse)
            
            elif metric == "MAE":
                mae = mean_absolute_error(target, predictions)
                print(f"MAE: {mae}")
                results.append(mae)

            elif metric == "MAPE":
                try:
                    mape = np.mean(np.abs((target - predictions) / target)) * 100
                    print(f"MAPE: {mape}")
                    results.append(mape)
                except ZeroDivisionError:
                    raise ValueError("No se puede calcular el MAPE cuando hay valores en el target iguales a cero")
           
            elif metric == "GRAPH":
                plt.scatter(target, predictions)
                plt.xlabel("Real")
                plt.ylabel("Predicción")
                plt.title("Gráfico de Dispersión: Valores reales VS Valores predichos")
                plt.show()

     # Clasificación         
    elif problem_type == "clasificacion":

        for metric in metrics:
            
            if metric == "ACCURACY":
                accuracy = accuracy_score(target, predictions)
                print(f"Accuracy: {accuracy}")
                results.append(accuracy)

            elif metric == "PRECISION":
                precision = precision_score(target, predictions, average = "macro")
                print(f"Precision: {precision}")
                results.append(precision)

            elif metric == "RECALL":
                recall = recall_score(target, predictions, average = "macro")
                print(f"Recall: {recall}")
                results.append(recall)

            elif metric == "CLASS_REPORT":
                print("Informe de Clasificación:")
                print(classification_report(target, predictions))

            elif metric == "MATRIX":
                print("Matriz de Confusión (Valores Absolutos):")
                print(confusion_matrix(target, predictions))

            elif metric == "MATRIX_RECALL":
                cm_normalized_recall = confusion_matrix(target, predictions, normalize = "true")
                disp = ConfusionMatrixDisplay(confusion_matrix = cm_normalized_recall)
                disp.plot()
                plt.title("Matriz de Confusión (Normalizado por Recall)")
                plt.show()

            elif metric == "MATRIX_PRED":
                cm_normalized_pred = confusion_matrix(target, predictions, normalize = "pred")
                disp = ConfusionMatrixDisplay(confusion_matrix = cm_normalized_pred)
                disp.plot()
                plt.title("Matriz de Confusión (Normalizado por Prediction)")
                plt.show()

            elif "PRECISION_" in metric:
                class_label = metric.split("_")[-1]
                try:
                    precision_class = precision_score(target, predictions, labels = [class_label])
                    print(f"Precisión para la clase {class_label}: {precision_class}")
                    results.append(precision_class)
                except ValueError:
                    raise ValueError(f"La clase {class_label} no está presente en las predicciones")
                
            elif "RECALL_" in metric:
                class_label = metric.split("_")[-1]
                try:
                    recall_class = recall_score(target, predictions, labels = [class_label])
                    print(f"Recall para la clase {class_label}: {recall_class}")
                    results.append(recall_class)
                except ValueError:
                    raise ValueError(f"La clase {class_label} no está presente en las predicciones")
                
    # Si no es regresión o clasificación
    else:
        raise ValueError("El tipo de problema debe ser de regresión o clasificación")

    return tuple(results)

###############################################################################

# Función | get_num_features_for_classification

def get_num_features_for_classification(dataframe, target_col = "", columns = None, pvalue = 0.05):

    """
    Selecciona las columnas numéricas de un DataFrame que pasan una prueba de ANOVA frente a la columna objetivo,
    según un nivel de significación especificado.

    Args:
    dataframe (pd.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo para la clasificación. Valor por defecto es una cadena vacía.
    columns (list): Lista de nombres de columnas a considerar. Por defecto es None, lo que considera todas las columnas numéricas.
    pvalue (float): Nivel de significación para la prueba de ANOVA. Por defecto es 0.05.

    Returns:
    list: Lista de nombres de columnas que cumplen con el criterio de significación especificado.
    """
    
    # Validar entradas
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe debe ser un DataFrame de pandas")
    if not isinstance(target_col, str):
        raise ValueError("target_col debe ser un string")
    if columns is not None and not all(isinstance(col, str) for col in columns):
        raise ValueError("columns debe ser una lista de strings")
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        raise ValueError("pvalue debe ser un número entre 0 y 1")
    
    # Si columns es None, igualar a las columnas numéricas del DataFrame
    if columns is None:
        columns = dataframe.select_dtypes(include = ['number']).columns.tolist()
    else:
        # Filtrar solo las columnas numéricas que están en la lista
        columns = [col for col in columns if dataframe[col].dtype in ['float64', 'int64']]
    
    # Asegurarse de que target_col esté en el DataFrame
    if target_col and target_col not in dataframe.columns:
        raise ValueError(f"{target_col} no está en el DataFrame")
    
    # Filtrar columnas que cumplen el test de ANOVA
    valid_columns = []
    if target_col:
        unique_classes = dataframe[target_col].unique()
        for col in columns:
            groups = [dataframe[dataframe[target_col] == cls][col].dropna() for cls in unique_classes]
            if len(groups) > 1 and all(len(group) > 0 for group in groups):
                f_val, p_val = f_oneway(*groups)
                if p_val < pvalue:
                    valid_columns.append(col)
    else:
        valid_columns = columns

    return valid_columns

###############################################################################

# Función | plot_num_features_for_classification

def plot_num_features_for_classification(dataframe, target_col = "", columns = None, pvalue = 0.05):

    """
    Genera pairplots para visualizar la relación entre las columnas numéricas de un DataFrame y una columna objetivo, 
    filtrando aquellas columnas que pasan una prueba de ANOVA según un nivel de significación especificado.

    Args:
    dataframe (pd.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo para la clasificación. Valor por defecto es una cadena vacía.
    columns (list): Lista de nombres de columnas a considerar. Por defecto es None, lo que considera todas las columnas numéricas.
    pvalue (float): Nivel de significación para la prueba de ANOVA. Por defecto es 0.05.

    Returns:
    list: Lista de nombres de columnas que cumplen con el criterio de significación especificado.
    """

    # Validar entradas
    if not isinstance(dataframe, pd.DataFrame):
        print("dataframe debe ser un DataFrame de pandas")
        return None
    if not isinstance(target_col, str):
        print("target_col debe ser un string")
        return None
    if columns is not None and not all(isinstance(col, str) for col in columns):
        print("columns debe ser una lista de strings")
        return None
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        print("Eso no es un pvalue válido")
        return None

    # Si columns es None, igualar a las columnas numéricas del DataFrame
    if columns is None:
        columns = dataframe.select_dtypes(include = ['number']).columns.tolist()
    else:
        # Verificar si todas las columnas en la lista existen en el DataFrame
        missing_columns = [col for col in columns if col not in dataframe.columns]
        if missing_columns:
            print("Esas columnas no existen en tu DataFrame:", missing_columns)
            return None
        # Filtrar solo las columnas numéricas que están en la lista
        columns = [col for col in columns if dataframe[col].dtype in ['float64', 'int64']]
        if len(columns) == 0:
            print("Debes elegir columnas numéricas")
            return None

    # Asegurarse de que target_col esté en el DataFrame
    if target_col and target_col not in dataframe.columns:
        print(f"Esa columna no existe en tu DataFrame")
        return None

    # Filtrar columnas que cumplen el test de ANOVA
    valid_columns = []
    if target_col:
        unique_classes = dataframe[target_col].unique()
        for col in columns:
            groups = [dataframe[dataframe[target_col] == cls][col].dropna() for cls in unique_classes]
            if len(groups) > 1 and all(len(group) > 0 for group in groups):
                f_val, p_val = f_oneway(*groups)
                if p_val < pvalue:
                    valid_columns.append(col)
    else:
        valid_columns = columns

    # Si no hay columnas válidas, retornar un mensaje
    if not valid_columns:
        print("Ninguna columna cumple con el pvalue indicado")
        return []

    # Excluir la columna objetivo de los resultados
    if target_col in valid_columns:
        valid_columns.remove(target_col)

    # Crear pairplots
    max_cols_per_plot = 5  # Máximo de columnas por plot
    if target_col:
        num_classes = len(dataframe[target_col].unique())
        for i in range(0, len(valid_columns), max_cols_per_plot):
            plot_columns = valid_columns[i:i+max_cols_per_plot]
            plot_columns.append(target_col)
            sns.pairplot(dataframe[plot_columns], hue = target_col)
            plt.show()
    else:
        # Sin target_col, dividir en grupos de max_cols_per_plot
        for i in range(0, len(valid_columns), max_cols_per_plot):
            plot_columns = valid_columns[i:i+max_cols_per_plot]
            sns.pairplot(dataframe[plot_columns])
            plt.show()
    
    return valid_columns

###############################################################################

# Función | get_cat_features_for_classification

def get_cat_features_for_classification(df, target_col, normalize = False, mi_threshold = 0):

    """
    Devuelve una lista con las columnas categóricas del DataFrame cuyo valor de mutual information 
    con "target_col" iguale o supere el valor de "mi_threshold". Si "normalize" es True, el valor de 
    mutual information se normaliza.

    Args:
    df (pd.DataFrame): DataFrame a analizar.
    target_col (str): Nombre de la columna objetivo del DataFrame.
    normalize (bool): Si queremos que el valor de mutual information se normalice. Por defecto es False.
    mi_threshold (float): Valor a superar al analizar "mutual information".

    Returns:
    list: Lista de las columnas que superan el umbral impuesto.
    """

    # Verificar que el DataFrame es de tipo pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'df' debe ser un DataFrame.")
        return None

    # Verificar que target_col es una columna del DataFrame
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el DataFrame.")
        return None

    # Verificar que mi_threshold es un float
    if not isinstance(mi_threshold, float):
        print("El argumento 'mi_threshold' debe ser un valor de tipo float.")
        return None

    # Si normalize es True, verificar que mi_threshold está entre 0 y 1
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("El argumento 'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return None

    # Obtener las columnas categóricas del DataFrame excluyendo la columna target
    result_class = classify_variables(df, 5, 10)
    columns = result_class[result_class["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()

    # Verificar que target_col es una columna categórica
    if target_col not in columns:
        print(f"La columna '{target_col}' debe ser de tipo categórico.")
        return None
    
    # Eliminar la target_col de la lista
    columns = [col for col in columns if col != target_col]

    # Calcular la información mutua
    mi_values = mutual_info_classif(df[columns], df[target_col], discrete_features = True)

    # Normalizar los valores de información mutua si normalize es True
    if normalize:
        total_mi = sum(mi_values)
        if total_mi == 0:
            print("La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi_values = mi_values / total_mi

    # Filtrar las columnas que cumplen con el umbral de información mutua
    selected_columns = [col for col, mi in zip(columns, mi_values) if mi >= mi_threshold]

    return selected_columns

###############################################################################

# Función | plot_cat_features_for_classification

def plot_cat_features_for_classification(df, target_col = "", columns = [], mi_threshold = 0.0, normalize = False):

    """
    Selecciona de la lista "columns" los valores que correspondan a columnas categóricas del DataFrame 
    cuyo valor de mutual information respecto de "target_col" supere el umbral puesto en "mi_threshold", 
    y para los valores seleccionados, pinta la distribución de etiquetas de cada valor respecto a los valores de 
    la columna "target_col". Si la lista "columns" está vacía, considera todas las columnas categóricas del DataFrame.

    Args:
    df (pd.DataFrame): DataFrame a analizar.
    target_col (str): Nombre de la columna objetivo del DataFrame.
    columns (list): Lista de columnas a comparar con target_col. Por defecto es una lista vacía.
    mi_threshold (float): Valor a superar al analizar "mutual information". Por defecto es 0.0.
    normalize (bool): Si queremos que el valor de mutual information se normalice. Por defecto es False.

    Returns:
    list: Lista de las columnas que superan el umbral impuesto.
    """

    # Verificar que el DataFrame es de tipo pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        print("El argumento 'dataframe' debe ser un pandas DataFrame.")
        return None

    # Verificar que target_col es una columna del DataFrame
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no está en el DataFrame.")
        return None

    # Verificar que mi_threshold es un float
    if not isinstance(mi_threshold, float):
        print("El argumento 'mi_threshold' debe ser un valor de tipo float.")
        return None

    # Si normalize es True, verificar que mi_threshold está entre 0 y 1
    if normalize and (mi_threshold < 0 or mi_threshold > 1):
        print("El argumento 'mi_threshold' debe estar entre 0 y 1 cuando 'normalize' es True.")
        return None

    # Si la lista está vacía, igualar columns a las variables categóricas del DataFrame reusando classify_variables
    if not columns:
        result_class = classify_variables(df, 5, 10)
        columns = result_class[result_class["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()

    # Por si no hay categóricas
    if not columns:
        print("No se encontraron columnas categóricas válidas en la lista proporcionada.")
        return None
    
    # Verificar que target_col es una columna categórica
    if target_col not in columns:
        print(f"La columna '{target_col}' debe ser de tipo categórico.")
        return None
    
    # Para no analizar target_col consigo misma
    columns = [col for col in columns if col != target_col]

    # Calcular la información mutua
    mi_values = mutual_info_classif(df[columns], df[target_col], discrete_features = True)

    # Normalizar los valores de información mutua si normalize es True
    if normalize:
        total_mi = sum(mi_values)
        if total_mi == 0:
            print("La suma de los valores de información mutua es 0, no se puede normalizar.")
            return None
        mi_values = mi_values / total_mi

    # Filtrar las columnas que cumplen con el umbral de información mutua
    selected_columns = [col for col, mi in zip(columns, mi_values) if mi >= mi_threshold]

    if not selected_columns:
        print("No se encontraron columnas que superen el umbral de información mutua.")
        return None

    # Pintar la distribución de etiquetas de cada columna seleccionada respecto a target_col
    for col in selected_columns:
        plt.figure(figsize = (10, 6))
        sns.countplot(data = df, x = col, hue = target_col)
        plt.title(f'Distribución de {col} respecto a {target_col}')
        plt.xlabel(col)
        plt.ylabel('Conteo')
        plt.legend(title=target_col)
        plt.show()

    return selected_columns

###############################################################################

# Función | super_selector

def super_selector(dataset, target_col = "", selectors = None, hard_voting = []):

    """
    Selecciona features de un DataFrame utilizando varios métodos y realiza un hard voting entre las listas seleccionadas.
    
    Args:
    dataset (pd.DataFrame): DataFrame con las features y el target.
    target_col (str): Columna objetivo en el dataset. Puede ser numérica o categórica.
    selectors (dict): Diccionario con los métodos de selección a utilizar. Puede contener las claves "KBest", "FromModel", "RFE" y "SFS".
    hard_voting (list): Lista de features para incluir en el hard voting.

    Returns:
    dict: Diccionario con las listas de features seleccionadas por cada método y una lista final por hard voting.
    """
    
    # Inicializar el diccionario de selectores si es None
    if selectors is None:
        selectors = {}

    # Separar features y target del dataset
    features = dataset.drop(columns = [target_col]) if target_col else dataset
    target = dataset[target_col] if target_col else None
    
    result = {}

    # Caso en que selectores esté vacío o sea None
    if target_col and target_col in dataset.columns:
        if not selectors:
            # Filtrar features que no son constantes y tienen más de una categoría
            filtered_features = [col for col in features.columns if
                                 (features[col].nunique() / len(features) < 0.9999) and
                                 (features[col].nunique() > 1)]
            result["all_features"] = filtered_features

    # Aplicación de selectores si no está vacío
    if selectors:
        if "KBest" in selectors:
            k = selectors["KBest"]
            selector = SelectKBest(score_func = f_classif, k = k)
            selector.fit(features, target)
            selected_features = features.columns[selector.get_support()].tolist()
            result["KBest"] = selected_features

        if "FromModel" in selectors:
            model, threshold_or_max = selectors["FromModel"]
            if isinstance(threshold_or_max, int):
                selector = SelectFromModel(model, max_features = threshold_or_max, threshold = -np.inf)
            else:
                selector = SelectFromModel(model, threshold = threshold_or_max)
            selector.fit(features, target)
            selected_features = features.columns[selector.get_support()].tolist()
            result["FromModel"] = selected_features

        if "RFE" in selectors:
            model, n_features, step = selectors["RFE"]
            selector = RFE(model, n_features_to_select = n_features, step = step)
            selector.fit(features, target)
            selected_features = features.columns[selector.get_support()].tolist()
            result["RFE"] = selected_features

        if "SFS" in selectors:
            model, k_features = selectors["SFS"]
            sfs = SequentialFeatureSelector(model, n_features_to_select = k_features, direction = "forward")
            sfs.fit(features, target)
            selected_features = features.columns[sfs.get_support()].tolist()
            result["SFS"] = selected_features

    # Hard Voting
    if hard_voting or selectors:
        voting_features = []
        if "hard_voting" not in result:
            voting_features = hard_voting.copy()
        for key in result:
            voting_features.extend(result[key])

        # Contar la frecuencia de cada feature seleccionada
        feature_counts = pd.Series(voting_features).value_counts()
        
        # Seleccionar las features que aparecen más de una vez
        hard_voting_result = feature_counts[feature_counts > 1].index.tolist()
        
        # Si no hay features repetidas, usar todas
        result["hard_voting"] = hard_voting_result if hard_voting_result else list(feature_counts.index)

    return result