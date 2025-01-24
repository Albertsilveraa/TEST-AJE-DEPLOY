import pandas as pd
import re
import os
import uuid

# Carpeta donde se encuentran los archivos CSV
folder_path = '../data'

# Crear una lista de archivos CSV en la carpeta
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Función para limpiar y procesar cada archivo
def process_file(file_path, file_id):
    # Leer el archivo CSV
    df = pd.read_csv(file_path)

    # Filtrar solo productos que tengan la subcategoría 'Coffee, Tea & Beverages'
    df = df[df['sub_category'].str.lower().str.strip() == 'coffee, tea & beverages']

    # Filtrar solo productos que terminan en 'ml' o 'l' en la columna 'name'
    df = df[df['name'].str.lower().str.endswith(('ml', 'l'))]

    # Limpiar y convertir precios
    df['discount_price'] = (
        df['discount_price']
        .astype(str)
        .str.replace('₹', '', regex=True)
        .str.replace(',', '', regex=True)
        .astype(float)
        .fillna(0)  # Rellenar valores nulos con 0
    )
    df['actual_price'] = (
        df['actual_price']
        .astype(str)
        .str.replace('₹', '', regex=True)
        .str.replace(',', '', regex=True)
        .astype(float)
        .fillna(0)  # Rellenar valores nulos con 0
    )

    # Calcular porcentaje de descuento (rellenar 0 si actual_price es 0)
    df['discount_percent'] = (
        (df['actual_price'] - df['discount_price']) / df['actual_price'] * 100
    ).fillna(0)

    # Limpiar columnas de texto
    def clean_text(text):
        if pd.isnull(text):
            return ''
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Quitar caracteres no alfanuméricos
        return text

    df['name'] = df['name'].apply(clean_text)
    df['sub_category'] = df['sub_category'].apply(clean_text)
    
    # Si tienes más columnas de texto, puedes limpiarlas de manera similar
    # Por ejemplo:
    # df['otra_columna'] = df['otra_columna'].apply(clean_text)

    # Limpiar y convertir 'no_of_ratings' a entero
    if 'no_of_ratings' in df.columns:
        df['no_of_ratings'] = (
            df['no_of_ratings']
            .astype(str)
            .str.replace(r'[^0-9]', '', regex=True)  # Eliminar cualquier carácter no numérico
            .replace('', '0')  # Reemplazar cadenas vacías por '0'
            .astype(int)
        )
    else:
        # Si la columna no existe, crearla con valor 0
        df['no_of_ratings'] = 0

    # Opcional: Limpiar 'ratings' si es necesario
    if 'ratings' in df.columns:
        # Supongamos que 'ratings' es un número flotante, si tiene caracteres no deseados
        df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce').fillna(0.0)
    else:
        # Si la columna no existe, crearla con valor 0.0
        df['ratings'] = 0.0

    # Añadir índice único a cada fila usando UUID
    df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Seleccionar características relevantes
    columns_to_keep = [
        'id', 'name', 'sub_category', 'ratings', 'no_of_ratings',
        'discount_price', 'actual_price', 'discount_percent'
    ]

    # Mantener solo las columnas relevantes
    df_relevant = df[columns_to_keep]

    return df_relevant

# Crear una lista para almacenar todos los DataFrames procesados
all_processed_dfs = []

# Procesar todos los archivos CSV
for i, file in enumerate(csv_files):
    file_path = os.path.join(folder_path, file)
    
    # Procesar el archivo CSV
    processed_df = process_file(file_path, file_id=i)
    
    # Agregar el DataFrame procesado a la lista
    all_processed_dfs.append(processed_df)

# Concatenar todos los DataFrames procesados en uno solo
combined_df = pd.concat(all_processed_dfs, ignore_index=True)

# Guardar el archivo combinado en un solo CSV
combined_df.to_csv('combined_processed_data.csv', index=False)

print("Todos los archivos han sido procesados y combinados.")
