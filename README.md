# Análisis Predictivo de Ingresos de Freelancers en Upwork

## MIT Summer Program - Uruguay 2026
### Proyecto de Machine Learning

Este repositorio contiene un análisis predictivo diseñado para estimar los ingresos potenciales de freelancers en la plataforma **Upwork**, utilizando técnicas de Machine Learning y Análisis Exploratorio de Datos (EDA), culminando en una interfaz interactiva para predicción en tiempo real.

---

## 📄 Descripción del Proyecto — ABT (Action, Background, Thesis)

### Action (Acción)
Construir un modelo predictivo que ayude a los freelancers a navegar el competitivo ecosistema de Upwork, estimando su tarifa horaria potencial y cuantificando oportunidades de trabajo para empoderarlos a tomar decisiones informadas.

### Background (Contexto)
En la economía globalizada actual, el mercado freelance está en auge, con miles de profesionales buscando trabajo en plataformas como Upwork. Sin embargo, este crecimiento ha generado desafíos significativos: los freelancers frecuentemente luchan para establecer tarifas adecuadas, entender la demanda real de sus habilidades y evaluar la competitividad de los avisos de trabajo. La falta de orientación basada en datos lleva al sub-precio de servicios, esfuerzo desperdiciado en ofertas altamente competitivas y una sensación general de incertidumbre.

### Thesis (Hipótesis)
**Hipótesis:** Al desarrollar un modelo predictivo de estimación de tarifa horaria y un sistema de puntuación de oportunidades, podemos proveerle a los freelancers los insights cruciales para superar estos obstáculos. El modelo aprovecha datos históricos de trabajos para sugerir tarifas horarias competitivas basadas en skills, experiencia y país del cliente, transformando la búsqueda de trabajo freelance de un juego de adivinanzas en un proceso estratégico y basado en datos.

---

## 👥 Autores
*   **Abril Ocampo**
*   **Camilo Castellano**
*   **Gerardo Gonzalez**

---

## 📊 Dataset

El proyecto utiliza el dataset **"Upwork Freelancers"** (disponible en [Kaggle](https://www.kaggle.com/datasets)) con aproximadamente **60,000 registros** de publicaciones de trabajo de la plataforma Upwork.

### Variables principales del dataset

| Variable | Descripción |
|---|---|
| `Hourly_Rate` | Tarifa horaria ofrecida (string, e.g., `"$20-$40"`) |
| `Start_rate` / `End_rate` | Límites inferior y superior de la tarifa |
| `Search_Keyword` | Categoría de búsqueda del trabajo |
| `EX_level_demand` | Nivel de experiencia requerido (`Entry level`, `Intermediate`, `Expert`) |
| `Client_Country` | País del cliente |
| `Category_1` a `Category_3` | Skills/categorías requeridas |
| `Applicants_Num` | Número de postulantes |
| `Connects_Num` | Número de "connects" requeridos |
| `Feedback_Num` | Número de feedbacks del cliente |
| `Freelancers_Num` | Número de freelancers contratados por el cliente |
| `Spent($)` | Total gastado por el cliente en la plataforma |
| `Payment_Situation` | Estado de verificación del pago |
| `Payment_Type` | Tipo de pago (`Hourly` / `Fixed`) |

---

## 🛠️ Tecnologías y Librerías

El análisis se ha desarrollado en **Python** dentro de un **Jupyter Notebook** (compatible con Google Colab y VS Code). Las principales librerías son:

| Librería | Uso |
|---|---|
| `pandas` & `numpy` | Manipulación y limpieza de datos |
| `matplotlib` & `seaborn` | Visualización de datos y EDA |
| `scikit-learn` | Preprocesamiento, modelado y evaluación de ML |
| `ipywidgets` | Interfaz interactiva de predicción |
| `IPython.display` | Renderizado de outputs en el notebook |

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd Posting-Job-on-Upwork
```

### 2. Instalar dependencias
Se recomienda usar un entorno virtual:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets openpyxl
```

### 3. Preparar el dataset
Descarga el archivo `Final_Upwork_Dataset.csv` desde Kaggle y colócalo en tu Google Drive (ruta: `MyDrive/Final_Upwork_Dataset.csv`) o ajusta la ruta en la celda de carga del notebook.

### 4. Ejecutar el Notebook
Abre `FINAL_Upwork_freelancer_Project_training.ipynb` en **Google Colab**, **Jupyter Lab** o **VS Code**. Ejecuta las celdas en orden secuencial.

> **Nota:** El notebook requiere el montaje de Google Drive (`drive.mount('/content/drive')`) si se ejecuta en Google Colab.

---

## 📈 Estructura del Análisis

### Paso 0 — Carga y Limpieza de Datos
- Carga del dataset desde CSV.
- **Análisis de valores faltantes (NA):** Se eliminan columnas con más del 50% de NAs, conservando manualmente `End_rate` y `Hourly_Rate` por ser necesarias para la variable objetivo.
- **Filtrado por tipo de pago:** Se conservan únicamente registros con `Payment_Type = Hourly`.
- **Eliminación de columnas URL** que no aportan información predictiva.
- **Creación de la variable objetivo `Hourly_rate`:** Se parsea el string de `Hourly_Rate` (e.g., `"$20-$40"`) calculando el promedio entre los extremos.
- **Eliminación de outliers con método IQR:** Se calcula el rango aceptable `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]` y se descartan los valores fuera de ese rango.

### Paso 1 — Análisis Exploratorio de Datos (EDA)
- Estadísticas descriptivas de `Hourly_rate` (media, mediana, desviación estándar, cuartiles).
- Tarifas promedio por `Search_Keyword` (top 10 keywords).
- Tarifas promedio por `EX_level_demand`.
- Correlaciones numéricas con la variable objetivo.

### Data Visualization
Se generan las siguientes visualizaciones clave:
- **Histograma + KDE** de la distribución de `Hourly_rate` (con líneas de media y mediana).
- **Boxplot** de `Hourly_rate`.
- **Barplot horizontal** — Tarifa promedio por hora para los **Top 30 skills** (más frecuentes).
- **Barplot horizontal** — Tarifa promedio por hora para los **Top 10 skills**.
- **Barplots (2x2)**:
  - Tarifa promedio por nivel de experiencia.
  - Tarifa promedio por Search Keyword.
  - Tarifa promedio por Top 10 países del cliente.
  - Tarifa promedio por Top 15 skills.
- **Violin Plot** — Distribución de tarifas por nivel de experiencia (con cuartiles internos).
- **KDE superpuesto** — Distribución de tarifas por keyword (con áreas rellenas).
- **Heatmap de correlaciones** entre features numéricas y `Hourly_rate`.

### Paso 2 — Feature Engineering y Preparación de Datos

#### Variables categóricas codificadas con One-Hot Encoding
| Variable | Tipo | Detalle |
|---|---|---|
| `Search_Keyword` | One-Hot | Cada keyword como columna binaria |
| `EX_level_demand` | One-Hot | `Entry level`, `Intermediate`, `Expert` |
| `Client_Country` | One-Hot | Top 15 países + `Other` |
| `Payment_Situation` | Binaria | `payment_verified` (1 = verificado) |

#### Variables de skills
- Se extraen skills de `Category_1`, `Category_2` y `Category_3`.
- Se crea `skills_list` (lista de skills por trabajo) y `num_skills` (conteo).
- Se generan **dummies para los Top 30 skills** más frecuentes.

#### Variables numéricas
| Feature | Descripción |
|---|---|
| `num_skills` | Número de skills requeridas |
| `Applicants_Num` | Número de postulantes (normalizado) |
| `Connects_Num` | Número de connects requeridos |
| `Feedback_Num` | Número de feedbacks del cliente |
| `Freelancers_Num` | Freelancers contratados por el cliente |
| `Spent($)` | Total gastado por el cliente |
| `log_spent` | Logaritmo de `Spent($)` (reduce asimetría) |
| `applicants_numeric` | Applicants normalizado (maneja strings tipo "Less than 5") |

**Total de features:** combinación de variables numéricas + keywords + experiencia + países + skills (~70+ columnas).

### Paso 3 — Entrenamiento de Modelos

El dataset se divide **80% train / 20% test** (`random_state=42`). Los modelos lineales usan `StandardScaler`; los modelos de ensemble usan los datos sin escalar.

#### Modelos entrenados

| Modelo | Hiperparámetros clave |
|---|---|
| **Linear Regression** | Default |
| **Ridge** | `alpha=0.1` |
| **Lasso** | `alpha=0.01`, `max_iter=10000` |
| **ElasticNet** | `alpha=0.1`, `l1_ratio=0.5`, `max_iter=10000` |
| **Random Forest** | `n_estimators=200`, `max_depth=20`, `min_samples_leaf=3`, `max_features='sqrt'` |
| **Gradient Boosting** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8` |

#### Métricas de evaluación
- **MAE** (Mean Absolute Error) — Error promedio en dólares.
- **RMSE** (Root Mean Squared Error) — Penaliza errores grandes.
- **MSE** (Mean Squared Error).
- **R²** — Porcentaje de varianza explicada.
- **CV R²** — R² de validación cruzada (3-fold) para detectar overfitting.

#### Rendimiento esperado
Los modelos de ensemble (Gradient Boosting y Random Forest) obtienen los mejores resultados con un **R² de ~0.18-0.20**.

> **Nota sobre el R²:** Un R² de ~0.20 es un resultado sólido para este problema porque:
> 1. Las tarifas dependen de negociación subjetiva y urgencia no capturada en los datos.
> 2. No se dispone de datos del perfil del freelancer (rating, años de experiencia reales).
> 3. El dataset representa la oferta del cliente (budget), no el contrato final acordado.
> El modelo identifica correctamente tendencias generales (qué skills valen más, qué países pagan mejor), suficiente para una **recomendación base** sólida.

### Paso 4 — Análisis de Importancia de Features
- Se visualizan los **Top 20 features más importantes** usando `feature_importances_` de Gradient Boosting (o Random Forest).
- Se analizan los **coeficientes de la Regresión Lineal** para entender la dirección del efecto (positivo ↑ / negativo ↓).

### Paso 5 — Sistema de Predicción para Freelancers

Se entrena el modelo final (Gradient Boosting o Random Forest) con **todos los datos disponibles** (sin split) para maximizar la información en producción.

#### Función `predecir_freelancer()`
```python
predecir_freelancer(
    skills_usuario=['Python', 'Data Analysis', 'SQL'],
    keyword='Data science',
    exp_level='Entry Level',
    client_country='United States'
)
```
**Retorna:**
```python
{
    'tarifa_estimada': 35.50,           # USD/hr
    'rango_sugerido': (30.17, 40.82),   # ±15% del estimado
    'trabajos_compatibles': 12430,      # trabajos con al menos 1 skill del usuario
    'total_trabajos': 45000             # total en la base de datos
}
```

### Paso 6 — Interfaz Interactiva (`ipywidgets`)
Se construye una interfaz interactiva completa dentro del notebook:
- **Dropdown** para seleccionar `Search_Keyword`.
- **SelectMultiple** para skills (se actualiza dinámicamente según el keyword seleccionado, mostrando los 15 skills más relevantes para esa categoría).
- **Dropdown** para nivel de experiencia (`Entry level`, `Intermediate`, `Expert`).
- **Dropdown** para país del cliente (Top 15 + Other).
- **Botón "Predict Hourly Rate"** (habilitado solo cuando hay al menos 1 skill seleccionado).
- **Tabla de resultados** con: tarifa estimada, rango sugerido (mín/máx) y trabajos compatibles en la base de datos.

---

## 🗂️ Estructura del Repositorio

```
Posting-Job-on-Upwork/
├── FINAL_Upwork_freelancer_Project_training.ipynb   # Notebook principal
└── README.md                                         # Este archivo
```

---

## 📌 Principales Hallazgos (Insights)

- **Nivel de experiencia:** Los roles `Expert` tienen tarifas medias significativamente más altas que `Intermediate` y `Entry level`.
- **País del cliente:** Clientes de **Estados Unidos** tienden a ofrecer las tarifas más altas, consistente con diferencias en costo de vida y mercado laboral global.
- **Keywords:** Categorías como `Data science` y `3D` se asocian con tarifas más elevadas comparadas con otras categorías.
- **Skills premium:** Skills técnicas especializadas (e.g., Machine Learning, Python, 3D Modeling) se correlacionan con tarifas horarias más altas.
- **Competencia:** Un mayor número de postulantes (`Applicants_Num`) no necesariamente reduce la tarifa ofrecida por el cliente.

---

*MIT Summer Program — Uruguay 2026*
