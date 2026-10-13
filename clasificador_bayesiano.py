"""
Ejercicio 6 - Clasificador Bayesiano con Python
Detección de noticias falsas (fake) o reales (real) con Naive Bayes.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Datos
datos = {
    "texto": [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños",
    ],
    "etiqueta": ["real", "fake", "real", "fake", "real",
                 "fake", "real", "fake", "real", "fake"],
}
df = pd.DataFrame(datos)

# 2. División entrenamiento / prueba (70% / 30%, manteniendo la proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["etiqueta"], test_size=0.3, random_state=42, stratify=df["etiqueta"]
)

# 3. Vectorización: cada noticia se convierte en un conteo de palabras (bolsa de palabras)
vectorizador = CountVectorizer(lowercase=True)
X_train_vec = vectorizador.fit_transform(X_train)
X_test_vec = vectorizador.transform(X_test)

# 4. Entrenamiento del modelo Naive Bayes multinomial
modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)

# 5. Evaluación
y_pred = modelo.predict(X_test_vec)
etiquetas = ["fake", "real"]

print("=== RESULTADOS DEL MODELO ===")
print("Noticias de prueba:")
for texto, real, pred in zip(X_test, y_test, y_pred):
    print(f"  - {texto!r} -> real: {real} | predicha: {pred}")

print(f"\nPrecisión total (accuracy): {accuracy_score(y_test, y_pred):.2f}")

print("\nMatriz de confusión (filas = real, columnas = predicho):")
cm = confusion_matrix(y_test, y_pred, labels=etiquetas)
print(pd.DataFrame(cm, index=[f"real_{e}" for e in etiquetas],
                   columns=[f"pred_{e}" for e in etiquetas]))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, labels=etiquetas, zero_division=0))

# 6. Clasificación de nuevos mensajes (modelo reentrenado con todos los datos)
modelo_final = MultinomialNB()
vect_final = CountVectorizer(lowercase=True)
modelo_final.fit(vect_final.fit_transform(df["texto"]), df["etiqueta"])

nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos",
]

print("=== CLASIFICACIÓN DE NUEVAS NOTICIAS ===")
X_nuevas = vect_final.transform(nuevas_noticias)
predicciones = modelo_final.predict(X_nuevas)
probabilidades = modelo_final.predict_proba(X_nuevas)
for noticia, pred, prob in zip(nuevas_noticias, predicciones, probabilidades):
    detalle = ", ".join(f"{c}: {p:.2f}" for c, p in zip(modelo_final.classes_, prob))
    print(f"  - {noticia!r} -> {pred.upper()} ({detalle})")