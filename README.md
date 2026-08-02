# Pipeline Base — Proyecto Integrador (Deep Learning)

Entrega 1 del proyecto integrador: infraestructura técnica inicial y ciclo de vida básico de un modelo de clasificación en PyTorch.

## Dataset

Se utilizó **Iris** (clásico, vía `sklearn.datasets.load_iris`) como dataset de referencia: 150 muestras, 4 features numéricas, 3 clases balanceadas. Se eligió por ser liviano y no depender de descargas externas, lo cual permite validar el pipeline de forma rápida y reproducible antes de escalar a un dataset más complejo en las próximas entregas.

- Split: 80% train / 20% validación, **estratificado** por clase.
- Normalización: `StandardScaler` ajustado únicamente con el set de train (para evitar data leakage hacia validación).

## Arquitectura

`MLPClassifier`, implementado con `nn.Sequential`:
Input(4) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(3)

No se aplica `Softmax` en la salida porque `nn.CrossEntropyLoss` ya combina `LogSoftmax` + `NLLLoss` internamente.

## Configuración del experimento (checkpoint)

| Parámetro | Valor |
|---|---|
| Versión de PyTorch | `2.13.0` |
| Optimizador | Adam |
| Learning rate | `0.01` |
| Batch size | 16 |
| Épocas | 50 |
| Función de pérdida | `CrossEntropyLoss` |
| Semilla | 42 |

Se eligió un `learning_rate` de `0.01` porque con Adam sobre un dataset pequeño y bien normalizado como Iris permite converger en pocas épocas sin generar oscilaciones grandes en la pérdida; valores más altos (`0.1`) mostraron inestabilidad en pruebas exploratorias, y valores más bajos (`0.001`) requerían muchas más épocas para estabilizarse.

## Resultados e interpretación de la curva de pérdida

La pérdida de entrenamiento bajó de forma sostenida durante las 50 épocas, pasando de ~0.81 a ~0.007, lo cual confirma que `backward()` + `optimizer.step()` están actualizando los pesos correctamente y que el modelo efectivamente está aprendiendo. La pérdida de validación baja rápido en las primeras 10 épocas (de 0.48 a ~0.09) y luego se estabiliza; a partir de la época ~15-20 empieza a subir levemente mientras el train loss sigue bajando casi a cero. Esa apertura entre ambas curvas es la señal clásica de **overfitting leve**: el modelo sigue memorizando el set de entrenamiento mientras el desempeño en validación ya no mejora (accuracy de validación se mantiene estable alrededor de 0.93–0.97).

## Errores comunes evitados en la implementación

- **`zero_grad()`**: se llama explícitamente en cada iteración del batch, antes de `loss.backward()`, para evitar la acumulación de gradientes entre pasos.
- **Inconsistencia de dispositivos**: tanto el modelo como cada batch de datos se mueven al mismo `device` con `.to(device)` antes de cualquier operación, evitando errores de tipo `RuntimeError`.
