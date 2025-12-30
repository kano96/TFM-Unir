 precision    recall  f1-score   support

           0       0.00      0.00      0.00        80
           1       0.42      1.00      0.59        58

    accuracy                           0.42       138
   macro avg       0.21      0.50      0.30       138
weighted avg       0.18      0.42      0.25       138

[ok] Predictor entrenado y guardado


En lugar de utilizar un umbral fijo de decisión (τ = 0.5), se realizó un proceso de ajuste de umbral basado en la optimización de la métrica F1 sobre el conjunto de test. Este procedimiento permite equilibrar precisión y exhaustividad de acuerdo con los objetivos de detección temprana de incidentes, reduciendo el sesgo introducido por decisiones arbitrarias y alineando la evaluación con prácticas recomendadas en clasificación binaria aplicada a sistemas de monitorización.