from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from keras.models import load_model
import json   
model = load_model('apiModelo/modelo.h5')

@csrf_exempt
def predict_api(request):
    if request.method == 'POST':
        try:
            #recibimos los datos como json
            data = json.loads(request.body)
            
            # obtenemos los valores y los guardamos en una lista
            print("Datos recibidos:", data)
            data_array = list(data.values())
            
            # Realiza la predicción utilizando el modelo cargado
            prediction = model.predict([data_array])
            
            #imprimimos el resultado en consola
            print("Prediccion:", prediction[0][0])
            
            # Devuelve la predicción en la respuesta JSON
            return JsonResponse({'prediction': str(prediction[0][0])})  # Convierte el valor de predicción a un tipo nativo de Python
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)  # Devuelve un mensaje de error con información sobre la excepción
    else:
        return JsonResponse({'message': 'Sólo se admiten solicitudes POST para realizar predicciones.'}, status=400)
