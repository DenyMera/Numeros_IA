from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow import keras
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from scipy.ndimage import center_of_mass

app = FastAPI()

# Configurar CORS para permitir peticiones desde tu frontend
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Cambia a ["*"] para permitir todos (solo en desarrollo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo entrenado
modelo = keras.models.load_model('numeros_conv.h5')  # Asegúrate de usar el modelo correcto

# Función para centrar la imagen
def centrar_imagen(img_array):
    """
    Mueve el dígito al centro de la imagen utilizando el centro de masa.
    """
    filas, columnas = img_array.shape
    centro_actual = center_of_mass(img_array)
    centro_deseado = (filas // 2, columnas // 2)
    desplazamiento = np.array(centro_deseado) - np.array(centro_actual)
    
    # Crear una imagen centrada desplazando el contenido
    imagen_centrada = np.roll(img_array, int(desplazamiento[0]), axis=0)
    imagen_centrada = np.roll(imagen_centrada, int(desplazamiento[1]), axis=1)
    return imagen_centrada

@app.post("/predecir/")
async def predecir(request: dict):
    try:
        img_data = request.get("image")
        if not img_data:
            raise HTTPException(status_code=400, detail="No se proporcionó ninguna imagen.")

        # Extraer solo la parte base64 (quitar "data:image/png;base64,")
        img_data = img_data.split(",")[1]

        # Decodificar base64 y abrir imagen
        image = Image.open(BytesIO(base64.b64decode(img_data))).convert("L")
        image = image.resize((28, 28))  # Escalar inicialmente para centrado
        img_array = np.array(image, dtype=np.float32)

        # Centrar la imagen
        img_array = centrar_imagen(img_array)

        # Redimensionar al tamaño esperado y normalizar
        img_array = Image.fromarray(img_array).resize((28, 28), Image.BICUBIC)
        img_array = np.array(img_array) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predecir
        prediccion = modelo.predict(img_array)
        etiqueta = int(np.argmax(prediccion))

        return JSONResponse(content={"prediccion": etiqueta})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

