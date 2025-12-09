# Contract Review Agent – Expatriados Argentina  
**Revisión automática de Assignment Letters y contratos Global Mobility usando IA**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-black?logo=openai)](https://platform.openai.com/)

---

## 📌 Descripción

Esta aplicación usa **IA + LangChain + OpenAI** para analizar *Assignment Letters* y contratos de expatriados.

El sistema:
- Lee el PDF del contrato
- Extrae y divide el contenido inteligentemente
- Analiza información clave de Global Mobility y Tax AR
- Genera respuestas claras y en español
- Produce un **Excel profesional** con formato mejorado

Diseñado especialmente para:
- Equipos de **Tax**
- **Global Mobility**
- **Payroll**
- Expatriados y consultores

---

## 🧠 Tecnologías utilizadas

- **Python 3.14**
- **Streamlit 1.40**
- **LangChain 0.3+**
- **OpenAI GPT-4o-mini**
- **PyPDFLoader**
- **Pandas + OpenPyXL** para el Excel

---

## 📂 Estructura del proyecto

contract-review-agent/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## 🔧 Instalación y ejecución local

### 

1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd contract-review-agent

2. Crear entorno virtual (Windows)
py -m venv .venv

3. Instalar dependencias
pip install -r requirements.txt

4. Ejecutar la app
streamlit run app.py

---

## 🔑 Configuración de la OpenAI API Key

La app te pedirá tu API Key en la barra lateral.

Puedes obtenerla desde:
https://platform.openai.com/account/api-keys

---

## 📊 Exportación a Excel PRO

La app genera un archivo:

revision_expatriado_PRO.xlsx

Características:

Columnas anchas

Sin word wrapping

Alineación superior

Datos ordenados por campo/pregunta

---

## 🤝 Contribuciones

Pull requests y issues son bienvenidos.
Si tenés ideas para nuevas features (riesgo PE, shadow payroll, clustering de cláusulas, etc.), ¡abrí un issue!

---

## 📜 Licencia

MIT License. Libre para uso personal o profesional.

---

## 👤 Autor

Lucas Slater

Proyecto creado con fines profesionales para revisión de contratos de expatriados en Argentina.