# app.py – versión final estable (diciembre 2025)
import os
import streamlit as st
import pandas as pd
from io import BytesIO
from openpyxl.styles import Alignment

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# ===============================
# API KEY
# ===============================
if "OPENAI_API_KEY" not in os.environ:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("Key cargada!")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

st.title("Contract Review Agent – Expatriados Argentina")


# ===============================
# CARGA PDF
# ===============================
uploaded_file = st.file_uploader("Subí el Assignment Letter (PDF)", type=["pdf"])

if uploaded_file:

    # Guardar temporalmente
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Splitter actualizado (DEC 2025)
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)


    # ===============================
    # PROMPT
    # ===============================
    template = """Sos experto en Global Mobility y Tax Argentina.
    Revisá SOLO el contrato y respondé claro y en español.

    Contrato:
    {context}

    Pregunta: {question}
    Respuesta:
    """

    prompt = PromptTemplate.from_template(template)


    # ===============================
    # PREGUNTAS
    # ===============================
    preguntas = [
        "Fecha exacta de inicio y fin",
        "País home y país host",
        "Tipo de política (tax equalization, protection, etc.)",
        "Cláusula de repatriación anticipada",
        "Riesgo de Permanent Establishment en Argentina (explicar)",
        "Obligación de shadow payroll o inscripción AFIP",
        "Beneficios de vivienda, educación y mudanza"
    ]

    resultados = {}
    bar = st.progress(0)

    for i, q in enumerate(preguntas):

        # Usamos primer chunk grande (sobra contexto)
        context = "\n\n".join([c.page_content for c in chunks[:4]])

        answer = llm.invoke(prompt.format(context=context, question=q))
        resultados[q] = answer.content

        bar.progress((i + 1) / len(preguntas))


    # ===============================
    # RESULTADOS EN PANTALLA
    # ===============================
    st.balloons()
    st.success("¡Listo, rey!")

    for q, r in resultados.items():
        st.write(f"**{q}**")
        st.write(r)
        st.divider()


    # ===============================
    # EXPORTAR A EXCEL (PRO)
    # ===============================
    df = pd.DataFrame(list(resultados.items()), columns=["Campo", "Resultado"])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Revisión Contrato')

        workbook  = writer.book
        worksheet = writer.sheets['Revisión Contrato']

        # Columnas anchas
        worksheet.column_dimensions['A'].width = 40
        worksheet.column_dimensions['B'].width = 80

        # Con wrap + alineación vertical
        align = Alignment(wrap_text=True, vertical='top')
        for row in worksheet.iter_rows(min_row=1, max_row=worksheet.max_row):
            for cell in row:
                cell.alignment = align

    output.seek(0)

    st.download_button(
        label="📊 Descargar Excel",
        data=output,
        file_name="revision_expatriado_PRO.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
