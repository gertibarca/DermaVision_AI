# DermaVision AI

Projekt edukativ në Python për klasifikimin e lezioneve dermatoskopike në shtatë
kategori të dataset-it HAM10000. Aplikacioni kombinon ResNet18, test-time
augmentation, Grad-CAM, vlerësimin e pasigurisë, kontrollin teknik të fotografisë,
SQLite, grafikë interaktivë dhe raport PDF.

> Ky projekt nuk është pajisje mjekësore dhe nuk jep diagnozë. Çdo lezion i ri,
> që ndryshon, kruhet, dhemb ose gjakoset duhet kontrolluar nga dermatologu.

## Nisja në PyCharm

1. Hape folderin `DermaVision_AI` si projekt.
2. Zgjidh interpreterin `.venv/Scripts/python.exe`.
3. Ekzekuto konfigurimin `DermaVision AI` ose komandën:

```powershell
.\run_app.ps1
```

Aplikacioni hapet në `http://localhost:8501`.

## Arkitektura

```text
app.py                      ndërfaqja dhe faqet e Streamlit
dermavision/config.py       klasat, pragjet dhe metadata e modelit
dermavision/model.py        ResNet18, TTA, pasiguri dhe Grad-CAM
dermavision/quality.py      kontrolli teknik dhe heuristika e imazhit
dermavision/database.py     persistence me SQLite
dermavision/report.py       raport PDF
dermavision/ui.py           stili dhe komponentët vizualë
tests/                      teste automatike
```

## Modeli

- Arkitektura: ResNet18 me transfer learning.
- Dataset: HAM10000, shtatë kategori.
- Ndarja e testit: sipas `lesion_id`, jo rastësisht sipas fotografisë.
- Accuracy 7-klasësh në test: 74.9%.
- Rregulli i sinjalit: `P(akiec) + P(bcc) + P(mel) >= 0.15`.
- Recall i sinjalit në test: 92.0%; precision: 43.0%.
- Licenca e peshave/dataset-it: CC BY-NC 4.0, vetëm përdorim jo-komercial.

Burimi i peshave:
`miloszkrawczyk/skin-lesion-classifier` në Hugging Face.

## Publikimi online

Projekti përfshin `requirements.txt`, `.streamlit/config.toml` dhe `Dockerfile`.
Mund të publikohet në Streamlit Community Cloud ose në një host që mbështet
Docker. Për Streamlit Cloud, modeli shkarkohet gjatë build-it përmes
`download_model.py`.

