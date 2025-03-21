import streamlit as st
from datetime import datetime
from notebook_functions import predire_matchs, equipes

st.set_page_config(page_title='FootForecast', page_icon='⚽️', layout='wide', initial_sidebar_state='expanded')

st.markdown('## ⚽️ FootForecast : *Prédictions des matchs de la Premier League*')
gab_url = 'https://ghj95.github.io/portfolio//'
st.markdown(
    f"<a href='{gab_url}' target='_blank' style='text-decoration: none; color: inherit;'>`Par : Gabriel Hardy-Joseph, Karim Ghandour, Louis-Matteo Creplet et Mohamed-Aziz Charfeddine`</a>",
    unsafe_allow_html=True,
)
st.markdown(
    "<span style='text-decoration: none;'><code style='color: inherit;'>Projet réalisé dans le cadre du cours TECH20704</code></span>",
    unsafe_allow_html=True,
)

def appinfo():
    with st.container(border=True):
        st.write("Notre application s'appuie sur un modèle de classification **Extreme Gradient Boosting (XGBoost)** entraîné sur une décennie de résultats historiques (2015-2025). Les prédictions intègrent la *forme récente des équipes*, l'*analyse statistique des matchs précédents* et l'*historique des confrontations directes* entre les clubs.")
        st.write('Les prédictions du modèle ne sont pas des certitudes.')

st.sidebar.header('Détails du match')
def input_features():
    domicile = st.sidebar.selectbox('Équipe à domicile :', (equipes), index=None, placeholder='Séléctionez une équipe...')
    exterieur = st.sidebar.selectbox("Équipe à l'extérieur :", (equipes), index=None, placeholder='Séléctionez une équipe...')
    match_date = st.sidebar.date_input('Date du match :', datetime.today())
    return domicile, exterieur, match_date

domicile, exterieur, match_date = input_features()

appinfo()
'---'

prediction = None
if st.sidebar.button('Prédire'):
    valid_input = True
    if domicile and exterieur not in equipes:
        st.error('Veuillez séléctionner deux équipes parmi celles dans la liste')
        valid_input = False
    if match_date.year > datetime.today().year + 1:
        st.error('Veuillez sélectionner une date moins éloignée')
        valid_input = False
    if match_date.year < datetime.today().year - 1:
        st.error('Veuillez sélectionner une date moins éloignée')
        valid_input = False
    if valid_input:
        with st.spinner('Chargement des prédictions...'):
            prediction = predire_matchs(domicile, exterieur, match_date)

if prediction == 0:
    st.success(f'Le modèle prédit une victoire de **{exterieur}** !', icon='🎯')
elif prediction == 1:
    st.success(f'Le modèle prédit une victoire de **{domicile}** !', icon='🎯')