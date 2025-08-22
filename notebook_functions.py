# Important à lire!!
# Le fichier présent comporte uniquement des fonctions qui ont déjà été crées et entièrement expliquées au sein du jupyter notebook epl_predictor.ipynb, ce qui explique son absence de commentaires
# Il sert uniquement à recopier les fonctions au sein d'un fichier .py dans le but de les rendre disponibles pour l'application Streamlit contenue dans le fichier app.py
# Une explication étendue de l'entièreté du fichier présent avec des commentaires ajoutés est disponible dans le jupyter notebook epl_predictor.ipynb 

import pandas as pd
import pickle

matchs = pd.read_csv('epl_2000-2025.csv')

matchs.rename(columns={'HomeTeam':'Domicile',
                        'AwayTeam':'Exterieur',
                        'Season':'Saison',
                        'FTR':'Cible',
                        'FTHG':'Buts domicile', 
                        'FTAG':'Buts exterieur',
                        }, inplace=True)

for i in range(len(matchs)):                                     
    row = matchs['Date'][i]                                   
    if row[4:5] == '-':
        jour = row[2:4]                                           
        mois = row[5:7]                                           
        annee = row[-2:]                                          
        matchs.loc[i, 'Date'] = f'{jour}/{mois}/20{annee}'
    elif row[-3:-2] == '/':
        date = row[:-3]                                           
        annee = row[-2:]                                         
        matchs.loc[i, 'Date'] = f'{date}/20{annee}'      

matchs['Date'] = pd.to_datetime(matchs['Date'], dayfirst=True)

equipes = pd.concat([matchs['Domicile'], matchs['Exterieur']]).unique()   
equipes_df = pd.DataFrame({'Equipe':equipes})
equipes_df['CodeEquipe'] = equipes_df['Equipe'].astype('category').cat.codes
code_equipe = dict(zip(equipes_df['Equipe'], equipes_df['CodeEquipe']))

matchs['DomicileCode'] = matchs['Domicile'].map(code_equipe)                            
matchs['ExterieurCode'] = matchs['Exterieur'].map(code_equipe)

matchs['JoursSemaine'] = matchs['Date'].dt.day_of_week
matchs['Mois'] = matchs['Date'].dt.month
matchs['AnneeSaison'] = matchs['Saison'].str.split('-').str[0].astype(int)
matchs['JoursSaison'] = matchs.apply(lambda x: (x['Date'] - pd.Timestamp(year=x['AnneeSaison'], month=8, day=5)).days, axis=1)      
matchs['SemaineSaison'] = matchs['JoursSaison'] // 7 + 1

def stats_equipe(matchs):                      
    matchs = matchs.sort_values('Date')              
    matchs['DomicileForme'] = 0                         
    matchs['ExterieurForme'] = 0

    colonnes_buts = [
    'DomicileAvgButsMarques_Home', 'DomicileAvgButsEncaisses_Home',     
    'DomicileAvgButsMarques_Away', 'DomicileAvgButsEncaisses_Away',
    'ExterieurAvgButsMarques_Home', 'ExterieurAvgButsEncaisses_Home',
    'ExterieurAvgButsMarques_Away', 'ExterieurAvgButsEncaisses_Away']

    for col in colonnes_buts:
        matchs[col] = 0.0

    
    for i in range(len(matchs)):                         
        match_actuel = matchs.iloc[i]                   
        equipe_domicile = match_actuel['DomicileCode']  
        equipe_exterieur = match_actuel['ExterieurCode']
        date_match = match_actuel['Date']               
        saison_match = match_actuel['AnneeSaison']      

        prec_domicile_5 = matchs[(matchs['Date'] < date_match) &
                               ((matchs['DomicileCode'] == equipe_domicile) | (matchs['ExterieurCode'] == equipe_domicile))]
        prec_domicile_5 = prec_domicile_5.sort_values('Date', ascending=False).head(5)
        
        prec_exterieur_5 = matchs[(matchs['Date'] < date_match) & 
                                ((matchs['DomicileCode'] == equipe_exterieur) | (matchs['ExterieurCode'] == equipe_exterieur))]
        prec_exterieur_5 = prec_exterieur_5.sort_values('Date', ascending=False).head(5)

        prec_domicile_saison = matchs[(matchs['Date'] < date_match) & (matchs['AnneeSaison'] == saison_match) &
                               ((matchs['DomicileCode'] == equipe_domicile) | (matchs['ExterieurCode'] == equipe_domicile))]
        
        prec_exterieur_saison = matchs[(matchs['Date'] < date_match) & (matchs['AnneeSaison'] == saison_match) & 
                                ((matchs['DomicileCode'] == equipe_exterieur) | (matchs['ExterieurCode'] == equipe_exterieur))]

        points_domicile = 0
        for j in range(len(prec_domicile_5)):             
            match_prec = prec_domicile_5.iloc[j]          
            if match_prec['DomicileCode'] == equipe_domicile and match_prec['Cible'] == 1:
                points_domicile += 3
            elif match_prec['ExterieurCode'] == equipe_domicile and match_prec['Cible'] == 0:
                points_domicile += 3
            else:
                points_domicile += 1

        points_exterieur = 0
        for l in range(len(prec_exterieur_5)):          
            match_prec = prec_exterieur_5.iloc[l]          
            if match_prec['DomicileCode'] == equipe_exterieur and match_prec['Cible'] == 1:
                points_exterieur += 3
            elif match_prec['ExterieurCode'] == equipe_exterieur and match_prec['Cible'] == 0:
                points_exterieur += 3
            else:
                points_domicile += 1
    
        matchs.at[i, 'DomicileForme'] = points_domicile
        matchs.at[i, 'ExterieurForme'] = points_exterieur    

        dom_home_matches = prec_domicile_saison[prec_domicile_saison['DomicileCode'] == equipe_domicile]
        dom_away_matches = prec_domicile_saison[prec_domicile_saison['ExterieurCode'] == equipe_domicile]
        
        ext_home_matches = prec_exterieur_saison[prec_exterieur_saison['DomicileCode'] == equipe_exterieur]
        ext_away_matches = prec_exterieur_saison[prec_exterieur_saison['ExterieurCode'] == equipe_exterieur]

        if len(dom_home_matches) > 0:
            matchs.at[i, 'DomicileAvgButsMarques_Home'] = dom_home_matches['Buts domicile'].mean()
            matchs.at[i, 'DomicileAvgButsEncaisses_Home'] = dom_home_matches['Buts exterieur'].mean()
        
        if len(dom_away_matches) > 0:
            matchs.at[i, 'DomicileAvgButsMarques_Away'] = dom_away_matches['Buts exterieur'].mean()
            matchs.at[i, 'DomicileAvgButsEncaisses_Away'] = dom_away_matches['Buts domicile'].mean()
        
        if len(ext_home_matches) > 0:
            matchs.at[i, 'ExterieurAvgButsMarques_Home'] = ext_home_matches['Buts domicile'].mean()
            matchs.at[i, 'ExterieurAvgButsEncaisses_Home'] = ext_home_matches['Buts exterieur'].mean()
        
        if len(ext_away_matches) > 0:
            matchs.at[i, 'ExterieurAvgButsMarques_Away'] = ext_away_matches['Buts exterieur'].mean()
            matchs.at[i, 'ExterieurAvgButsEncaisses_Away'] = ext_away_matches['Buts domicile'].mean()
    
    matchs['DiffForme'] = matchs['DomicileForme'] - matchs['ExterieurForme']
    matchs['DiffButsDomicile'] = matchs['DomicileAvgButsMarques_Home'] - matchs['DomicileAvgButsEncaisses_Home']
    matchs['DiffButsExterieur'] = matchs['ExterieurAvgButsMarques_Away'] - matchs['ExterieurAvgButsEncaisses_Away']
    matchs['DiffButs'] = matchs['DiffButsDomicile'] - matchs['DiffButsExterieur']
    matchs['DiffButsGlobal'] = ((matchs['DomicileAvgButsMarques_Home'] - matchs['DomicileAvgButsEncaisses_Home'] +
                          matchs['DomicileAvgButsMarques_Away'] - matchs['DomicileAvgButsEncaisses_Away']) - 
                          (matchs['ExterieurAvgButsMarques_Home'] - matchs['ExterieurAvgButsEncaisses_Home'] +
                          matchs['ExterieurAvgButsMarques_Away'] - matchs['ExterieurAvgButsEncaisses_Away'])) 
    return matchs

def faf_equipes(matchs):
    matchs = matchs.sort_values('Date')                  
    matchs['FAF_VictoiresDomicile_Dom'] = 0              
    matchs['FAF_VictoiresDomicile_Ext'] = 0              
    matchs['FAF_VictoiresExterieur_Dom'] = 0             
    matchs['FAF_VictoiresExterieur_Ext'] = 0             
    matchs['FAF_Nuls_Dom'] = 0                           
    matchs['FAF_Nuls_Ext'] = 0                           

    for i in range(len(matchs)):                         
        
        match_actuel = matchs.iloc[i]                    
        equipe_domicile = match_actuel['DomicileCode']   
        equipe_exterieur = match_actuel['ExterieurCode'] 
        date_match = match_actuel['Date']                

        matchs_prec = matchs[matchs['Date'] < date_match]
        
        cas1 = matchs_prec[(matchs_prec['DomicileCode'] == equipe_domicile) & (matchs_prec['ExterieurCode'] == equipe_exterieur)]
        cas2 = matchs_prec[(matchs_prec['DomicileCode'] == equipe_exterieur) & (matchs_prec['ExterieurCode'] == equipe_domicile)]
        
        victoires_domicile_a_dom = sum(cas1['Cible'] == 1)  
        victoires_exterieur_a_ext = sum(cas1['Cible'] == 0) 
        nul_dom_a_dom = sum(cas1['Cible'] == 2)             
        
        victoires_exterieur_a_dom = sum(cas2['Cible'] == 1) 
        victoires_domicile_a_ext = sum(cas2['Cible'] == 0)  
        nul_dom_a_ext = sum(cas1['Cible'] == 2)             

        
        matchs.at[i, 'FAF_VictoiresDomicile_Dom'] = victoires_domicile_a_dom        
        matchs.at[i, 'FAF_VictoiresDomicile_Ext'] = victoires_domicile_a_ext        
        matchs.at[i, 'FAF_VictoiresExterieur_Dom'] = victoires_exterieur_a_dom      
        matchs.at[i, 'FAF_VictoiresExterieur_Ext'] = victoires_exterieur_a_ext      
        matchs.at[i, 'FAF_Nul_Domicile_Dom'] = nul_dom_a_dom                        
        matchs.at[i, 'FAF_Nul_Domicile_Ext'] = nul_dom_a_ext                        
        
        matchs['FAF_Diff'] = matchs['FAF_VictoiresDomicile_Dom'] - matchs['FAF_VictoiresExterieur_Ext']
        matchs['FAF_DiffGlobal'] = (matchs['FAF_VictoiresDomicile_Dom'] - matchs['FAF_VictoiresExterieur_Ext']) + (matchs['FAF_VictoiresDomicile_Ext'] - matchs['FAF_VictoiresExterieur_Dom'])
        matchs['FAF_DiffNul'] = matchs['FAF_Nul_Domicile_Dom'] - matchs['FAF_Nul_Domicile_Ext']
    return matchs

with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

def predire_matchs(domicile, exterieur, date):

    features = [
    'DomicileCode', 'ExterieurCode', 'DomicileForme', 'ExterieurForme', 
    'DiffForme', 'FAF_VictoiresDomicile_Dom', 'FAF_Diff', 'FAF_DiffGlobal', 
    'FAF_VictoiresExterieur_Dom', 'FAF_VictoiresDomicile_Ext', 
    'FAF_VictoiresExterieur_Ext', 'SemaineSaison', 'Mois', 'AnneeSaison',
    'DomicileAvgButsMarques_Home', 'DomicileAvgButsEncaisses_Home', 
    'DomicileAvgButsMarques_Away', 'DomicileAvgButsEncaisses_Away', 
    'ExterieurAvgButsMarques_Home', 'ExterieurAvgButsEncaisses_Home', 
    'ExterieurAvgButsMarques_Away', 'ExterieurAvgButsEncaisses_Away', 
    'DiffButsDomicile', 'DiffButsExterieur', 'DiffButs', 'DiffButsGlobal',
    'FAF_Nul_Domicile_Dom', 'FAF_Nul_Domicile_Ext', 'FAF_DiffNul' 
]
    date_limite = pd.to_datetime(date) - pd.DateOffset(years=2)
    predire_df = matchs[matchs['Date'] >= date_limite].copy().reset_index(drop=True)
    
    new_row = {}
    for col in features:
        if col in predire_df.columns:
            new_row[col] = 0.0
    
    date = pd.to_datetime(date)

    new_row['Date'] = date

    if domicile not in code_equipe:
        print(f"ERREUR: L'équipe '{domicile}' n'existe pas dans le dictionnaire.")
        print(f"Équipes disponibles: {list(code_equipe.keys())}")
        return None
    
    if exterieur not in code_equipe:
        print(f"ERREUR: L'équipe '{exterieur}' n'existe pas dans le dictionnaire.")
        print(f"Équipes disponibles: {list(code_equipe.keys())}")
        return None
    
    new_row['DomicileCode'] = code_equipe.get(domicile)
    new_row['ExterieurCode'] = code_equipe.get(exterieur)
    new_row['DomicileCode'] = int(new_row['DomicileCode'])
    new_row['ExterieurCode'] = int(new_row['ExterieurCode'])
    new_row['Mois'] = (date.month)

    if date > pd.Timestamp(date.year, month=8, day=5):
        new_row['AnneeSaison'] = int(date.year - 1)
    else:
        new_row['AnneeSaison'] = int(date.year)

    new_row['SemaineSaison'] = int(((pd.Timestamp(date) - pd.Timestamp(year=int(new_row['AnneeSaison']), month=8, day=5)).days // 7) + 1)
    
    predire_df.loc[len(predire_df)] = new_row


    stats_df = stats_equipe(predire_df)
    predire_df = faf_equipes(stats_df)

    X_pred = predire_df.iloc[-1:][features]
    
    for col in X_pred.columns:
        if X_pred[col].dtype == 'object':
            X_pred[col] = pd.to_numeric(X_pred[col])

    with open('xgb_model.pkl', 'rb') as file:
        best_xgb = pickle.load(file)

    y_pred = best_xgb.predict(X_pred)
    y_pred_prob = best_xgb.predict_proba(X_pred)
    if y_pred == 0:
        prediction = print(f"Le modèle prédit une victoire de {exterieur} à {y_pred_prob[0, 0]:.0%}!")
    elif y_pred==1:
        prediction = print(f"Le modèle prédit une victoire de {domicile} à {y_pred_prob[0, 1]:.0%}!")
    else:
        prediction = print(f"Le modèle prédit un match nul à {y_pred_prob[0, 2]:.0%}!")
    return prediction