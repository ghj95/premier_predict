import pandas as pd
import xgboost as xgb

matchs = pd.read_csv('epl_2015-2025.csv')

matchs.rename(columns={'HomeTeam':'Domicile',
                        'AwayTeam':'Exterieur',
                        'Season':'Saison',
                        'FTR':'Cible',
                        'FTHG':'Buts domicile', 
                        'FTAG':'Buts exterieur',
                        }, inplace=True)

for i in range(380, 760):                                  
    row = matchs['Date'][i]                                   
    jour = row[2:4]                                           
    mois = row[5:7]                                           
    annee = row[-2:]                                          
    matchs.loc[i, 'Date'] = f'{jour}/{mois}/20{annee}'  

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
    matchs = matchs.sort_values('Date')                 # on trie les données par date
    
    matchs['DomicileForme'] = 0                         # on initialise deux colonnes ou seront stockés les points de forme
    matchs['ExterieurForme'] = 0

    # on initialise les colonnes où seront stockées les statistiques de but
    colonnes_buts = [
    'DomicileAvgButsMarques_Home', 'DomicileAvgButsEncaisses_Home',     
    'DomicileAvgButsMarques_Away', 'DomicileAvgButsEncaisses_Away',
    'ExterieurAvgButsMarques_Home', 'ExterieurAvgButsEncaisses_Home',
    'ExterieurAvgButsMarques_Away', 'ExterieurAvgButsEncaisses_Away']

    for col in colonnes_buts:
        matchs[col] = 0.0

    for i in range(len(matchs)):                        # on itère à travers les matchs
        match_eval = matchs.iloc[i]                     # on stocke les information du match évalué dans la variable match_eval
        equipe_domicile = match_eval['DomicileCode']
        equipe_exterieur = match_eval['ExterieurCode']
        date_match = match_eval['Date']
        saison_match = match_eval['AnneeSaison']

        # pour l'équipe à domicile la fonction trouve tous les matchs antérieurs à la date du match actuel où l'équipe a joué et les trie par date décroissante en consérvant les 5 plus récents
        prec_domicile_5 = matchs[(matchs['Date'] < date_match) &
                               ((matchs['DomicileCode'] == equipe_domicile) | (matchs['ExterieurCode'] == equipe_domicile))]
        prec_domicile_5 = prec_domicile_5.sort_values('Date', ascending=False).head(5)
        
        # pour l'équipe à domicile la fonction trouve tous les matchs antérieurs depuis le début de la saison où l'équipe a joué et les triepar date décroissante
        prec_domicile_saison = matchs[(matchs['Date'] < date_match) & (matchs['AnneeSaison'] == saison_match) &
                               ((matchs['DomicileCode'] == equipe_domicile) | (matchs['ExterieurCode'] == equipe_domicile))]
        
        # même processus pour l'équipe à l'extérieur
        prec_exterieur_5 = matchs[(matchs['Date'] < date_match) & 
                                ((matchs['DomicileCode'] == equipe_exterieur) | (matchs['ExterieurCode'] == equipe_exterieur))]
        prec_exterieur_5 = prec_exterieur_5.sort_values('Date', ascending=False).head(5)

        prec_exterieur_saison = matchs[(matchs['Date'] < date_match) & (matchs['AnneeSaison'] == saison_match) & 
                                ((matchs['DomicileCode'] == equipe_exterieur) | (matchs['ExterieurCode'] == equipe_exterieur))]

        # calcul des points de forme pour l'équipe à domicile
        # on initialise la variable qui comptera le nombre de points récoltés à domicile
        points_domicile = 0
        for j in range(len(prec_domicile_5)):             # on itère à travers les matchs précédents
            match_prec = prec_domicile_5.iloc[j]          # on stocke les informations de chaque itération du match précédant dans la variable match_prec
            # on itére à travers les 5 derniers matchs et on ajoute 1 point pour chaque victoire à domicile ou à l'extérieur
            if match_prec['DomicileCode'] == equipe_domicile and match_prec['Cible'] == 1:
                points_domicile += 1
            elif match_prec['ExterieurCode'] == equipe_domicile and match_prec['Cible'] == 0:
                points_domicile += 1

        # on répète la même procédure pour l'équipe à l'extérieur
        points_exterieur = 0
        for l in range(len(prec_exterieur_5)):          
            match_prec = prec_exterieur_5.iloc[l]          
            if match_prec['DomicileCode'] == equipe_exterieur and match_prec['Cible'] == 1:
                points_exterieur += 1
            elif match_prec['ExterieurCode'] == equipe_exterieur and match_prec['Cible'] == 0:
                points_exterieur += 1
       
        # on met à jour les colonnes récoltant le total des points à domicile et à l'extérieur
        matchs.at[i, 'DomicileForme'] = points_domicile
        matchs.at[i, 'ExterieurForme'] = points_exterieur    

        # calcul des statistiques de buts pour l'équipe à domicile
        dom_home_matches = prec_domicile_saison[prec_domicile_saison['DomicileCode'] == equipe_domicile]
        dom_away_matches = prec_domicile_saison[prec_domicile_saison['ExterieurCode'] == equipe_domicile]
        
        # calcul des statistiques de buts pour l'équipe à l'exterieur
        ext_home_matches = prec_exterieur_saison[prec_exterieur_saison['DomicileCode'] == equipe_exterieur]
        ext_away_matches = prec_exterieur_saison[prec_exterieur_saison['ExterieurCode'] == equipe_exterieur]

        # calcul des moyennes de buts pour l'équipe à domicile
        if len(dom_home_matches) > 0:
            matchs.at[i, 'DomicileAvgButsMarques_Home'] = dom_home_matches['Buts domicile'].mean()
            matchs.at[i, 'DomicileAvgButsEncaisses_Home'] = dom_home_matches['Buts exterieur'].mean()
        
        if len(dom_away_matches) > 0:
            matchs.at[i, 'DomicileAvgButsMarques_Away'] = dom_away_matches['Buts exterieur'].mean()
            matchs.at[i, 'DomicileAvgButsEncaisses_Away'] = dom_away_matches['Buts domicile'].mean()
        
        # calculs des moyennes de buts pour l'équipe extérieur  
        if len(ext_home_matches) > 0:
            matchs.at[i, 'ExterieurAvgButsMarques_Home'] = ext_home_matches['Buts domicile'].mean()
            matchs.at[i, 'ExterieurAvgButsEncaisses_Home'] = ext_home_matches['Buts exterieur'].mean()
        
        if len(ext_away_matches) > 0:
            matchs.at[i, 'ExterieurAvgButsMarques_Away'] = ext_away_matches['Buts exterieur'].mean()
            matchs.at[i, 'ExterieurAvgButsEncaisses_Away'] = ext_away_matches['Buts domicile'].mean()
    
    # on crée une nouvelle colonne récoltant la différence entre les points de forme à domicile et à l'extérieur
    matchs['DiffForme'] = matchs['DomicileForme'] - matchs['ExterieurForme']

    # on crée de nouvelles colonnes récoltant la différence des performance de buts entre l'équipe à domicile et celle à l'extérieur
    # on fait cela en calculant le différentiel de buts, représentant la différence entre la moyenne de buts marqués et la moyenne de buts encaissés
    matchs['DiffButsDomicile'] = matchs['DomicileAvgButsMarques_Home'] - matchs['DomicileAvgButsEncaisses_Home']
    matchs['DiffButsExterieur'] = matchs['ExterieurAvgButsMarques_Away'] - matchs['ExterieurAvgButsEncaisses_Away']
    matchs['DiffButs'] = matchs['DiffButsDomicile'] - matchs['DiffButsExterieur']
    matchs['DiffButsGlobal'] = ((matchs['DomicileAvgButsMarques_Home'] - matchs['DomicileAvgButsEncaisses_Home'] +
                          matchs['DomicileAvgButsMarques_Away'] - matchs['DomicileAvgButsEncaisses_Away']) - 
                          (matchs['ExterieurAvgButsMarques_Home'] - matchs['ExterieurAvgButsEncaisses_Home'] +
                          matchs['ExterieurAvgButsMarques_Away'] - matchs['ExterieurAvgButsEncaisses_Away'])) 
    return matchs

def faf_equipes(matchs):
    matchs = matchs.sort_values('Date')                 # on trie les données par date
    matchs['FAF_VictoiresDomicile'] = 0                 # on initialise deux colonnes ou seront stockés les victoires en face-à-face
    matchs['FAF_VictoiresExterieur'] = 0

    for i in range(len(matchs)):
        match_eval = matchs.iloc[i]                     # on stocke les information du match évalué dans la variable match_eval
        equipe_domicile = match_eval['DomicileCode']
        equipe_exterieur = match_eval['ExterieurCode']
        date_match = match_eval['Date']

        # on cherche à travers tous les matchs précédent la date du match évalué pour stocker dans prec_faf les matchs ou les deux équipes ont joué l'une contre l'autre
        prec_faf = matchs[(matchs['Date'] < date_match) & 
                    (((matchs['DomicileCode'] == equipe_domicile) & (matchs['ExterieurCode'] == equipe_exterieur)) | 
                    ((matchs['DomicileCode'] == equipe_exterieur) & (matchs['ExterieurCode'] == equipe_domicile)))]

        # calcul du nombre de victoires et de matchs nuls lors des face-à-face
        # on initialise les variables qui vont compter les victoires lors des faces-à-faces
        victoires_domicile = 0
        victoires_exterieur = 0
        
        for j in range(len(prec_faf)):                  # on itère à travers les matchs précédents
            faf = prec_faf.iloc[j]                      # on stocke les informations de chaque itération du match précédant dans la variable faf
            # on commence par itérer à travers le cas où l'équipe à évaluer (Domicile) a joué un face-à-face également à domicile
            if faf['DomicileCode'] == equipe_domicile: 
                # on augmente la variable victoire_domicile dans l'éventualité où l'équipe à évaluer a gagné
                if faf['Cible'] == 1:
                    victoires_domicile += 1
                # on augmente la variable victoire_extérieur dans l'éventualité où l'équipe à évaluer a perdu
                elif faf['Cible'] == 0:
                    victoires_exterieur += 1
            # on répète la même procédure dans le cas où l'équipe à évaluer a joué un face-à-face à l'extérieur
            elif faf['DomicileCode'] == equipe_exterieur: 
                if faf['Cible'] == 1:
                    victoires_exterieur += 1
                elif faf['Cible'] == 0:
                    victoires_domicile += 1
        
        # on met à jour les colonnes récoltant le total des victoires et matchs nuls en face-à-face
        matchs.at[i, 'FAF_VictoiresDomicile'] = victoires_domicile
        matchs.at[i, 'FAF_VictoiresExterieur'] = victoires_exterieur
    
    return matchs

best_params = {
 'subsample': 0.7,
 'scale_pos_weight': 1.225,
 'reg_lambda': 10.0,
 'reg_alpha': 0.5,
 'n_estimators': 1500,
 'min_child_weight': 7,
 'max_depth': 4,
 'learning_rate': 0.01,
 'gamma': 0.1,
 'colsample_bytree': 0.8,
 'colsample_bylevel': 0.8
 }

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgb_model.json')

def predire_matchs(domicile, exterieur, match_date):

    predire_df = matchs

    features = [
        'DomicileCode', 'ExterieurCode', 'DomicileForme',
        'ExterieurForme', 'DiffForme', 'FAF_VictoiresDomicile', 
        'FAF_VictoiresExterieur', 'SemaineSaison', 'Mois', 'AnneeSaison',
        'DomicileAvgButsMarques_Home', 'DomicileAvgButsEncaisses_Home',
        'DomicileAvgButsMarques_Away', 'DomicileAvgButsEncaisses_Away',
        'ExterieurAvgButsMarques_Home', 'ExterieurAvgButsEncaisses_Home',
        'ExterieurAvgButsMarques_Away', 'ExterieurAvgButsEncaisses_Away',
        'DiffButsDomicile', 'DiffButsExterieur', 'DiffButsGlobal'
        #'DiffButs',
    ]
    
    new_row = {col: 0 for col in features if col in predire_df.columns}
    date = pd.to_datetime(match_date)
    new_row['Date'] = date
    new_row['DomicileCode'] = code_equipe.get(domicile)
    new_row['ExterieurCode'] = code_equipe.get(exterieur)
    new_row['Mois'] = date.month
    
    if date > pd.Timestamp(date.year, month=8, day=5):
        new_row['AnneeSaison'] = date.year - 1
    else:
        new_row['AnneeSaison'] = date.year
    reference_date = pd.Timestamp(year=date.year, month=8, day=5)
    new_row['SemaineSaison'] = ((date - reference_date).days // 7) + 1
    
    matchs.loc[len(predire_df)] = new_row

    predire_df = stats_equipe(predire_df)
    predire_df = faf_equipes(predire_df)

    X_pred = predire_df.sort_values('Date', ascending=False).head(1)[features]
    y_pred = xgb_model.predict(X_pred)

    return y_pred