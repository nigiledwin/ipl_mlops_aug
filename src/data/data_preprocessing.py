import pandas as pd
import numpy as np
import os

#function to load csv to dataframe
def load_data(data_url):

    if not os.path.exists(data_url):
        print(f"Error: The file {data_url} does not exists")
        return None
    try:
        df=pd.read_csv(data_url)
        print("Data Loaded succesfully")
        return df
    except Exception as e:
        print(f"Eroor loading data:{e}")
        return None

data_url=r"C:\Python_Projects\IPL_MLOPS_Aug\ipl_mlops_aug\data\raw\all_season_details.csv"

df=load_data(data_url)


#Save final processed data

def save_split_df(df,path):
    try:
        df.to_csv(path,index=False)
        print("File saved succesfully")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")    

#Preprocessing function

rolling_window=30

def clean_data(df):
        return(
                 df.drop("comment_id",axis=1)
                 .assign(
                        home_team=lambda df_:(
                                df_
                                .home_team
                                .str.replace("PWI","PBKS")
                                .str.replace('GL','GT')
                                .str.replace("KXIP","PBKS")  
                                ),
                        away_team=lambda df_:(
                                df_
                                .away_team
                                .str.replace("PWI","PBKS")
                                .str.replace('GL','GT')
                                .str.replace("KXIP","PBKS")  
                                ),
                        current_innings=lambda df_:(
                                df_
                                .current_innings
                                .str.replace("PWI","PBKS")
                                .str.replace('GL','GT')
                                .str.replace("KXIP","PBKS")  
                                )               
                        )
              )

def bowling_team(row):
    match_teams=row['match_name'].split('v')
    if row['current_innings']==match_teams[0]:
        return match_teams[1]
    else:
        return match_teams[0]


def feature_engineering(df):

    return(
        df.assign
        (total_score=
                    df.groupby(['match_id','innings_id'])['runs'].transform('sum'),
         batsman_total_runs=
                    df.groupby(['match_id','batsman1_name'])['wkt_batsman_runs'].transform('sum'),
         batsman_total_balls=
                    df.groupby(['match_id','batsman1_name'])['wkt_batsman_balls'].transform('sum'),
         current_runs=
                    df.groupby(['match_id','innings_id'])['runs'].transform('cumsum'),
         rolling_back_30balls_runs=
                    df.groupby(['match_id','innings_id'])['runs'].rolling(window=rolling_window,min_periods=1).sum().reset_index(level=[0,1],drop=True),
         rolling_back_30balls_wkts=
                    df.groupby(['match_id','innings_id'])['wicket_id'].rolling(window=rolling_window,min_periods=1).count().reset_index(level=[0,1],drop=True),
         bowling_team=df.apply(bowling_team,axis=1)
         ).rename(columns={'current_innings':'batting_team'})
         [['total_score','batting_team','bowling_team','over', 'ball','current_runs','rolling_back_30balls_runs','rolling_back_30balls_wkts']]
         

        
    )
 
df = load_data("data/raw/all_season_details.csv")  
df_clean=clean_data(df)
df_final=feature_engineering(df_clean).iloc[30:,:]
print(df_final.head())
save_split_df(df_final,"data/processed/df_final.csv")