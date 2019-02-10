import numpy as np
import pandas as pd
import implicit.als as als
import scipy.sparse as sparse


def create_dataframes(data):
    '''
    The function will create 
    user_data: user specific information
    machine_data: game machine specific information
    session_data: each user's session information played on each game machine
    '''
    
    user_data = user_data = data[['player_id', 'city','state','zipCode','gender','birthDate','dateFirstRegistered','playerClubLevel']].drop_duplicates()
    machine_data = data[['model','manufacturer','gameDenomination','gameTitle']].drop_duplicates()
    machine_data['machine_id'] = range(len(machine_data))
    machine_data.reset_index(inplace=True)
    new_frame = pd.merge(data, machine_data, on=['model','gameDenomination','gameTitle'])
    session_data = new_frame[['player_id', 'gameTitle','coinIn', 'netWin', 'gamePlays','gamesWon','sessionDuration','machine_id']]
    
    return user_data, machine_data, session_data
  
    
        
def rating_dataframe(session_data):
    '''
    The idea is to create a rating from each user for each game machine according to the session information
    rating is the amount of money player put in the machine weighted by the winning percentage of the game, and sessions of player playing specific game over overall player's sessions
    '''
    rating = session_data.groupby(['player_id','machine_id']).sum().reset_index()
    session_count = session_data.groupby(['player_id','machine_id']).count()['sessionDuration']
    session_count = session_count.reset_index()
    session_count.rename(columns={'sessionDuration':'sessionCount'},inplace=True)
    player_sessions = session_count.groupby(['player_id']).sum()['sessionCount'].to_frame()
    player_sessions.rename(columns = {'sessionCount':'playerSessions'},inplace=True)
    result = pd.merge(rating, session_count, on = ['player_id','machine_id'])
    result = pd.merge(result, player_sessions, on = ['player_id'])
# Rating metric could be polished more
    result['rating'] = round(result.coinIn*result.gamesWon/result.gamePlays*(result.sessionCount/result.playerSessions))
#     to_drop = result[result.gamePlays==0]
    result = result[result.gamePlays!=0]
#     matrix = result.pivot(index='player_id', columns='machine_id', values='rating')
    
    return result

# Matrix Factorization recommender
class recommender_mf():
# 
    def fit(self, rating):
        self.als = als.AlternatingLeastSquares(factors = 8)
        
        self.player_machine = sparse.csr_matrix((rating['rating'],(rating['player_id'],rating['machine_id'])))
        self.machine_player = sparse.csr_matrix((rating['rating'],(rating['machine_id'],rating['player_id'])))
        
        self.als.fit(self.machine_player)
        
#         print(type(self.model))
#         print(type(self.machine_player))
        return (self)

    def recommends(self, player_ids):
        recommendations = {}
        for player_id in player_ids:
            machine_ids = []
            recommends = self.als.recommend(player_id,self.player_machine,N=10,filter_already_liked_items=False)
            for x in recommends:
                machine_ids.append(x[0])
            recommendations[player_id] = machine_ids
        return recommendations

        
if __name__=='__main__':
    
    data = pd.read_csv('sample_data_for_100_players.csv',sep=',')
    user_data, machine_data, session_data = create_dataframes(data)
    rating = rating_dataframe(session_data)
#     print(rating.head())
    mf = recommender_mf()
    model = mf.fit(rating)
    player_ids = list(user_data['player_id'])
    recommendations = model.recommends(player_ids)

#     The final result prints out the top 10 recommendations for each player
    for k,v in recommendations.items():
        print ('For player ' + str(k) +' Recommendations are ' + '\n')
        for x in v:
            print (str(machine_data.loc[x]['gameTitle']) + ' with $' + str(machine_data.loc[x]['gameDenomination']) + ' Model ' + str(machine_data.loc[x]['model']))
        print ('\n\n')
    
