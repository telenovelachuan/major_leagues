import pandas as pd

raw_input = pd.read_csv("../data/interim/spi_matches.csv")
club_spi_rankings = pd.read_csv("../data/raw/spi_global_rankings.csv")
RESULT_TYPES = ['W', 'D', 'L']  # win, draw, lose

'''
Pick the weakest 25% teams(lowest spi) from each league
'''
sorted = club_spi_rankings.sort_values(['league', 'spi'])
percentile_30s = sorted.groupby('league').quantile(.3).reset_index(level=0)
percentile_70s = sorted.groupby('league').quantile(.7).reset_index(level=0)
print "percentile_30s:{}".format(percentile_30s)


def get_match_result(row, for_whom):
    if row.score1 == row.score2:
        return "D"
    result = "W" if (row.team1 == for_whom and row.score1 > row.score2) or (row.team2 == for_whom and row.score2 > row.score1) else "L"
    return result


def get_past_matches(df, team, count=None, opponents=None):
    if not count:
        count = len(df.index)
    if opponents:
        return df[((df.team1 == team) & df.team2.isin(opponents)) | ((df.team2 == team) & df.team1.isin(opponents))][:count]
    return df[(df.team1 == team) | (df.team2 == team)][:count]


def get_last_n_count(df, team, result_type, n=None):
    def _get_score_condition(row, isHome):
        if result_type == 'D':
            return row.score1 == row.score2
        elif result_type == "W":
            return row.score1 > row.score2 if isHome else row.score1 < row.score2
        else:
            return row.score1 < row.score2 if isHome else row.score1 > row.score2

    past_n_matches = get_past_matches(df, team, n)
    return sum([1 for _, m in past_n_matches.iterrows() if ((m.team1 == team and _get_score_condition(m, True)) or (m.team2 == home_team and _get_score_condition(m, False)))])


def get_streaks(df, team, result_type):
    past_matches_reverse = get_past_matches(df, team).iloc[::-1]
    team_form = [get_match_result(row, team) for _, row in past_matches_reverse.iterrows()]
    neg_types = [t for t in RESULT_TYPES if t != result_type]
    return min([len(team_form) if nt not in team_form else team_form.index(nt) for nt in neg_types])


def get_goal_count(df, team, type, match_count=None, opponents=None):
    matches = get_past_matches(df, team, match_count, opponents)
    if match_count and opponents:
        #  eliminate cases where there's not enough past matches
        if len(matches.index) < match_count:
            return -1
    result = 0
    for _, row in matches.iterrows():
        if row.team1 == team:
            result += (row['score1'] if type == "for" else row['score2'])
        else:
            result += (row['score2'] if type == "for" else row['score1'])
    return result


NEW_ATTRIBUTES = ["h_won_row", "h_lost_row", "h_won_last_5", "h_lst_last_5", "h_won_last_10", "h_lst_last_10", "h_drn_last_5", "h_drn_last_10",
                  "a_won_row", "a_lost_row", "a_won_last_5", "a_lst_last_5", "a_won_last_10", "a_lst_last_10", "a_drn_last_5", "a_drn_last_10",
                  "h_gf_5", "h_gf_10", "h_ga_5", "h_ga_10", "h_gl_dif_5", "h_gl_dif_10", "a_gf_5", "a_gf_10", "a_ga_5", "a_ga_10", "a_gl_dif_5", "a_gl_dif_10",
                  "h_gfw_5", "h_gfs_5", "h_gaw_5", "h_gas_5", "a_gfw_5", "a_gfs_5", "a_gaw_5", "a_gas_5"
                  ]
NEW_ATTR_DICT = {attr: [] for attr in NEW_ATTRIBUTES}
row_len = len(raw_input.index)
for idx, row in raw_input.iterrows():
    home_team = row.team1
    away_team = row.team2
    past_matches = raw_input[:idx]

    #  generate past N matches
    NEW_ATTR_DICT['h_won_last_5'].append(get_last_n_count(past_matches, home_team, 'W', 5))
    NEW_ATTR_DICT['a_won_last_5'].append(get_last_n_count(past_matches, away_team, 'W', 5))
    NEW_ATTR_DICT['h_won_last_10'].append(get_last_n_count(past_matches, home_team, 'W', 10))
    NEW_ATTR_DICT['a_won_last_10'].append(get_last_n_count(past_matches, away_team, 'W', 10))

    NEW_ATTR_DICT['h_drn_last_5'].append(get_last_n_count(past_matches, home_team, 'D', 5))
    NEW_ATTR_DICT['a_drn_last_5'].append(get_last_n_count(past_matches, away_team, 'D', 5))
    NEW_ATTR_DICT['h_drn_last_10'].append(get_last_n_count(past_matches, home_team, 'D', 10))
    NEW_ATTR_DICT['a_drn_last_10'].append(get_last_n_count(past_matches, away_team, 'D', 10))

    NEW_ATTR_DICT['h_lst_last_5'].append(get_last_n_count(past_matches, home_team, 'L', 5))
    NEW_ATTR_DICT['a_lst_last_5'].append(get_last_n_count(past_matches, away_team, 'L', 5))
    NEW_ATTR_DICT['h_lst_last_10'].append(get_last_n_count(past_matches, home_team, 'L', 10))
    NEW_ATTR_DICT['a_lst_last_10'].append(get_last_n_count(past_matches, away_team, 'L', 10))

    #  generate streaks
    NEW_ATTR_DICT['h_won_row'].append(get_streaks(past_matches, home_team, 'W'))
    NEW_ATTR_DICT['a_won_row'].append(get_streaks(past_matches, away_team, 'W'))
    NEW_ATTR_DICT['h_lost_row'].append(get_streaks(past_matches, home_team, 'L'))
    NEW_ATTR_DICT['a_lost_row'].append(get_streaks(past_matches, away_team, 'L'))

    #  generate recent scores
    NEW_ATTR_DICT['h_gf_5'].append(get_goal_count(past_matches, home_team, 'for', 5))
    NEW_ATTR_DICT['h_gf_10'].append(get_goal_count(past_matches, home_team, 'for', 10))
    NEW_ATTR_DICT['h_ga_5'].append(get_goal_count(past_matches, home_team, 'against', 5))
    NEW_ATTR_DICT['h_ga_10'].append(get_goal_count(past_matches, home_team, 'against', 10))
    NEW_ATTR_DICT['h_gl_dif_5'].append(get_goal_count(past_matches, home_team, 'for', 5) - get_goal_count(past_matches, home_team, 'against', 5))
    NEW_ATTR_DICT['h_gl_dif_10'].append(get_goal_count(past_matches, home_team, 'for', 10) - get_goal_count(past_matches, home_team, 'against', 10))

    NEW_ATTR_DICT['a_gf_5'].append(get_goal_count(past_matches, away_team, 'for', 5))
    NEW_ATTR_DICT['a_gf_10'].append(get_goal_count(past_matches, away_team, 'for', 10))
    NEW_ATTR_DICT['a_ga_5'].append(get_goal_count(past_matches, away_team, 'against', 5))
    NEW_ATTR_DICT['a_ga_10'].append(get_goal_count(past_matches, away_team, 'against', 10))
    NEW_ATTR_DICT['a_gl_dif_5'].append(get_goal_count(past_matches, away_team, 'for', 5) - get_goal_count(past_matches, away_team, 'against', 5))
    NEW_ATTR_DICT['a_gl_dif_10'].append(get_goal_count(past_matches, away_team, 'for', 10) - get_goal_count(past_matches, away_team, 'against', 10))

    '''
    metrics on club
    '''
    if row.league not in percentile_30s:
        bottom_p30_in_league = []
    else:
        league_p30 = float(percentile_30s[percentile_30s.league == row.league].spi)
        bottom_p30_in_league = [r['name'] for _, r in sorted[(sorted.league == row.league) & (sorted.spi <= league_p30)].iterrows()]
    if row.league not in percentile_70s:
        top_p30_in_league = []
    else:
        league_p70 = float(percentile_70s[percentile_70s.league == row.league].spi)
        top_p30_in_league = [r['name'] for _, r in sorted[(sorted.league == row.league) & (sorted.spi >= league_p70)].iterrows()]

    # a. goal scored during last 5 games against weak teams
    NEW_ATTR_DICT['h_gfw_5'].append(get_goal_count(past_matches, home_team, 'for', 5, bottom_p30_in_league))
    NEW_ATTR_DICT['a_gfw_5'].append(get_goal_count(past_matches, away_team, 'for', 5, bottom_p30_in_league))

    # b. goal scored during last 5 games again strong teams
    NEW_ATTR_DICT['h_gfs_5'].append(get_goal_count(past_matches, home_team, 'for', 5, top_p30_in_league))
    NEW_ATTR_DICT['a_gfs_5'].append(get_goal_count(past_matches, away_team, 'for', 5, top_p30_in_league))

    # c. goals conceded during last 5 games against weak teams
    NEW_ATTR_DICT['h_gaw_5'].append(get_goal_count(past_matches, home_team, 'against', 5, bottom_p30_in_league))
    NEW_ATTR_DICT['a_gaw_5'].append(get_goal_count(past_matches, away_team, 'against', 5, bottom_p30_in_league))

    # d. goals conceded during last 5 games against strong teams
    NEW_ATTR_DICT['h_gas_5'].append(get_goal_count(past_matches, home_team, 'against', 5, top_p30_in_league))
    NEW_ATTR_DICT['a_gas_5'].append(get_goal_count(past_matches, away_team, 'against', 5, top_p30_in_league))

    print "{} out of {} finished".format(idx, row_len)

print "NEW_ATTR_DICT constructed"
for attr in NEW_ATTRIBUTES:
    raw_input[attr] = NEW_ATTR_DICT[attr]
raw_input.to_csv(index=False, path_or_buf="../data/processed/processed.csv")
print "done."

