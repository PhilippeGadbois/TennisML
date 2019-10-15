import const as const
from datetime import date, datetime
from scipy.stats import zscore



def pre_clean(m):
    # Invalid Percentages
    m = m[(m['FirstServePercentage_1'] > 0) & (m['FirstServePercentage_1'] < 1)]
    m = m[(m['FirstServePercentage_2'] > 0) & (m['FirstServePercentage_2'] < 1)]
    m = m[(m['WonFirstServePercentage_1'] > 0) & (m['WonFirstServePercentage_1'] < 1)]
    m = m[(m['WonFirstServePercentage_2'] > 0) & (m['WonFirstServePercentage_2'] < 1)]
    m = m[(m['WonSecondServePercentage_1'] > 0) & (m['WonSecondServePercentage_1'] < 1)]
    m = m[(m['WonSecondServePercentage_2'] > 0) & (m['WonSecondServePercentage_2'] < 1)]
    m = m[(m['WonReturnPercentage_1'] > 0) & (m['WonReturnPercentage_1'] < 1)]
    m = m[(m['WonReturnPercentage_2'] > 0) & (m['WonReturnPercentage_2'] < 1)]
    m = m[(m['BreakPointsWonPercentage_1'] > 0) & (m['BreakPointsWonPercentage_1'] < 1)]
    m = m[(m['BreakPointsWonPercentage_2'] > 0) & (m['BreakPointsWonPercentage_2'] < 1)]
    m = m[(m['NetApproachesWonPercentage_1'] > 0) & (m['NetApproachesWonPercentage_1'] < 1)]
    m = m[(m['NetApproachesWonPercentage_2'] > 0) & (m['NetApproachesWonPercentage_2'] < 1)]
    m = m[(m['TotalPointsWonPercentage_1'] > 0) & (m['TotalPointsWonPercentage_1'] < 1)]
    m = m[(m['TotalPointsWonPercentage_2'] > 0) & (m['TotalPointsWonPercentage_2'] < 1)]
    # Invalid Speeds
    m = m[(m['AverageFirstServeSpeed_1'] > 120) | (m['AverageFirstServeSpeed_1'].isna())]
    m = m[(m['AverageFirstServeSpeed_2'] > 120) | (m['AverageFirstServeSpeed_2'].isna())]
    m = m[(m['AverageSecondServeSpeed_1'] > 100) | (m['AverageSecondServeSpeed_1'].isna())]
    m = m[(m['AverageSecondServeSpeed_2'] > 100) | (m['AverageSecondServeSpeed_2'].isna())]
    return m


def post_clean(f):
    # First Serve
    f = f[(f['FS'] >= -1) & (f['FS'] <= 1)]
    # Won 1st Serve Percentage
    f = f[(f['W1SP'] >= -1) & (f['W1SP'] <= 1)]
    # Won 2nd Serve Percentage
    f = f[(f['W2SP'] >= -1) & (f['W2SP'] <= 1)]
    # Won Serve Percentage
    f = f[(f['WSP'] >= -1) & (f['WSP'] <= 1)]
    # Won Return Percentage
    f = f[(f['WRP'] >= -1) & (f['WRP'] <= 1)]
    # Total Points Won Percentage
    f = f[(f['TPW'] >= -1) & (f['TPW'] <= 1)]
    # Aces
    f = f
    # Double Faults
    f = f
    # Unforced Errors (not enough data)
    f = f.drop(['UE'], axis=1)
    # Winners (not enough data)
    f = f.drop(['WIS'], axis=1)
    # Break Points
    f = f[(f['BP'] >= -1) & (f['BP'] <= 1)]
    # Net Approaches (not enough data)
    f = f.drop(['NA'], axis=1)
    # Average 1st Speed (not enough data)
    f = f.drop(['A1S'], axis=1)
    # Average 2nd Speed (not enough data)
    f = f.drop(['A2S'], axis=1)
    # Complete
    f = f[(f['COMPLETE'] >= -1) & (f['COMPLETE'] <= 1)]
    # Serve Advantage
    f = f[(f['SERVEADV'] >= -1) & (f['SERVEADV'] <= 1)]
    # Direct
    f = f[(f['DIRECT'] >= 0) & (f['DIRECT'] <= 1)]
    # Uncertainty Threshold
    f = f[(f['UNC'] > const.UNC_T)]
    # Remove na
    # f = f.dropna()
    # Normalize data
    f[['FS', 'W1SP', 'W2SP', 'WSP', 'WRP', 'TPW', 'ACES', 'DF', 'BP', 'COMPLETE', 'SERVEADV']] = f[['FS', 'W1SP', 'W2SP', 'WSP', 'WRP', 'TPW', 'ACES', 'DF', 'BP', 'COMPLETE', 'SERVEADV']].apply(zscore)
    return f


