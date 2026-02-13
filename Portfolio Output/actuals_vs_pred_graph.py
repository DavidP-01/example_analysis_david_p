import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.ticker as mtick

def actuals_vs_pred_graph_func(y_actual , y_pred ):
    '''Add the two pandas dataframe columns for the y_actual and y_pred'''
    sns.set_theme()
    #get r_squared value
    r2 = r2_score(y_actual, y_pred)
    #compare graphically, the closer the points are to the 45degree line the better fitting model
    plt.figure(figsize=(4, 4))
    sns.scatterplot(y = y_actual, x = y_pred, alpha = 0.4)
    #45 degree line
    max_val = max(y_actual.max(), y_pred.max())
    min_val = min(y_actual.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    
    #axis format
    fmt = '{x:,.12g}'
    tick = mtick.StrMethodFormatter(fmt)
    plt.gca().xaxis.set_major_formatter(tick)
    plt.gca().yaxis.set_major_formatter(tick) 
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=3)

    plt.ylabel('Actuals')
    plt.xlabel('Predictions')
    plt.title('Actuals vs Predictions - ' + f'$R^2$ Score: {r2:.2f}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()
