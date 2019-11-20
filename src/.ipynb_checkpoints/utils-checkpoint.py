import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def df_augmentation(x1,x2,features,df = pd.DataFrame()):
    
    X1Squared = None
    X2Squared = None
    X1X2 = None
    sinX1 = None
    sinX2 = None
    
    x1,x2 = np.expand_dims(x1,1),np.expand_dims(x2,1)
    
    if df.empty:
        df = pd.DataFrame({'X1':x1.flatten().tolist(),'X2':x2.flatten().tolist()})

    
    if 'X1Squared' in features:
        df['X1Squared'] = (x1**2).flatten().tolist()
    if 'X2Squared' in features:
        df['X2Squared'] = (x2**2).flatten().tolist()
    if 'X1X2' in features:
        df['X1X2'] = (x1*x2).flatten().tolist()
    if 'sinX1' in features:
        df['sinX1'] = np.sin(x1).flatten().tolist()
    if 'sinX2' in features:
        df['sinX2'] = np.sin(x2).flatten().tolist()
    return df

def create_meshgrid(x1,x2):
    
    h = .05  # step size in the mesh
    x1_min, x1_max =x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))
    return xx,yy


def plot_decision_boundary(x1,x2,clf,features):
    
    xx,yy = create_meshgrid(x1,x2)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    df = df_augmentation(xx,yy,features)
    df = df[features]
    feature_vectors=[df.values[:,col].ravel() for col in range(df.values.shape[1])]

    Z = clf.predict(np.column_stack(feature_vectors))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #C = plt.contour(xx, yy, Z, cmap=plt.cm.Paired, linewidths=2)
    C = plt.contourf(xx, yy, Z, cmap=plt.cm.bone)
    plt.contour(xx, yy, Z, colors='#4879c9', linewidths=4, levels=1)
    plt.colorbar(C,drawedges=True);
    
def plot_results(df,clf,features):
    
    sns.set();
    sns.set(rc={'axes.facecolor':'e5eaee', 'figure.facecolor':'e5eaee'})
    plt.subplots(figsize=(10,10))
    plot_decision_boundary(df.X1,df.X2,clf,features)
    ax = sns.scatterplot(data=df,x='X1',y='X2',hue='label',palette=['#e89f51','#4581bc'])
    ax.set(title = 'Output');
    
    
