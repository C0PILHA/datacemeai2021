"""
Created on Wed Mar 24 21:04:23 2021

@author: Rodrigo Contreras
"""

import numpy as np
import pandas as pd
import matplotlib


def get_columns(df):

  aggr = {}

  for gas_type in df.dropna(subset=['gas_type_id']).gas_type_id.unique():

    df_ = df[df.gas_type_id == gas_type]
    
    sum_quantity_in_litres_using_shell_box = np.sum(df_[df_.using_shell_box == 1]['quantity_in_litres'])
    sum_quantity_in_litres_not_using_shell_box = np.sum(df_[df_.using_shell_box == 0]['quantity_in_litres'])

    mean_price_per_litre = np.mean(df_['price_per_litre'])
    
    total_using_shell_box = df_[df_.using_shell_box == 1].shape[0]
    total_not_using_shell_box = df_[df_.using_shell_box == 0].shape[0]

    aggr[gas_type] = {
        'sum_quantity_in_litres_using_shell_box': sum_quantity_in_litres_using_shell_box,
        'sum_quantity_in_litres_not_using_shell_box': sum_quantity_in_litres_not_using_shell_box,
        'mean_price_per_litre': mean_price_per_litre,
        'total_using_shell_box': total_using_shell_box,
        'total_not_using_shell_box': total_not_using_shell_box
    }

  return aggr

def count_consumer(consumers_df, store_df, trans_df, pumplog_df, store_id):
    
    num_consumer_use_shell_box = trans_df[(trans_df['app_origin']==1) & (trans_df['store_id']==store_id)].consumer_id.unique().size
    num_consumers = trans_df[(trans_df['store_id']==store_id)].consumer_id.unique().size
    num_consumer_shell_box_gas = trans_df[(trans_df['type']=='gas') & (trans_df['store_id']==store_id)].consumer_id.unique().size
    num_consumer_shell_box_select = trans_df[(trans_df['type']=='select') & (trans_df['store_id']==store_id)].consumer_id.unique().size

    mean_fidelity_consumer = 0

    array_of_index = trans_df[trans_df['store_id']==store_id].consumer_id.unique()
    num_ids = len(array_of_index)

    num_registered_consumers = 0
    for i in range(num_ids):
      consumer_id = array_of_index[i]

      consumer = consumers_df.loc[consumers_df['consumer_id']==int(consumer_id)]

      if consumer.size == 0:
        fid = 0
      else:
        fid = consumers_df.loc[consumers_df['consumer_id']==int(consumer_id)].stores.values[0]
        num_registered_consumers = num_registered_consumers + 1

      mean_fidelity_consumer = mean_fidelity_consumer + fid

    # numcons = trans_df[trans_df['store_id']==store_id].consumer_id.unique().size
    if num_registered_consumers != 0:
      mean_fidelity_consumer = mean_fidelity_consumer/num_registered_consumers
    else:
      mean_fidelity_consumer = 0

    return [num_consumer_use_shell_box, num_consumers, num_consumer_shell_box_gas, num_consumer_shell_box_select, mean_fidelity_consumer]



def transform_data(store_id, store_data, store_df, consumers_df, trans_df, pumplog_df):

  data = {'store_id': store_id}

  for gas_type in store_data:
    for d in store_data[gas_type]:
      data[' '.join([gas_type, d])] = store_data[gas_type][d]

  conv = store_df.loc[store_df['store_id']==store_id].shell_select.values[0]
  if np.isnan(conv):
    data['grocery_existence'] = 0
  else:
    data['grocery_existence'] = 1
    data['shell_box_tokens'] = conv # Número de caixas que aceitam shell box

  
  [num_consumer_use_shell_box, num_consumers, num_consumer_shell_box_gas, num_consumer_shell_box_select, mean_fidelity_consumer] = count_consumer(consumers_df, store_df, trans_df, pumplog_df, store_id)

  data['num_consumer_use_shell_box'] = num_consumer_use_shell_box
  data['num_consumers'] = num_consumers
  data['num_consumer_shell_box_gas'] = num_consumer_shell_box_gas
  data['num_consumer_shell_box_select'] = num_consumer_shell_box_select
  data['mean_fidelity_consumer'] = mean_fidelity_consumer

  return data

def calc_plot(X, number_of_groups, use_scale):
    
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    nli, ncol = X.shape
    print("Original number of stores: ", ncol)
    print("Original number of features: ", nli)

    if use_scale:
      from sklearn.preprocessing import StandardScaler
      X = MinMaxScaler().fit_transform(X)#StandardScaler().fit_transform(X)
    
    Tsvd = TruncatedSVD(n_components=3)  
    U = Tsvd.fit_transform(X)
    Sigma = Tsvd.singular_values_ #explained_variance_ratio_
    Vt = Tsvd.components_
    
    # Quantos autovetores?
    r = 3
     
    # Clusterização do espaço latente
    kmeans = KMeans(n_clusters=number_of_groups).fit((np.dot(np.diag(Sigma[:r]),Vt[:r,:])).T)
    rotulacao_latente = kmeans.labels_
    
    
    # projetando os docs_semelhantes nas duas direções principais
    prj = np.dot(np.diag(Sigma[:2]),Vt[:2,:])
    
    return [prj, rotulacao_latente]


def calc_and_plot_latent(st, X, number_of_groups, use_scale, IDs):
    
    import matplotlib.pyplot as plt

    
    [prj, rotulacao_latente] = calc_plot(X, number_of_groups, use_scale)

    colormap = ["blue","green","red","magenta","cyan","yellow","black","white"]
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    
    n,m = prj.shape
    
    for i in range(m):
      ax.scatter(prj[0,i],prj[1,i],color=colormap[rotulacao_latente[i]])
      ax.text(prj[0,i]+.003, prj[1,i]+.003, IDs[i], fontsize=12)
      
    st.write(fig)
    

def calc_and_plot_latent_plotlyexpress(st, df, prj, x_name, y_name, title_plot, colorvalues, IDs, sizescale, colorscale, initial_id):
    
    import plotly.express as px
    import plotly.graph_objects as go

    
        
    name = [np.where(IDs == idstore)[0][0] for idstore in  IDs]
    opacity = 0.3*np.ones(IDs.shape)#[1 if initial_id == np.where(IDs == idstore)[0][0] else 0 for idstore in  IDs]
    opacity[initial_id] = 1
    
    newdf = {}
    newdf[x_name] = prj[0,:]
    newdf[y_name] = prj[1,:]
    newdf['labels'] = colorvalues
#    if colors=='labels':
#        newdf['labels'] = rotulacao_latente
#    else:
#        newdf['labels'] = df[colors].values#[str(v) for v in df[colors].values]
    newdf['name'] = name#IDs
    newdf['size'] = sizescale#df[sizes].values
    newdf['opacity'] = opacity
    
    prjdf = pd.DataFrame(newdf)
    
    # plot the value
    fig = px.scatter(prjdf,
                    x=x_name,
                    y=y_name,
                    title=title_plot,
                    hover_name='name',
                    color='labels',
                    size='size',
                    opacity=0.5,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    width=600, height=600)#color_discrete_map = colorscale)
    
    
    fig.add_annotation(x=prj[0,initial_id], y=prj[1,initial_id],
                text=str(initial_id),    
                showarrow=True,
                  arrowcolor='black',
                  arrowhead = 2,
                  arrowsize = 2)
   

    st.plotly_chart(fig)
    

def build_dataframe():
    
    pumplog_df = pd.read_csv('pumplogs.csv', low_memory=False)
    trans_df = pd.read_csv('transactions.csv', low_memory=False)
    consumers_df = pd.read_csv('consumers.csv', low_memory=False)
    store_df = pd.read_csv('stores.csv', low_memory=False)
    
        
    table_data = {}
    
    for store_id in store_df.store_id.values:
    
      table_data[store_id] = get_columns(pumplog_df[pumplog_df.store_id == store_id])
      
    dataframe = pd.DataFrame([transform_data(store_id, table_data[store_id], store_df, consumers_df, trans_df, pumplog_df) for store_id in table_data])
    
    dataframe_without_nan = dataframe.dropna(subset=dataframe.columns[1:3], how='all').reset_index(drop=True)
    dataframe_without_nan = dataframe_without_nan.fillna(value=0.)
    
    dataframe_without_id = dataframe_without_nan.drop(columns=['store_id'])
    
    return [dataframe, dataframe_without_nan, dataframe_without_id]


def map_color(source_range, nun_colors, map_color_name):
    
    import matplotlib.cm as cm
    
    minima = min(source_range)
    maxima = max(source_range)
    
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(map_color_name, nun_colors))
    
    return mapper

def calc_color(mapper, value):
    return matplotlib.colors.rgb2hex(mapper.to_rgba(value))

def plot_map(df_store, df_tb, IDs, sizes, colorvalues, init_lat, init_lon, selected_ID, mapcolor, mapsize):
    
    from streamlit_folium import folium_static
    import folium
    
    mapper = map_color(source_range=colorvalues,#df_tb[colors].values,
                       nun_colors=df_store['store_id'].size,
                       map_color_name='viridis')
    
    m=folium.Map(width=500,height=500, location=[init_lat,init_lon], tiles="OpenStreetMap",zoom_start=13)

    for store_id in IDs:
        
        posto = df_store.loc[df_store['store_id']==store_id]
        
        
        color_value = colorvalues[np.where(IDs==store_id)[0][0]]#posto_demais_informacoes[colors].values[0]
        radius=mapsize[np.where(IDs==store_id)[0][0]]
        
        lat = posto['latitude'].values[0]
        lon = posto['longitude'].values[0]
        
        folium.CircleMarker([lat, lon],
                            radius=radius,
                                fill_opacity=0.7,
                                tooltip = str(np.where(IDs == store_id)[0][0]),
                                popup = "<p>Please select the new store using the sidebar</p>",#str(lon),
                                color=mapcolor[str(color_value)],#"black",
                                fill=True,
                                fill_color=mapcolor[str(color_value)],#calc_color(mapper, color_value)#"green",   
                               ).add_to(m)
        folium.Marker(
            [init_lat, init_lon], popup="<p>Selected Store <b> Real ID: "+str(selected_ID)+" <b>ID: "+str(np.where(IDs == store_id)[0][0])+"</p>", tooltip="Selected Store"
        ).add_to(m)
    folium_static(m)
    m.save('LA collisions.html')
    
    
    
def calc_scale_color_size(df_store, color_numeric_values, size_numeric_value):
    
    color = dict([])
    
    
    mapper = map_color(source_range=color_numeric_values,
                       nun_colors=df_store['store_id'].size,
                       map_color_name='viridis')
    
    n = len(color_numeric_values)
    for i in range(n):
        val = color_numeric_values[i]
        color[str(val)] = calc_color(mapper, val)
        
    multp_scale = 10
    add_scale = 0.2
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=10, random_state=0)
    size = multp_scale*qt.fit_transform(size_numeric_value.reshape(-1, 1))+add_scale
    
    size = size.reshape(len(size))
    
    return [color, size]
    
def load_image(store_ID):
    from PIL import Image
    import requests

    url = 'https://github.com/C0PILHA/datacemeai2021/blob/main/'+str(store_ID)+'_1.jpg?raw=true'
    
    return Image.open(requests.get(url, stream=True).raw)


def read_tb_postos(download):
    
    if download:
        urlstore = 'https://github.com/C0PILHA/datacemeai2021/blob/main/stores.csv?raw=true'
        urltb = 'https://github.com/C0PILHA/datacemeai2021/blob/main/tabela_sem_NaN_quintafeira_a_tarde_novas_infos2.csv?raw=true'#'https://github.com/C0PILHA/datacemeai2021/blob/main/tabela_sem_NaN_quartafeira_a_tarde_rodrigo.csv?raw=true'
    else:
        urltb = 'tabela_sem_NaN_quartafeira_a_tarde_rodrigo.csv'
        urlstore = 'stores.csv'
        
    df_tb = pd.read_csv(urltb)
    df_store = pd.read_csv(urlstore)
    
    ids = df_tb['store_id'].values
    
    return [df_tb.drop(columns=['store_id']), df_store, ids]
    