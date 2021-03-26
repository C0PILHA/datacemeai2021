"""
Created on Wed Mar 24 11:13:03 2021

@author: Rodrigo Contreras
"""
#
#Comandos importantes para executar a aplicação:
#    
#    ---
#    Go to streamlit apps folder
#    1) cd "C:\Users\rodri\AppData\Roaming\Python\Python36\Scripts" 
#    
#    ---
#    Host the server
#    2) streamlit run "C:\Users\rodri\Google Drive\Pós-doutorado\Projeto - Gustavo\CEMEAI 2021\Dados Hackathon USP IMPA - Shell Box (Raízen)\aplicacao_streamlit.py"
#
#
# Na sequência, aparecerá o endereço de acesso à aplicação do "command". 
#Geralmente é na forma http://localhost:<<NÚMERO_PORTA>>


import numpy as np
import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import folium
import funcoes_auxiliares as faux
import plotly.express as px


@st.cache  # 👈 This function will be cached
def my_slow_function():
    
    [df, df_store, IDs] = faux.read_tb_postos(True)
    
    return [df, df.values, IDs, df_store]


st.set_page_config(page_title="VI Workshop CEMEAI - Raízen", layout='wide')

left_column, right_column = st.beta_columns((1,1))

[df, X, IDs, df_store] = my_slow_function()



#%% Definição do tamanho dos circulos
size_select_box = st.sidebar.selectbox(
    'How would you like to define the size?',
    (df.columns.values)
)
size_values = df[size_select_box].values

#%% Definição da coloração dos círculos

# Vai definir por medida do dataframe ou por clusterização?
clustering = st.sidebar.checkbox('Color by k-means')

color_select_box = ""
prj=[]
rotulacao_latente = []
color_values = []
if clustering:
    color_select_box = "labels"
    [prj, rotulacao_latente] = faux.calc_plot(X=X.T,
                                        number_of_groups=4,
                                        use_scale=True)
    color_values = rotulacao_latente
else:
    color_select_box = st.sidebar.selectbox(
        'How would you like to define the color?',
        (df.columns[1:].values)
    )
    color_values = df[color_select_box].values


#%% Definição do "posto" inicial
store_select_box = st.sidebar.selectbox(
    'Select a store ID',
    ([np.where(IDs == idstore)[0][0] for idstore in  IDs])#(IDs)
)


#%% Cálculo de escala de cores e tamanhos dos círculos
[color, size] = faux.calc_scale_color_size(df_store, 
                                            color_values,
                                            size_values)
#                                        df[color_select_box].values,
#                                        df[size_select_box].values)


#%% Construção do gráfico
plot_type = st.sidebar.radio("Choose the plot:", ('Scatter Plot','SVD Projection')) 


x_name = ''
y_name = ''
title_plot = ''

if plot_type == 'Scatter Plot':
#    st.write("nao foi selecionado nnhum tipo")
    select_x = st.sidebar.selectbox(
        'Select the first axis',
        (df.columns.values)
    )
    select_y = st.sidebar.selectbox(
        'Select the second axis',
        (df.columns.values),
        index = 1
    )
    prj = np.zeros((2,len(IDs)))
    
    prj[0,:] = df[select_x].values
    prj[1,:] = df[select_y].values
    
    x_name = select_x
    y_name = select_y
    title_plot = "Scatter Plot"
    
else:
    if not(clustering):
        [prj, rotulacao_latente] = faux.calc_plot(X=X.T,
                                    number_of_groups=4,
                                    use_scale=True)
    
    x_name = 'x'
    y_name = 'y'
    title_plot = "Latent (SVD) Projection"
    
faux.calc_and_plot_latent_plotlyexpress(st=left_column, 
                                        df=df, 
                                        prj=prj,#np.asarray(prj).reshape(len(prj)), 
                                        x_name=x_name,
                                        y_name=y_name, 
                                        title_plot=title_plot,
                                        colorvalues=color_values,
                                        IDs=IDs, 
                                        sizescale=size, 
                                        colorscale=color,
                                        initial_id=store_select_box)
        

#%% Definição do mapa
with right_column:
    posto = df_store.loc[df_store['store_id']==IDs[store_select_box]]
    init_lat = posto['latitude'].values[0]
    init_lon = posto['longitude'].values[0]
    faux.plot_map(df_store, df, IDs, size_select_box,color_values, init_lat, init_lon,IDs[store_select_box], color, size)
    
    im = faux.load_image(IDs[store_select_box])


st.image(im, caption='Store '+str(IDs[store_select_box]))
