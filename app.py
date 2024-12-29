import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor



# pipe = pickle.load(open('pipe.pkl','rb'))
df = pd.read_csv('D:\Career\Data Science\Projects_DA_DS\Watches\Watch_Market\cleaned_watched.csv')

watch_rec1 = pd.read_csv('D:\Career\Data Science\Projects_DA_DS\Watches\Watch_Market\watch_rec1.csv')
watch_rec2 = pd.read_csv('D:\Career\Data Science\Projects_DA_DS\Watches\Watch_Market\watch_rec2.csv')

from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

st.title("Watch Recommendations")

st.sidebar.title('Select Option')
options = st.sidebar.selectbox('Options',['External','Internal','Price_Prediction1'])

# Impute by mean
num1 = ['Price']
# Impute by mean
cat1 = ['Case material','Bracelet material','Shape','Crystal','Clasp']

a1 = watch_rec1[['Price','Case material','Bracelet material','Shape','Crystal','Clasp']]

# Column Transformer for preprocessing
preprocessor1 = ColumnTransformer( 
                    transformers=[ 
                        ('cat', OneHotEncoder(), cat1), 
                        ('num', Normalizer(), num1) 
                    ])
# Fit the preprocessor
preprocessed_data1 = preprocessor1.fit_transform(a1)

def recommend_watches1(example_watch1, a1, preprocessor1, top_n=3):
    '''
    Recommend watches similar to the example_watch.
    Parameters:
        - example_watch: DataFrame containing the example watch features.
        - df: DataFrame containing the watch dataset.
        - preprocessor: ColumnTransformer used for preprocessing
        - top_n: Number of top recommendations to return.
    Returns:
        - DataFrame containing the recommended watches.
    '''
    # Transform the example watch
    transformed_example_watch1 = preprocessor1.transform(example_watch1)
    
    # Compute cosine similarity
    similarity_scores1 = cosine_similarity(transformed_example_watch1, preprocessed_data1)
    
    # Get top N recommendations
    top_n_indices = similarity_scores1[0].argsort()[-top_n:][::-1]
    
    # Recommended watches
    recommended_watches1 = df[['Brand','Movement','Gender','Price','Volume']].iloc[top_n_indices]
    return recommended_watches1



# __________________________________________________________________________________________________________

# Impute by mean
num2 = ['Price']
# Impute by mean
cat2 = ['Bracelet color','Dial','Gender','Crystal']

a2 = watch_rec2[['Price','Bracelet color','Dial','Gender','Crystal']]

# Column Transformer for preprocessing
preprocessor2 = ColumnTransformer( 
                    transformers=[ 
                        ('cat', OneHotEncoder(), cat2), 
                        ('num', Normalizer(), num2) 
                    ])
# Fit the preprocessor
preprocessed_data2 = preprocessor2.fit_transform(a2)

def recommend_watches2(example_watch2, a2, preprocessor2, top_n2=3):
    '''
    Recommend watches similar to the example_watch.
    Parameters:
        - example_watch: DataFrame containing the example watch features.
        - df: DataFrame containing the watch dataset.
        - preprocessor: ColumnTransformer used for preprocessing
        - top_n: Number of top recommendations to return.
    Returns:
        - DataFrame containing the recommended watches.
    '''
    # Transform the example watch
    transformed_example_watch2 = preprocessor2.transform(example_watch2)
    
    # Compute cosine similarity
    similarity_scores2 = cosine_similarity(transformed_example_watch2, preprocessed_data2)
    
    # Get top N recommendations
    top_n_indices2 = similarity_scores2[0].argsort()[-top_n2:][::-1]
    
    # Recommended watches
    recommended_watches2 = df[['Brand','Movement','Gender','Price','Volume']].iloc[top_n_indices2]
    return recommended_watches2

# ___________________________________________________________________________________________________________

if options == 'External':
    
        # Price
    Price = st.number_input('Price',min_value=df['Price'].min(),max_value=df['Price'].max())

    # Case material
    Case_material = st.selectbox('Case material',df['Case material'].unique())

    #'Bracelet material',
    Bracelet_material = st.selectbox('Bracelet material',df['Bracelet material'].unique())

    #  'Shape',
    Shape = st.selectbox('Shape',df['Shape'].unique())

    # 'Crystal', 
    Crystal = st.selectbox('Crystal',df['Crystal'].unique())
    
    # 'Clasp'
    Clasp = st.selectbox('Clasp',df['Clasp'].unique())

    # Input field for top_n 
    top_n = st.number_input('Number of Recommendations', min_value=1, max_value=25, value=3, step=1)

    example_watch1 = pd.DataFrame({
        'Price': [Price],
        'Case material': [Case_material],
        'Bracelet material': [Bracelet_material],
        'Shape': [Shape],
        'Crystal': [Crystal],
        'Clasp' : [Clasp]
        })
    
    if st.button('Recommend Watch'):
        recommendations1 = recommend_watches1(example_watch1,a1,preprocessor1,top_n)
        st.write("Recommended Watches:")
        st.write(recommendations1)

elif options == 'Internal':
    # Price
    Price2 = st.number_input('Watch Price',min_value=df['Price'].min(),max_value=df['Price'].max())

    # Case material
    Bracelet_colors = st.selectbox('Bracelet color',df['Bracelet color'].unique())

    #'Bracelet material',
    Dial2 = st.selectbox('Dial',df['Dial'].unique())

    #  'Shape',
    Gender2 = st.selectbox('Gender',df['Gender'].unique())

    # 'Crystal', 
    Crystal2 = st.selectbox('Watch Crystal',df['Crystal'].unique())
        

    # Input field for top_n 
    top_n2 = st.number_input('Numbers of Recommendations', min_value=1, max_value=25, value=3, step=1)

    example_watch2 = pd.DataFrame({
        'Price': [Price2],
        'Bracelet color': [Bracelet_colors],
        'Dial': [Dial2],
        'Gender' : [Gender2],
        'Crystal': [Crystal2],
        })
    
    if st.button('Recommend Watch'):
        recommendations2 = recommend_watches2(example_watch2,a2,preprocessor2,top_n2)
        st.write("Recommended Watches:")
        st.write(recommendations2)

elif options == 'Price_Prediction1':
    # Load the trained model 
    pred_options = st.sidebar.selectbox('Pred_Options',['Level 1','Level 2','Level 3'])

    internal_pr = pd.read_csv('D:\Career\Data Science\Projects_DA_DS\Watches\Watch_Market\internal_pr.csv').drop(columns=['Unnamed: 0'])

    if pred_options == 'Level 1':
        X1 = internal_pr.drop(columns=['Price'])
        y1 = internal_pr['Price']

        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

        ohe = OneHotEncoder()
        norm = Normalizer()
        scaler = StandardScaler()

        X_train_sc1 = ohe.fit_transform(X_train1)
        
        # X_test_sc1 = ohe.transform(X_test1)

        y_train_sc1 = scaler.fit_transform(pd.DataFrame(y_train1))

        # y_test1_sc1 = norm.transform(pd.DataFrame(y_test1))

        dt1 = DecisionTreeRegressor()
        reg = dt1.fit(X_train_sc1,y_train_sc1)
        
        # Title 
        st.title('Price Prediction')

        # Movement
        Movement3 = st.selectbox('Movement',df['Movement'].unique())

        # Gender
        Gender3 = st.selectbox('GenderType',df['Gender'].unique())

        # Dial
        Dial3 = st.selectbox('DialType',df['Dial'].unique())

        # Bracelet color
        Bracelet_color3 = st.selectbox('Bracelet_color_Type',df['Bracelet color'].unique())

        example_inp1 = pd.DataFrame({
            'Movement': [Movement3],
            'Gender' : [Gender3],
            'Dial' : [Dial3],
            'Bracelet color' : [Bracelet_color3]
        })
        inp1 = ohe.transform(example_inp1)


        if st.button('Predict_Price'):
            # inp1 = pd.DataFrame(inp1,columns=['Movement', 'Gender', 'Dial', 'Bracelet color'])

            y_pred1 = reg.predict(inp1)
            y_pred1 = scaler.inverse_transform(y_pred1.reshape(-1, 1))
            st.title("Rs " + str(np.round(y_pred1[0])))
