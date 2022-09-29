import pandas as pd
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt


st.set_page_config(page_title="Superb_project", page_icon='random', layout="centered",
                   initial_sidebar_state="expanded", menu_items=None)
skills = pd.read_excel('public_use-industry-skills-needs.xlsx', sheet_name='Industry Skills Needs')

st.sidebar.write('The model predicts skills according to their ranking')

st.title('Superb final project on ADS')
st.write('we are going to look at a datasets that analysis public skills most used from 2015-2019 '
         'compared to 2022 ')
#st.dataframe(data=skills, width=None, height=None)

#st.write('The Data is been displayed to highlight(in yellow, the maximum skills in the set of skills listed')

X = skills.drop('skill_group_rank', axis=1)
y = skills['skill_group_rank']

from sklearn.model_selection import train_test_split
#spliting the data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)
#X_train.shape, X_test.shape, y_train.shape, y_test.shape


#changing the data type of some columns to int
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
OHE_model = OneHotEncoder(handle_unknown='ignore')
categorical_columns = ['isic_index', 'isic_section_name','industry_name',
                       'skill_group_category', 'skill_group_name']
encoder = OneHotEncoder(handle_unknown='ignore')
transformer = ColumnTransformer(transformers = [('OneHotEncoder', encoder,categorical_columns),
                                                ('scaler', scaler, ['year'])],
                                remainder='passthrough')

t_X_train = transformer.fit_transform(X_train)
t_X_test = transformer.transform(X_test)
t_X_train

#transform to pd.dataframe
trains = pd.DataFrame(t_X_train)
#trains.head(10)

#t_X_train.dtype

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(t_X_train, y_train)
model.score(t_X_test, y_test)


chart_data = pd.DataFrame(skills, columns=['skill_group_rank', 'year'])
st.line_chart(chart_data)

st.sidebar.write('Code on Github repository')
st.sidebar.write('https://github.com/Superb25/Amina_ADS_proj/blob/main/ADS%20PROJECT.ipynb')

# creating a function for prediction
def skill_rank_prediction():
    y_train_pred = load_model.predict(t_X_test)
    y_train_pred
    if (y_train_pred[0] == 0):
        return 'y_train_pred'
    else:
        return 'skill not ranked'

def main():

    st.title('Skill Rank Prediction App')
    # giving data input
    year = st.text_input('Enter year')
    isic_section_name = st.text_input('Enter isic_section_name')
    skill_group_category = st.text_input('Enter skill_group_category')

    # code for prediction
    skill_pred = ''

    # create a prediction button
    if st.button('Skill_rank_prediction'):
        skill_pred = skill_rank_prediction()

    st.success(skill_pred)


if __name__ == '__main__':
    main()


st.text_input('Enter the skills you think are most needed in the public sector in 2022')
st.text_area('Why did you chose the skill above', max_chars=100, placeholder='Enter Reasons here: ')

skills_new = skills[['year','skill_group_category','skill_group_rank']]
skill_rank = skills[['year','skill_group_rank']]

st.sidebar.title('Public Skills Analysis')
menu_options = ['Skill_Display', 'Datasets', 'skill_rank']
selection = st.sidebar.selectbox('Public_skills_2022: ', menu_options)
skill_year_rank = skills.skill_group_category.unique()


#st.sidebar.line_chart(skill_year_rank)

if selection == 'Skill_Display':
    st.caption('Display skill criteria')
    st.dataframe(skills_new)

if selection == 'Datasets':
    st.caption('Display skill_cat')
    st.sidebar.dataframe(skills)
    #st.dataframe(skill_year_rank)


if selection == 'skill_rank':
    st.caption('Display skill_rank')
    st.sidebar.dataframe(skill_year_rank)
    #st.dataframe(skill_year_rank)
