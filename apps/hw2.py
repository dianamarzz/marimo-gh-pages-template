import marimo

__generated_with = "0.18.0"
app = marimo.App(auto_download=["ipynb"])


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Data loading & preprocessing
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('insurance_dataset.csv')
    df.columns = [s.strip().replace(' ', '_') for s in df.columns]
    return (df,)


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df, pd):
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
            'missing_count': missing_data,
            'missing_percent': missing_percent
        })
    missing_df
    return


@app.cell
def _(df):
    df['medical_history'] = df['medical_history'].fillna('No Record')
    df['family_medical_history'] = df['family_medical_history'].fillna('No Record')
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    return LinearRegression, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One-hot encoding
    """)
    return


@app.cell
def _(df, pd):
    def preprocess_insurance_data(df):
        df1 = df.copy()

        categorical_columns = ['smoker', 'region', 'medical_history',
                              'family_medical_history', 'exercise_frequency',
                              'occupation', 'coverage_level']

        for col in categorical_columns:
            df1[col] = df1[col].astype('category')

        df1 = pd.get_dummies(df1, prefix_sep='_', drop_first=True)

        return df1


    df1 = preprocess_insurance_data(df)
    df1
    return (df1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Creating list of predictors and outcoms
    """)
    return


@app.cell
def _(df1):
    excludeColumns = ('charges')
    predictors = [column for column in df1.columns if column not in excludeColumns]
    outcome = 'charges'
    return outcome, predictors


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Partition data
    """)
    return


@app.cell
def _(df1, outcome, predictors, train_test_split):
    X = df1[predictors]
    y = df1[outcome]
    train_X, holdout_X, train_y, holdout_y = train_test_split(
    X, 
    y, 
     test_size=0.4, 
     train_size=0.6, 
     random_state=1, 
     shuffle=True, 
    )
    return holdout_X, holdout_y, train_X, train_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fitting model
    """)
    return


@app.cell
def _(LinearRegression):
    model = LinearRegression()
    return (model,)


@app.cell
def _():
    from sklearn.model_selection import cross_validate
    return (cross_validate,)


@app.cell
def _(cross_validate, model, train_X, train_y):
    scoring = {'neg_RMSE': 'neg_root_mean_squared_error',
     'neg_MAE': 'neg_mean_absolute_error'}
    scores = cross_validate(model, train_X, train_y, cv=5, scoring=scoring)
    return (scores,)


@app.cell
def _(scores):
    scores
    return


@app.cell
def _(model, train_X, train_y):
    model.fit(train_X, train_y)
    return


@app.cell
def _(model, pd, train_X, train_y):
    train_pred = model.predict(train_X)
    train_results = pd.DataFrame({
        'charges': train_y,
        'predicted': train_pred,
        'residual': train_y - train_pred,
    })
    train_results.head()
    return (train_results,)


@app.cell
def _(holdout_X, holdout_y, model, pd):
    holdout_pred = model.predict(holdout_X)
    holdout_results = pd.DataFrame({
        'charges': holdout_y,
        'predicted': holdout_pred,
        'residual': holdout_y - holdout_pred,
    })
    holdout_results.head()
    return (holdout_results,)


@app.cell
def _():
    from mlba import regressionSummary
    return (regressionSummary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Training set
    """)
    return


@app.cell
def _(regressionSummary, train_results):
    print("\nTraining Set")
    regressionSummary(y_true=train_results.charges, y_pred=train_results.predicted)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Holdout set
    """)
    return


@app.cell
def _(holdout_results, regressionSummary):
    print("\nHoldout Set")
    regressionSummary(y_true=holdout_results.charges, y_pred=holdout_results.predicted)
    return


@app.cell
def _(mo):
    age = mo.ui.slider(start=1, stop=120, step=1, value=5, label="age")
    bmi = mo.ui.slider(start=1, stop=64, step=0.1, value=5, label="bmi")
    children = mo.ui.slider(start=1, stop=20, step=1, value=5, label="children")
    smoker = mo.ui.dropdown(options=['yes', 'no'], value='no',label='smoker')
    gender = mo.ui.dropdown(options=['male', 'female'], value='male',label='gender')
    region = mo.ui.dropdown(options=['northeast', 'northwest', 'southeast', 'southwest'], value='northeast', label='region')
    medical_history = mo.ui.dropdown(options=['Diabetes', 'Heart disease', 'High blood pressure', 'No Record'], value='No Record', label='medical_history')
    family_medical_history = mo.ui.dropdown(options=['Diabetes', 'Heart disease', 'High blood pressure', 'No Record'], value='No Record', label='family_medical_history')
    exercise_frequency = mo.ui.dropdown(options=['Never', 'Occasionally', 'Rarely'], value='Occasionally',label='exercise_frequency')
    occupation = mo.ui.dropdown(options=['Blue collar', 'Student', 'Unemployed', 'White collar'], value='White collar', label='occupation')
    coverage_level = mo.ui.dropdown(options=['Basic', 'Premium', 'Standard'], value='Standard', label='coverage_level')
    return (
        age,
        bmi,
        children,
        coverage_level,
        exercise_frequency,
        family_medical_history,
        gender,
        medical_history,
        occupation,
        region,
        smoker,
    )


@app.cell
def _(
    age,
    bmi,
    children,
    coverage_level,
    exercise_frequency,
    family_medical_history,
    gender,
    medical_history,
    occupation,
    pd,
    region,
    smoker,
):
    data = {
            'age': [age.value],
            'bmi': [bmi.value],
            'children': [children.value]
        }
    
    new_data = pd.DataFrame(data)

    new_data['gender_male'] = 1 if gender.value == "male" else 0
    
    new_data['smoker_yes'] = 1 if smoker.value == "yes" else 0
    
    new_data['region_northwest'] = 1 if region.value == "northwest" else 0
    new_data['region_southeast'] = 1 if region.value == "southeast" else 0
    new_data['region_southwest'] = 1 if region.value == "southwest" else 0
    
    new_data['medical_history_Heart disease'] = 1 if medical_history.value == "Heart disease" else 0
    new_data['medical_history_High blood pressure'] = 1 if medical_history.value == "High blood pressure" else 0
    new_data['medical_history_No Record'] = 1 if medical_history.value == "No Record" else 0
    
    new_data['family_medical_history_Heart disease'] = 1 if family_medical_history.value == "Heart disease" else 0
    new_data['family_medical_history_High blood pressure'] = 1 if family_medical_history.value == "High blood pressure" else 0
    new_data['family_medical_history_No Record'] = 1 if family_medical_history.value == "No Record" else 0
    
    new_data['exercise_frequency_Never'] = 1 if exercise_frequency.value == "Never" else 0
    new_data['exercise_frequency_Occasionally'] = 1 if exercise_frequency.value == "Occasionally" else 0
    new_data['exercise_frequency_Rarely'] = 1 if exercise_frequency.value == "Rarely" else 0
    
    new_data['occupation_Student'] = 1 if occupation.value == "Student" else 0
    new_data['occupation_Unemployed'] = 1 if occupation.value == "Unemployed" else 0
    new_data['occupation_White collar'] = 1 if occupation.value == "White collar" else 0
    
    new_data['coverage_level_Premium'] = 1 if coverage_level.value == "Premium" else 0
    new_data['coverage_level_Standard'] = 1 if coverage_level.value == "Standard" else 0

    new_data
    return (new_data,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(model, new_data):
    new_data['pred_charges'] = model.predict(new_data)
    new_data
    return


@app.cell
def _(
    age,
    bmi,
    children,
    coverage_level,
    exercise_frequency,
    family_medical_history,
    gender,
    medical_history,
    occupation,
    region,
    smoker,
):
    age, bmi, children, smoker, gender, region, medical_history, family_medical_history, exercise_frequency, occupation, coverage_level
    return


@app.cell
def _(mo, new_data):
    value = new_data["pred_charges"].iloc[0]
    mo.md(f"## Estimated insurance premium: **${value:,.2f}**")
    return


if __name__ == "__main__":
    app.run()
