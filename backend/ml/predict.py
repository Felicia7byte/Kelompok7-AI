import pandas as pd
import joblib

ingredient = 12
instruction_len = 950

model = joblib.load(r"D:\react-native-recipe-app-master\backend\ml\meal_classifier.pkl")

df_input = pd.DataFrame([[ingredient, instruction_len]], columns=['ingredient_count', 'instruction_length'])

prediction = model.predict(df_input)

if prediction[0] == 1:
    print("Makanan berat")
else:
    print("Makanan ringan")
