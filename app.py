from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_pGeneral_Health():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            General_Health = request.form.get('General_Health'),
            Checkup	 = request.form.get('Checkup'),
            Exercise = request.form.get('Exercise'),
            Skin_Cancer = request.form.get('Skin_Cancer'),
            Other_Cancer = request.form.get('Other_Cancer'),
            Depression = request.form.get('Depression'),
            Diabetes = request.form.get('Diabetes'),
            Arthritis = request.form.get('Arthritis'),
            Sex = request.form.get('Sex'),
            Age_Category = request.form.get('Age_Category'),
            Height = float(request.form.get('Height')),
            Weight = float(request.form.get('Weight')),
            BMI = float(request.form.get('BMI')),
            Smoking_History = request.form.get('Smoking_History'),
            Alcohol_Consumption = float(request.form.get('Alcohol_Consumption')),
            Fruit_Consumption = float(request.form.get('Fruit_Consumption')),
            Green_Vegetables_Consumption = float(request.form.get('Green_Vegetables_Consumption')),
            FriedPotato_Consumption = float(request.form.get('FriedPotato_Consumption'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=pred
        if result == 0:
            return render_template("result.html",final_result = "you don't have Heart Disease")
        elif result == 1:
            return render_template("result.html",final_result = "you have Heart Disease")






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
