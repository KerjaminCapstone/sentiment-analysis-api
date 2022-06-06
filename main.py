"""
    Libraries
"""
from flask import Flask
from flask_restful import reqparse
from run_nlp import NlpPredict

app = Flask(__name__)

"""
    Parser 
"""
parser = reqparse.RequestParser()
parser.add_argument("komentar", type=str, required=True, help="Komentar harus diisi")

"""
    Resouce 
"""
@app.route('/predict', methods=['POST'])
def predict():
    args = parser.parse_args()
    model = NlpPredict('nlp_model/sentimentanalysisv4.h5')
    model.set_sentence(args["komentar"])

    nlp_score = model.predict()
    return {"data": {
        "nlp_score": nlp_score
    }}, 200

"""
    Main 
"""
if __name__ == '__main__':
    app.run()