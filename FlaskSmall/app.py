from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    text_list = processed_text.split(',')
    text_list.sort()
    return text_list[0]

if __name__ == "__main__":
    app.run(debug=True)