from flask import Flask , redirect, url_for,render_template
app= Flask(__name__)

@app.route("/")
def hello_word():
    return "<h1>Thái Viết Lập</h1>"

if __name__=="__main__":
    app.run(debug=True)