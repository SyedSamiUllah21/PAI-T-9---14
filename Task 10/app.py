from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def get_bot_response(user_input):
    user_input = user_input.lower()
    if "menu" in user_input:
        return "We serve pizza, pasta, burgers, and salads."
    elif "time" in user_input or "hours" in user_input:
        return "We are open from 10 AM to 10 PM every day."
    elif "location" in user_input:
        return "We're located at 123 Foodie Street, Flavor Town."
    elif "contact" in user_input or "phone" in user_input:
        return "You can contact us at (123) 456-7890."
    elif "hello" in user_input or "hi" in user_input:
        return "Hello! How can I assist you today?"
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"

@app.route("/")
def index():
    return render_templete("index.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form["msg"]
    response = get_bot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
