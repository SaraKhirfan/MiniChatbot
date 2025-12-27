from flask import Flask, render_template, request, session, redirect, url_for
from main import get_answer

app = Flask(__name__)
app.secret_key = "study_buddy_ultra_secret" # Essential for sessions

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_question = request.form["question"]
        answer = get_answer(user_question)
        
        # Save to history
        history = session["chat_history"]
        history.append({"question": user_question, "answer": answer})
        session["chat_history"] = history
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/clear")
def clear():
    session.pop("chat_history", None) # Remove the chat history
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)