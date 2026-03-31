from flask import Flask, request, render_template
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from openai import OpenAI
import os

# 🔑 API KEY (dán key của bạn vào đây hoặc dùng env)
os.environ["OPENAI_API_KEY"] = "sk-xxxx"

client = OpenAI()

app = Flask(__name__)

# 📚 Lấy nội dung từ Wikipedia
def get_wiki_text(title):
    try:
        wikipedia.set_lang("vi")
        page = wikipedia.page(title)
        return page.content
    except:
        return None

# 🧠 Tóm tắt bằng TF-IDF
def summarize_tfidf(text, n=3):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s) > 20]

    if len(sentences) < n:
        return "Không đủ dữ liệu để tóm tắt."

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    scores = X.sum(axis=1)
    ranked = np.argsort(scores, axis=0).flatten()[::-1]

    top_sentences = [sentences[i] for i in ranked[:n]]
    return '. '.join(top_sentences)

# 🤖 Tóm tắt bằng AI (fallback)
def summarize_ai(name):
    try:
        res = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": f"Tóm tắt nội dung truyện {name} trong 3-5 câu"}
            ]
        )
        return res.choices[0].message.content
    except:
        return "Lỗi khi gọi AI API."

# 🔁 Logic chính
def summarize_story(name):
    text = get_wiki_text(name)

    if text:
        return summarize_tfidf(text)
    else:
        return summarize_ai(name)

# 🌐 Route web
@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        name = request.form.get('name')
        if name:
            result = summarize_story(name)
        else:
            result = "Vui lòng nhập tên truyện."
    return render_template("index.html", result=result)

# 🚀 Run app
if __name__ == "__main__":
    app.run(debug=True)
