from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import random

app = Flask(__name__)


tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-squad")
model = AutoModelForQuestionAnswering.from_pretrained("savasy/bert-base-turkish-squad")
nlp=pipeline("question-answering", model=model, tokenizer=tokenizer)


text="""Ankara, Türkiye'nin başkenti ve en kalabalık ikinci ilidir. Nüfusu 2020 itibarıyla 5.663.322 kişidir.
Bu nüfus; 25 ilçe ve bu ilçelere bağlı 1425 mahallede yaşamaktadır.
İl genelinde nüfus yoğunluğu 215'tir.
Coğrafi olarak Türkiye'nin merkezine yakın bir konumda bulunur ve Batı Karadeniz Bölgesi'nde kalan kuzey kesimleri hariç, büyük bölümü İç Anadolu Bölgesi'nde yer alır.
Yüz ölçümü olarak ülkenin üçüncü büyük ilidir.
Bolu, Çankırı, Kırıkkale, Kırşehir, Aksaray, Konya ve Eskişehir illeri ile çevrilidir.
Anıtkabir, Koç Müzesi, Ankara Kalesi, Roma Hamamı, Kızılay Meydanı ve Hamamönü gezilmesi gereken yerlerdir.
İlin güney ve orta bölümlerinde karasal iklimin soğuk ve kar yağışlı kışları ile sıcak ve kurak yazları, kuzeyinde ise Karadeniz ikliminin ılıman ve yağışlı halleri görülebilir. 
Karasal iklimin hâkim olduğu bölgelerde gece ile gündüz, yaz ile kış mevsimi arasında önemli sıcaklık farkları bulunur. En sıcak ay temmuz veya ağustostur. İldeki yerine göre ortalama en yüksek gündüz sıcaklıkları 27-31° C'dir. 
En soğuk ay ise Ocak ayıdır, en düşük gece sıcaklıkları ildeki yerine göre ortalama -6 ila -1 °C arasındadır. Yağışlar en çok mayıs, en az temmuz veya ağustos ayında düşer. 
Ankara il merkezinde yıllık ortalama toplam yağış 415 mm, yıllık ortalama toplam yağış, 
60 cm (Kızılcahamam) ila 35 cm (Şereflikoçhisar) arasında değişir. Son yılların en soğuk gecesini -22 ile 26 Ocak 2016'da gördü."""

question_user=""

negative1 = ['Zor bi soru oldu sanırım. Ama cevabım, ','Cevap vemek biraz zorladı ama sanırım cevabım bu. ','Şasırtmacalı sorumu sordun emin olamadım. Ama bildiklerim: ']

negative2 = ['Senin için bir sonuç buldum ama pek emin değilim.','Galiba bir terslik oldu. Pek güzel cevap bulamadım bu sefer.','Zor bi sorumu sordun? Cevap vermek pek kolay olmadı.']

positive1 = ['Tam da çalıştığım yerden sordun. İşte bildiklerim: ','Bence bu soruya bu cevap tam uyacaktır.Sorunun cevabı, ','Sanırım ben bu soruyu çözmek için geliştirilmişim . İşte cevabım, ']

positive2 = ['Güzel bi soru sordun. İşte cevabım: ','Kendimden çok emin bir şekilde söylemeliyim ki ','Kesinlikle ']


result=""
def app_answer(score, answer):
  if score >= 0.90:
    result = random.choice(positive2) + answer
  elif score > 0.60:
    result = random.choice(positive1) + answer
  elif score > 0.30:
    result = random.choice(negative1) + answer
  elif score > 0.00 :
    result = random.choice(negative2) + answer
  else:
    result = "Error!"
  return result



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    full_answer=nlp(question=userText, context=text)
    answer = full_answer['answer']
    score = full_answer['score']
    return app_answer(float(score), answer)


if __name__ == "__main__":
    app.run()
    
#The day you stop learning is the day you begin decaying.
