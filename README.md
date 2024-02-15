# Bert-NLP-Project-Turkish

# Yapay Zeka Türkçe Metin Sınıflandırma ve Bilgi Sunma Projesi

## Proje Hakkında
Bu proje, doğal dil işleme tekniklerini kullanarak Türkçe metinleri sınıflandırmak ve sınıflandırılan kategoriye göre internet üzerinden bilgi toplayarak kullanıcıya sunmak için tasarlanmış bir yapay zeka uygulamasıdır.

## Özellikler
- Metin Sınıflandırma
- Sınıflandırılan metnin genel konusuna göre bilgi toplama
- Sonuçların bir SQLite veritabanında saklanması
- Metin girişi için basit bir kullanıcı arayüzü (GUI)
- Sınıflandırılan konu hakkında OpenAI API kullanarak internetten bilgi toplar.
- Toplanan bilgileri kullanıcıya arayüz üzerinden sunar.


## Dosyalama
Proje üç ana bölümden oluşmaktadır: NLP Model Eğitimi, Kullanıcı Arayüzü ve Veritabanı İşlemleri.

### Dataset
Sınıflandırma için Kaggle üzerinden alınan https://www.kaggle.com/datasets/savasy/ttc4900 veriseti kullanılmıştır.

### NLP Model Eğitimi
- Hugging Face'e ait `transformers` kütüphanesi kullanılarak BERT modeli (`dbmdz/bert-base-turkish-128k-uncased`) ile metin sınıflandırma gerçekleştirilmektedir.
- Model, belirlenen kategorilere göre metinleri sınıflandırmak üzere eğitilmiştir.
- Eğitim ve test süreçleri için gerekli veri seti `topics.csv` dosyasından alınmaktadır.

### Kullanıcı Arayüzü (Interface)
- Tkinter kütüphanesi kullanılarak oluşturulan GUI, kullanıcının metin girişi yapmasını ve sınıflandırma sonuçlarını görmesini sağlar.
- `classify_text` fonksiyonu, kullanıcının girdiği metni sınıflandırır ve `get_info_on_topic` fonksiyonu ile sınıflandırılan kategoriye göre bilgi toplar.

### OpenAI API ile Bilgi Toplama

Projede, OpenAI'nin GPT-3 API'si kullanılarak metin sınıflandırma sonrasında elde edilen konu hakkında bilgi toplama işlemi gerçekleştirilir. Bu işlem şu adımları içerir:

1. Kullanıcıdan alınan metnin konusu sınıflandırılır.
2. Sınıflandırılan konu, OpenAI API'sine bir istek olarak gönderilir.
3. OpenAI API, gelişmiş doğal dil işleme modelleri kullanarak konu hakkında bilgi toplar.
4. API'den alınan yanıt, kullanıcıya sunulur.

### Veritabanı İşlemleri (Database)
- SQLite kullanılarak oluşturulan `nlp_project.db` veritabanı, kullanıcı girişlerini ve sınıflandırma sonuçlarını saklar.

### Kaynaklar
- https://medium.com/swlh/understand-tweets-better-with-bert-sentiment-analysis-3b054c4b802a
- https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
- https://mccormickml.com/2019/07/22/BERT-fine-tuning/
- https://github.com/ayyucekizrak/BERTileSentimentAnaliz/blob/master/BERT_ile_SentimentAnaliz/BERT_ile_Sentiment_Analiz.ipynb
- https://medium.com/@ktoprakucar/bert-modeli-ile-t%C3%BCrk%C3%A7e-metinlerde-s%C4%B1n%C4%B1fland%C4%B1rma-yapmak-260f15a65611
- https://github.com/melihbodur/Text_and_Audio_classification_with_Bert
- https://platform.openai.com/docs/introduction

# [English Below]

# Artificial Intelligence Turkish Text Classification and Information Presentation Project

## About the Project
This AI application is designed to classify Turkish texts using natural language processing techniques and present information gathered from the internet based on the classified categories.

## Features
- **Text Classification**: Automated classification of texts into specific categories.
- **Information Gathering**: Collection of relevant information based on the classified text's subject.
- **Data Storage**: Storing of classification results and user inputs in a SQLite database.
- **User Interface**: A simple graphical user interface (GUI) for easy text input and result display.
- **OpenAI API Integration**: Utilizing OpenAI API for extensive internet-based information gathering about classified topics.
- **Presentation of Information**: Displaying the gathered information to the user through the GUI.

## Components
The project is divided into three main components: NLP Model Training, User Interface, and Database Operations.

### Dataset
- Dataset Source: [TTC4900 on Kaggle](https://www.kaggle.com/datasets/savasy/ttc4900)

### NLP Model Training
- Uses the `transformers` library from Hugging Face for text classification.
- Employs the BERT model (`dbmdz/bert-base-turkish-128k-uncased`).
- Training and testing are conducted using data from `topics.csv`.

### User Interface
- Developed using the Tkinter library.
- Functions:
  - `classify_text`: For classifying user-inputted text.
  - `get_info_on_topic`: For gathering information based on the classified category.

### Information Gathering with OpenAI API
- The project leverages OpenAI's GPT-3 API for post-classification information collection.
- Process Flow:
  1. Classification of the text topic.
  2. Sending classified topics to the OpenAI API.
  3. Collecting topic-related information using the API.
  4. Presenting the API responses to the user.

### Database Operations
- Utilizes SQLite to manage the `nlp_project.db` database.
- Stores user entries and classification outcomes.

### Sources
- https://medium.com/swlh/understand-tweets-better-with-bert-sentiment-analysis-3b054c4b802a
- https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
- https://mccormickml.com/2019/07/22/BERT-fine-tuning/
- https://github.com/ayyucekizrak/BERTileSentimentAnaliz/blob/master/BERT_ile_SentimentAnaliz/BERT_ile_Sentiment_Analiz.ipynb
- https://medium.com/@ktoprakucar/bert-modeli-ile-t%C3%BCrk%C3%A7e-metinlerde-s%C4%B1n%C4%B1fland%C4%B1rma-yapmak-260f15a65611
- https://github.com/melihbodur/Text_and_Audio_classification_with_Bert
- https://platform.openai.com/docs/introduction




