#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ad-Soyad:İrem ÇATAL
# Son Revize Tarihi: 15.02.2024


# In[4]:


get_ipython().system('pip install --upgrade ipywidgets')


# In[6]:


import tkinter as tk
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import warnings
from database_operations import add_text_to_db, get_texts_from_db
import openai


# In[7]:


tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

number_of_categories = 7  

model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-128k-uncased",
    num_labels=number_of_categories,
    output_attentions=False,
    output_hidden_states=False,
)

# Colab üzerindeki NLP kodunu eğitilmiş model ağırlıklarını yükleyerek projeye entegre ediyorum.
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

#Metnin sınıflandırılmasına göre OpenAI API ile kullanıcıya bilgi sunuluyor.

openai.api_key = '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'

def get_info_on_topic(topic):
    prompt = f"{topic} hakkında bilgi verin."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Model adını burada belirtin
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def classify_text():
    user_input = text_entry.get("1.0", tk.END)
    if not user_input:
        result_label.config(text="Lütfen metin giriniz.")
        return
    
    encoded_dict = tokenizer.encode_plus(
    user_input,
    add_special_tokens=True,
    max_length=250,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt',
)

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

    logits = outputs[0]
    predicted_label = torch.argmax(logits, dim=1).item()
    labels = ['dunya', 'ekonomi', 'kultur', 'saglik', 'siyaset', 'spor', 'teknoloji']
    predicted_category = labels[predicted_label]

    result_label.config(text=f"Tahmin edilen kategori: {predicted_category}")

    # Veritabanına bağlantısı
    add_text_to_db(user_input, predicted_category)
    
    info = get_info_on_topic(predicted_category)
    info_label.config(text=info)
    
root = tk.Tk()
root.title("Metin Sınıflandırma")

text_entry = tk.Text(root, height=10, width=50)
text_entry.pack()

classify_button = tk.Button(root, text="Konuyu bul", command=classify_text)
classify_button.pack()

result_label = tk.Label(root, text="Genel Konu: ")
result_label.pack()

info_label = tk.Label(root, wraplength=400)  
info_label.pack()

root.mainloop()


# In[ ]:




