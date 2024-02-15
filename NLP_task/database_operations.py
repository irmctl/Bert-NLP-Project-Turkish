#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ad-Soyad:İrem ÇATAL
# Son Revize Tarihi: 15.02.2024


# In[2]:


import sqlite3


# In[3]:


conn = sqlite3.connect('nlp_project.db')
c = conn.cursor()


# In[4]:


c.execute('''
          CREATE TABLE IF NOT EXISTS texts
          ([generated_id] INTEGER PRIMARY KEY, [text] text, [main_topic] text)
          ''')


# In[5]:


def add_text_to_db(text, main_topic):
    c.execute('''
              INSERT INTO texts (text, main_topic)
              VALUES (?, ?)
              ''', (text, main_topic))
    conn.commit()


# In[6]:


def get_texts_from_db(main_topic=None):
    if main_topic:
        c.execute('''
                  SELECT * FROM texts WHERE main_topic = ?
                  ''', (main_topic,))
    else:
        c.execute('''
                  SELECT * FROM texts
                  ''')
    return c.fetchall()


# In[ ]:





# In[ ]:




