from tranformers import MarianMTModel,MarianTokenizer
Model_name='Helsinki-NLP/opus-mt-en-de'
tokinzer=MarianTokenizer.from_pretranined(Model_name)
model=MarianMTModel.from_pretranined(Model_name)
def Translate(text:str,source_lang:str="en",target_lang:str="wo")