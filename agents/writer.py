# Diğer backendcinin eksik dosyası (Mock testi için geçici olarak oluşturuldu)
class AcademicWriter:
    def __init__(self, llm=None):
        self.llm = llm

    def generate_section(self, **kwargs):
        return {"text": "Yazı taslağı", "metadata": {}}
    