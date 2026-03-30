# Zadanie 1: Wybór tematu objętego wspomaganiem podejmowania decyzji

**Temat:** Inteligentny system wspomagania decyzji w zakresie diagnozowania i ochrony roślin przed chorobami.

**Opis problemu i zakres wspomagania decyzji:**
Zbyt późne lub błędne rozpoznanie choroby rośliny uprawnej (np. pomidora, ziemniaka czy jabłoni) często prowadzi do znacznej utraty plonów oraz nadużywania niewłaściwych środków chemicznych, co jest kosztowne i szkodliwe dla środowiska. Opracowywany system będzie wspomagał rolników oraz amatorskich ogrodników w podejmowaniu decyzji o doborze odpowiedniego środka ochrony roślin (np. fungicydu, insektycydu) lub działań prewencyjnych, opierając się na wizualnych objawach na liściach.

**Źródło danych:**
Źródłem danych będą cyfrowe zdjęcia liści roślin (w formacie JPG lub PNG) bezpośrednio wgrywane do systemu przez użytkownika w czasie rzeczywistym.

**Metoda sztucznej inteligencji i jej zadanie:**
Aby system był nowoczesny i wysoce skuteczny, jako metoda sztucznej inteligencji zastosowana zostanie **konwolucyjna sieć neuronowa (CNN) / Vision Transformer**.

Zadaniem modelu AI będzie klasyfikacja dostarczonego obrazu, czyli rozpoznanie ze zdjęcia gatunku rośliny i konkretnej jednostki chorobowej z przypisanym poziomem prawdopodobieństwa (ang. *confidence score*). Wynik ten będzie następnie przekazywany do algorytmu wnioskującego.

przykładowy zbiór danych.
https://www.kaggle.com/datasets/emmarex/plantdisease
