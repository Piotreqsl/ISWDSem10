# Zadanie 2: Opracowanie koncepcji inteligentnego systemu wspomagania podejmowania decyzji

**Analizowany problem i zakres podejmowanych decyzji:**
Zasadniczym problemem polegającym analizie w systemie jest automatyczna, bezbłędna identyfikacja patogenów roślinnych na najwcześniejszym możliwym etapie na podstawie wizualnych objawów. Zakres podejmowanych decyzji dotyczy optymalnego wyboru konkretnej kuracji dla rośliny, jej dawkowania oraz określenia działań opryskowych. Zamiast ręcznej analizy atlasów chorób, system odciąża użytkownika, rekomendując konkretne działanie zaradcze na bazie zdiagnozowanego przez AI problemu.

**Przetwarzane dane i sposób ich przechowywania/przetwarzania:**
1. **Dane wejściowe:** Zdjęcia uszkodzonych/chorych liści wgrywane do aplikacji przez użytkownika. Obrazy będą przetwarzane w pamięci ulotnej (RAM) systemu w czasie rzeczywistym, bez konieczności ich trwałego gromadzenia na serwerze (dla zapewnienia szybkości i prywatności).
2. **Dane analityczne:** Wynik detekcji modelu AI – zaklasyfikowana kategoria choroby wraz ze współczynnikiem pewności (prawdopodobieństwa).
3. **Baza wiedzy:** Gromadzi zalecenia eksperckie. Przechowywana będzie jako wydajny, strukturalny plik (np. JSON / baza ustrukturyzowana wewnętrznie), zawierający tabele mapujące rozpoznaną chorobę z rekomendowanymi środkami ochrony (nazwa, dawka, czas aplikacji).

**Moduły projektowanego systemu:**
1. **Moduł Interfejsu Użytkownika (UI):** Responsywna aplikacja webowa pozwalająca swobodnie wgrać zdjęcie (tzw. "drag & drop") i prezentująca finalną, uargumentowaną diagnozę.
2. **Moduł AI (Wizja Komputerowa):** Wykorzystuje pobrany, gotowy i zoptymalizowany model klasyfikacji obrazu. Stanowi rdzeń "diagnostyczny", odpowiadający za rozpoznanie wizualne obrazów i zwrócenie ustrukturyzowanej etykiety choroby (np. *Tomato___Late_blight*).
3. **Moduł Bazy Wiedzy:** Elektroniczny słownik reguł wiążący zidentyfikowane klasy chorób z fizycznymi nazwami preparatów ratunkowych.
4. **Moduł Algorytmu Wnioskowania (Silnik Regułowy):** Mechanizm decyzyjny pośredniczący między modułem AI a Bazą Wiedzy. Implementuje reguły (logikę) generujące ostateczną rekomendację.

**Zaimplementowany algorytm wnioskowania (Silnik Regułowy - metodyka IF-THEN):**
Proces decyzyjny będzie prowadzony w oparciu o zestaw heurystyk oceniających m.in. pewność modelu:
*   **JEŚLI** Moduł AI rozpozna chorobę *Choroba_X* z poziomem pewności > 85% **ORAZ** w Bazie Wiedzy figuruje dedykowana kuracja, **WTEDY** system kategorycznie zaleca kurację.
*   **JEŚLI** pewność detekcji jest w przedziale 60% - 85%, **WTEDY** system proponuje rozwiązanie uniwersalne, podając diagnozę w wątpliwość ("Zalecana konsultacja uzupełniająca z doradcą rolniczym - możliwa choroba X").
*   **JEŚLI** Moduł AI ma pewność < 60%, **WTEDY** algorytm blokuje decyzję o użyciu silnej chemii z uwagi na ryzyko ("Zbyt niska pewność detekcji, proszę o dodanie wyraźniejszego zdjęcia").

**Metody wykonania / Technologie:**
System zostanie wykonany w środowisku **Python**. Ze względu na nacisk na element AI, wdrożona zostanie łatwa w obsłudze technologia sztucznej inteligencji wykorzystująca biblioteki klasyfikacyjne ze strony Hugging Face.

---

### 📝 Architektura i Plan Implementacji (Co dokładnie będziemy robić w kolejnych krokach)

Aby spełnić wymóg prostoty konstrukcji, ale przy tym dowieźć w pełni działający system, kolejne etapy projektu można zrealizować w oparciu o poniższy – bardzo wygodny – plan technologiczny:

1. **AI Model API (Gotowe rozwiązanie):** Zamiast pracochłonnego uczenia i trenowania sieci neuronowych, skorzystamy z biblioteki `transformers` (moduł `pipeline`), by jednym poleceniem pobrać i wykorzystać gotowy do działania model ze świetnie udokumentowanej platformy **Hugging Face** (np. `dima806/plant_disease_detection` wytrenowany na słynnym zbiorze *PlantVillage*).
2. **Oparcie logiki o Baze Wiedzy (JSON):** Przygotujemy niewielki plik `knowledge_base.json`, wpisując w nim kilkanaście popularnych chorób w formie słownika, np:
   *`"Tomato___Early_blight": "Użyj preparatu miedziowego (np. Miedzian 50 WP) w dawce 3g na 1 litr wody."`*
3. **Konstrukcja systemu u Algorytmu:** Przy użyciu języka Python napiszemy zwykłą funkcję realizującą algorytm wnioskowania opisany powyżej (instrukcje `if / else`).
4. **Stworzenie ładnego Panelu Front-end:** Użyjemy w 100% darmowej biblioteki **Streamlit** w Pythonie. Pozwala ona w kilkunastu linijkach kodu stworzyć bardzo nowoczesną i przejrzystą aplikację webową, która przyjmie od użytkownika zdjęcie, odpyta model Hugging Face, połączy z bazą wiedzy JSON i wypluje ładnie sformatowaną diagnozę. To najbardziej efektowny sposób na tego typu zaliczenia!
