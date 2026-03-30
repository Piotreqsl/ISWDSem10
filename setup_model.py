import os
from transformers import AutoModelForImageClassification, AutoImageProcessor

MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
LOCAL_DIR = "./local_model"

def main():
    print(f"Pobieranie modelu '{MODEL_NAME}' do katalogu lokalnego '{LOCAL_DIR}'...")
    
    # Tworzenie katalogu
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # Inicjalizacja komponentów
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    
    # Zapis lokalny dla trybu offline
    processor.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)
    
    print("Model pomyślnie zapisany na dysku.")
    
    # Utworzenie pliku ze szczegółowym opisem modelu na potrzeby zadania/raportu
    arch_file = "model_architecture.txt"
    with open(arch_file, "w") as f:
        f.write(f"Nazwa Modelu: {MODEL_NAME}\n")
        f.write(f"Architektura bazowa: MobileNetV2\n")
        f.write("Pełny opis warstw modelu (przydatny w podsumowaniu do dokumentacji projektu):\n\n")
        f.write(str(model))
        
    print(f"Architektura sieci wyrzucona do pliku '{arch_file}'!")

if __name__ == "__main__":
    main()
