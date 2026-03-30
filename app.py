import os
import json
import torch
import customtkinter as ctk
import threading
from tkinter import filedialog
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Bezpieczny import lokalnego wrappera LLM
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# Konfiguracja wyglądu okna
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

LOCAL_MODEL_DIR = "./local_model"
FALLBACK_MODEL_NAME = "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease"
KNOWLEDGE_BASE_PATH = "knowledge_base.json"
BIELIK_LLM_PATH = "./local_llm/Bielik-11B-v2.2-Instruct.Q4_K_M.gguf"

class PlantDiseaseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Inteligentny Detektor Chorób Roślin (ISWD) - UI Panelowe")
        self.geometry("1050x700")

        # Modele
        self.processor = None
        self.model = None
        self.kb = {}
        
        # Opcja 3 - System Ratunkowy
        self.fallback_processor = None
        self.fallback_model = None
        
        # Silnik Lokalnego LLM (Bielik_GGUF)
        self.llm_engine = None
        self.thinking = False
        
        # Przygotowanie głównej siatki okna
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Nagłówek
        self.title_label = ctk.CTkLabel(self, text="System Wsparcia dla rolników", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(15, 5))

        # 2. Narzędziówka (Przyciski)
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=1, column=0, padx=20, pady=5)

        self.load_model_btn = ctk.CTkButton(self.btn_frame, text="Zainicjalizuj Modele", command=self.load_models)
        self.load_model_btn.pack(side="left", padx=10)

        self.load_image_btn = ctk.CTkButton(self.btn_frame, text="Załaduj Zdjęcie do analizy", command=self.load_image, state="disabled")
        self.load_image_btn.pack(side="left", padx=10)

        # 3. Kafelki Układu Głównego
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1) # Lewa sekcja (Obraz)
        self.content_frame.grid_columnconfigure(1, weight=2) # Prawa sekcja (Panele AI, Status, Tekst)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # === LEWA SEKCJA (Obraz) ===
        self.left_panel = ctk.CTkFrame(self.content_frame)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.image_label = ctk.CTkLabel(self.left_panel, text="Brak wybranego zdjęcia", width=350, height=350, fg_color="gray20", corner_radius=10)
        self.image_label.pack(expand=True)

        # === PRAWA SEKCJA (Modułowa) ===
        self.right_panel = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_panel.grid_rowconfigure(0, weight=0) # Diagnostyka AI (stałe)
        self.right_panel.grid_rowconfigure(1, weight=0) # Status Operacyjny (stałe)
        self.right_panel.grid_rowconfigure(2, weight=1) # Rekomendacje (zajmujeresztę)
        self.right_panel.grid_columnconfigure(0, weight=1)

        # -- KAFELEK 1: DIAGNOSTYKA SIECI NEURONOWEJ --
        self.ai_panel = ctk.CTkFrame(self.right_panel, fg_color="gray15")
        self.ai_panel.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        
        self.ai_title = ctk.CTkLabel(self.ai_panel, text="Analiza Wizyjna", font=ctk.CTkFont(size=14, weight="bold"), text_color="cadetblue")
        self.ai_title.pack(anchor="w", padx=15, pady=(10, 0))
        
        self.pred_label = ctk.CTkLabel(self.ai_panel, text="Kategoria : --", font=ctk.CTkFont(size=18, weight="bold"))
        self.pred_label.pack(pady=(10, 0), padx=15, anchor="w")
        
        self.conf_label = ctk.CTkLabel(self.ai_panel, text="Pewność: --%", font=ctk.CTkFont(size=14))
        self.conf_label.pack(pady=(5, 5), padx=15, anchor="w")
        
        self.progress_bar = ctk.CTkProgressBar(self.ai_panel, width=300)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=(0, 15), padx=15, anchor="w")

        # -- KAFELEK 2: Dziennik Operacyjny (Log Zdarzeń) --
        self.status_panel = ctk.CTkFrame(self.right_panel, fg_color="gray10", border_width=1, border_color="gray30")
        self.status_panel.grid(row=1, column=0, pady=(0, 10), sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.status_panel, text="Gotowy do pracy.", font=ctk.CTkFont(size=13, slant="italic"), text_color="gray70")
        self.status_label.pack(padx=15, pady=8, anchor="w")

        # -- KAFELEK 3: REKOMENDACJE (LLM LUB JSON) --
        self.recom_panel = ctk.CTkFrame(self.right_panel, fg_color="gray15")
        self.recom_panel.grid(row=2, column=0, sticky="nsew")
        
        self.recom_title = ctk.CTkLabel(self.recom_panel, text="Rekomendacja", font=ctk.CTkFont(size=14, weight="bold"), text_color="#FFCC00")
        self.recom_title.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Ogromny i czysty textbox
        self.action_text = ctk.CTkTextbox(self.recom_panel, wrap="word", fg_color="gray20", font=ctk.CTkFont(size=14))
        self.action_text.pack(expand=True, fill="both", padx=15, pady=(0, 15))
        self.action_text.insert("0.0", "Czekam na uruchomienie procesu...")
        self.action_text.configure(state="disabled")
        
        # Wczytanie Bazy Wiedzy
        self.load_knowledge_base()

    def set_status(self, text, color="gray70"):
        self.status_label.configure(text=text, text_color=color)

    def load_knowledge_base(self):
        try:
            with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
                self.kb = json.load(f)
        except Exception as e:
            print(f"Błąd ładowania KB: {e}")

    def load_models(self):
        self.load_model_btn.configure(text="Ładowanie ...", state="disabled")
        self.set_status("Uruchamianie ....", color="yellow")
        self.update()
        
        try:
            # Ładujemy model detekcji główny offline (PlantVillage)
            if not os.path.exists(LOCAL_MODEL_DIR):
                target = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
            else:
                target = LOCAL_MODEL_DIR
                
            self.processor = AutoImageProcessor.from_pretrained(target)
            self.model = AutoModelForImageClassification.from_pretrained(target)
            
            # Ładujemy model Fallbackowy
            self.fallback_processor = AutoImageProcessor.from_pretrained(FALLBACK_MODEL_NAME)
            self.fallback_model = AutoModelForImageClassification.from_pretrained(FALLBACK_MODEL_NAME)
            
            # Inicjalizacja Bielika - jeśli pobrany
            if Llama and os.path.exists(BIELIK_LLM_PATH):
                self.load_model_btn.configure(text="Inicjowanie LLM...")
                self.update()
                # Ładujemy LLMa
                self.llm_engine = Llama(model_path=BIELIK_LLM_PATH, n_ctx=2048, verbose=False)
                self.load_model_btn.configure(text="Aktywne", fg_color="green")
                self.set_status("System gotowy.", color="lightgreen")
            else:
                self.load_model_btn.configure(text="Tylko Sieci Wizyjne", fg_color="orange")
                self.set_status("Ostrzeżenie: Praca bez agenta tekstowego.", color="orange")
                
            self.load_image_btn.configure(state="normal")
        except Exception as e:
            self.load_model_btn.configure(text="Błąd Inicjalizacji!", fg_color="red")
            self.set_status(f"FATAL BŁĄD: {e}", color="red")

    def load_image(self):
        if self.thinking:
            return 
            
        file_path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.jpeg *.png")])
        if file_path:
            img = Image.open(file_path).convert("RGB")
            # Przeskalowywanie do podglądu, używamy wymiarów by nie deformować GUI
            img_gui = ctk.CTkImage(light_image=img, size=(300, 300))
            self.image_label.configure(image=img_gui, text="")
            self.run_inference(img)

    def run_inference(self, image):
        self.set_status("Identyfikacja trwa...", color="lightblue")
        
        # 1. Inferencja - SILNIK GŁÓWNY
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
        confidence = top_prob.item() * 100
        predicted_label = self.model.config.id2label[top_idx.item()]
        
        used_fallback = False

        # --- SYSTEM FALLBACK --- #
        if confidence < 80:
            self.set_status(f"Fallback", color="orange")
            inputs_fb = self.fallback_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs_fb = self.fallback_model(**inputs_fb)
            probs_fb = torch.nn.functional.softmax(outputs_fb.logits, dim=-1)[0]
            top_prob_fb, top_idx_fb = torch.max(probs_fb, dim=0)
            conf_fb = top_prob_fb.item() * 100
            
            if conf_fb > confidence:
                confidence = conf_fb
                used_fallback = True
                self.set_status(f"Sukces Fallbacku! Zastąpiono wynik (Nowy wynik: {confidence:.1f}%)", color="lightgreen")
            else:
                self.set_status("Fallback nie poprawił oceny pewności.", color="lightcoral")

        # Aktualizacja Kafelka AI
        if used_fallback:
            self.pred_label.configure(text=f"Kategoria (FALLBACK!): {predicted_label}")
        else:
            self.pred_label.configure(text=f"Kategoria: {predicted_label}")
            if not used_fallback and confidence >= 80:
                self.set_status("Utrzymano pewny wynik pierwszego modelu.", color="lightgreen")
            
        self.conf_label.configure(text=f"Pewność: {confidence:.1f}%")
        self.progress_bar.set(confidence / 100.0)
        
        if confidence >= 85:
            self.progress_bar.configure(progress_color="green")
        elif confidence >= 60:
            self.progress_bar.configure(progress_color="orange")
        else:
            self.progress_bar.configure(progress_color="red")
            
        # 2. Silnik Decyzyjny JSON + LLM
        self.decision_engine(predicted_label, confidence, used_fallback)

    def write_recom(self, text, append=False):
        self.action_text.configure(state="normal")
        if not append:
            self.action_text.delete("0.0", "end")
        self.action_text.insert("end", text)
        self.action_text.configure(state="disabled")

    def decision_engine(self, label, confidence, used_fallback):
        matched_kb = None
        
        if "healthy" in label.lower() or "background" in label.lower():
            matched_kb = self.kb.get("Healthy")
        else:
            for kb_key, kb_data in self.kb.items():
                if kb_key.lower() in label.lower():
                    matched_kb = kb_data
                    break
                    
        if confidence < 60:
            err_txt = (
                "[PEWNOŚĆ ZBYT NISKA < 60%]"
            )
            self.write_recom(err_txt)
            return
            
        if matched_kb:
            # WIEDZA ZAKODOWANA LOKALNIE
            val = f"[ŹRÓDŁO: Baza Wiedzy]\n\nOpis Stanu:\n{matched_kb['description']}\n\nZalecana Czynność:\n{matched_kb['action']}"
            self.write_recom(val)
        else:
            # BRAK WIEDZY ZAKODOWANEJ - ODPALAMY POLSKIEGO LLM
            if self.llm_engine:
                notice = f"[Zidentyfikowano jednostkę chorobową]: {label}\n\Generowanie rekomendacji..."
                self.write_recom(notice)
                self.set_status("Przetwarzanie..", color="violet")
                
                self.thinking = True
                self.load_image_btn.configure(state="disabled")
                
                def bielik_task():
                    prompt = f"Opisz krótko (do 4 zdań) co to jest za choroba roślin: '{label}' i rekomendowane sposoby leczenia opryskami."
                    full_prompt = f"<|im_start|>system\nJesteś niezwykle precyzyjnym ekspertem ogrodnictwa. Odpowiadaj bezpośrednio i nie przekraczaj 4 zdań.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    
                    try:
                        # Zwiększony limit tokenów (max_tokens=600), inteligentne przerwanie przy kropce na końcu, 
                        # żeby zagwarantować spójne zdania
                        response = self.llm_engine(full_prompt, max_tokens=600, temperature=0.7, stop=["<|im_end|>"])
                        generated_text = response['choices'][0]['text'].strip()
                        
                        # Korzyść f-stringów w Pythonie (potrójnych) – rozwiązanie problemu uciętych wierszy na poziomie jądra Pythona!
                        final_msg = f"""
                        


Porada:
{generated_text}
"""
                        
                        self.after(0, self.write_recom, final_msg, True)
                        self.after(0, self.set_status, "Sukces", "lightgreen")
                    except Exception as err:
                        self.after(0, self.write_recom, f"\n\nBłąd: {err}", True)
                        self.after(0, self.set_status, "Błąd", "red")
                    finally:
                        self.thinking = False
                        self.after(0, lambda: self.load_image_btn.configure(state="normal"))
                        
                threading.Thread(target=bielik_task, daemon=True).start()
                
            else:
                out = (
                    f"BRAK DANYCH NA TEMAT:\n{label}\n\n"
                )
                self.write_recom(out)

if __name__ == "__main__":
    app = PlantDiseaseApp()
    app.mainloop()
