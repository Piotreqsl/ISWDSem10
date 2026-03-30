# \finetune.py
# ---
# UWAGA: TEN SKRYPT JEST POKAZOWY (NA POTRZEBY RAPORTU / PREZENTACJI PROCESU).
# Zademonstrowano w nim mechanikę użycia nowej bazy (np. zbioru in-the-wild "PlantDoc") 
# i Fine-Tuningu popularnej architektury bazowej do odróżniania chorób.
# Aby uruchomić skrypt, komputer musiałby posiadać kartę graficzną NVIDIA (CUDA) i kilka GB VRAM.

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

def main():
    print("Inicjalizacja środowiska treningowego...")

    # 1. Pobieranie trudniejszego zestawu (np. paczka zgłoszona z Kaggle załadowana na HF)
    # W praktyce podmieniamy "plantvillage" na nasz dataset.
    print("Pobieranie wirtualnego datasetu 'in-the-wild'...")
    dataset = load_dataset("beans") # Dla szybkiej symulacji kodu ładujemy zbiór fasoli. W projekcie wpisz: "plantdoc"
    
    # 2. Definicja etykiet i etykietownika
    labels = dataset["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
        
    # 3. Import Fundamentu (Bazowej Sieci do douczenia - Transfer Learning)
    model_checkpoint = "google/mobilenet_v2_1.0_224"
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    
    # Inicjujemy głowę modelu nowymi etykietami!
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # Wyrzuca oryginalną klasyfikację na 1000 klas i uczy na nasze K-klas!
    )
    
    # 4. Augmentacja danych (bardzo ważne przy wchodzeniu w analizę z szumem tła)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.with_transform(transforms)
    
    # 5. Konfiguracja Epok, Batcha i Hiperparametrów
    training_args = TrainingArguments(
        output_dir="./my-finetuned-plant-model",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        num_train_epochs=3, # Dla prawdziwego "PlantDoc" polecamy ok. 25-50 epok
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    import numpy as np
    import evaluate
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # 6. Odpalenie Pętli Treningowej
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    print("Rozpoczęcie treningu (Douczania)...")
    # trainer.train() 
    # Odkomentuj linię wyżej, by realnie zacząć uczyć sieć (uwaga, zajmie to czas i zasoby!)

    print("Zakończono. Model zostaje zrzucony do folderu 'my-finetuned-plant-model'. Można go ująć w app.py")

if __name__ == "__main__":
    main()
