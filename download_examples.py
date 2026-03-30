import os
import requests

EXAMPLES_DIR = "examples"

images = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/46/Early_blight_on_tomato_leaf.JPG",  # Pomidor - zaraza
        "filename": "tomato_early_blight.jpg"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Apple_scab_symptoms_on_apple_leaves.jpg/800px-Apple_scab_symptoms_on_apple_leaves.jpg",  # Jabłoń - parcha
        "filename": "apple_scab.jpg"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Tomato_leaf.jpg",  # Zdrowy liść pomidora
        "filename": "healthy_tomato.jpg"
    }
]

def main():
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    
    headers = {
        "User-Agent": "PlantDiseaseDemo/1.0 (https://github.com/example/plant-disease-demo; myemail@example.com) python-requests/2.32"
    }

    print("Pobieranie zdjęć przykładowych...")
    for img in images:
        path = os.path.join(EXAMPLES_DIR, img["filename"])
        # Wymuszamy pobranie by nadpisać zablokowane pliki "HTML Error"
        with open(path, "wb") as f:
            f.write(requests.get(img["url"], headers=headers).content)
        print(f"Pobrano {img['filename']}!")
            
    print("Gotowe. Zdjęcia znajdziesz w folderze /examples/")

if __name__ == "__main__":
    main()
