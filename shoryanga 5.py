import speech_recognition as sr
import requests
from transformers import CLIPTextModel, CLIPTokenizer, DALL_E
from PIL import Image
from io import BytesIO

def audio_to_text():
    # Initialize recognizer class in SpeechRecognition library
    recognizer = sr.Recognizer()

    # Capture audio from the microphone
    with sr.Microphone() as source:
        print("Listening for audio prompt...")
        audio = recognizer.listen(source)

    try:
        # Convert speech to text
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"Audio converted to text: {text}")
        return text
    except Exception as e:
        print(f"Error recognizing speech: {e}")
        return None

def text_to_image(prompt):
    # Load the pre-trained DALL·E model from Hugging Face
    model = DALL_E.from_pretrained("dalle-mini/dalle-mini/mega-1-fp16:latest")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate the image from the prompt using the model
    with torch.no_grad():
        generated_image = model.generate(**inputs)
    
    # Convert the generated image to PIL format
    pil_image = Image.fromarray(generated_image[0].cpu().numpy())
    return pil_image

def save_image(image, filename="generated_image.png"):
    # Save the generated image
    image.save(f"assets/{filename}")
    print(f"Image saved as assets/{filename}")

def main():
    text_prompt = audio_to_text()
    if text_prompt:
        generated_image = text_to_image(text_prompt)
        save_image(generated_image)
    else:
        print("No valid text prompt received.")

if __name__ == "__main__":
    main()
