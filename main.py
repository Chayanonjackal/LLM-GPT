from transformers import pipeline
import argparse
import os

# Initialize the text generation pipeline with a pre-trained GPT-2 model
# generator = pipeline('text-generation', model='gpt2')
# Initialize the text generation pipeline with a GPT-3 comparable model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

def generate_text(prompt, max_length=100):
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']

def main(text):
    # Example usage
    prompt = text
    generated_text = generate_text(prompt)
    print(generated_text)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Text-to-Speech using SpeechT5")
    parser.add_argument('--text', type=str, required=True, help="Text to convert to speech or path to a text file")

    args = parser.parse_args()

    if os.path.isfile(args.text):
        with open(args.text, 'r', encoding='utf-8') as file:
            text = file.read()

    else:
        text = args.text

    main(text)
