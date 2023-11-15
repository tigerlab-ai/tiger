import os
import openai
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai_api_key = os.environ.get('OPENAI_API_KEY')

def get_text_completion(prompt, model):
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=150  # You can adjust the max_tokens as needed
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error getting text completion for model {model}: {e}")
        return None

def process_file(filename):
    models = [
        "ada", 
        "text-davinci-003", 
        "text-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
    ]  # Add other models as needed
    responses = {model: [] for model in models}

    with open(filename, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                prompt = data['prompt']
                logging.info(f"Processing prompt: {prompt}")

                for model in models:
                    response = get_text_completion(prompt, model)
                    if response is not None:
                        responses[model].append({"input_prompt": prompt, "output_response": response})
                    else:
                        logging.warning(f"No response received for model {model} and prompt {prompt}")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from line: {e}")

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    for model, model_responses in responses.items():
        try:
            output_filename = f'data/{base_filename}_{model}_responses.json'
            with open(output_filename, 'w') as outfile:
                json.dump(model_responses, outfile, indent=4)
            logging.info(f"Successfully wrote responses to {output_filename}")
        except IOError as e:
            logging.error(f"Error writing to file {output_filename}: {e}")

def main():
    logging.info("Starting processing...")
    input_files = [
        '../evaluations/data/hate.txt', 
        '../evaluations/data/violence.txt', 
        '../evaluations/data/sexual.txt', 
        '../evaluations/data/harassment.txt', 
        '../evaluations/data/self-harm.txt'
    ]
    for file in input_files:
        logging.info(f"Processing file: {file}")
        process_file(file)
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
