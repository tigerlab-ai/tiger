import os

import openai


class OpenAIQueryAugmenter:
    def __init__(self, openai_text_model: str) -> None:
        self.openai_api_text_model = openai_text_model
        self._initialize_openai_api()

    def _initialize_openai_api(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set!")

    def get_augmented_query(self, query: str) -> str:
        prompt = f"Expand on the query: {query}"
        response = openai.Completion.create(engine=self.openai_api_text_model, prompt=prompt, max_tokens=100)
        augmented_query = response.choices[0].text.strip()
        return augmented_query
