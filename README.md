# IWantToSay

# DungeonFortress Chatbot

DungeonFortress is a chatbot built using the Langchain framework and the Anthropic API. It allows users to interact with a character named Thildiriel, a wise and powerful Moon Elf druid, who can engage in general conversation and answer specific questions based on the provided context.

## Features

- Supports two types of responses:
  - Conversation chain (CONV): For general conversational questions, greetings, or creative responses.
  - Retrieval-augmented-generation chain (RAG): For specific information, memories, or events related to the context.
- Uses the Anthropic API to decide the appropriate response type based on the user's query.
- Stores conversation history in JSON files and uses Chroma vector store for efficient retrieval.
- Utilizes the Ollama language model for generating responses.

## Prerequisites

- Python 3.x
- Langchain framework
- Local ollama install

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/DungeonFortress.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the `switcher.py` script:
   ```
   python switcher.py
   ```

2. Enter a query when prompted. The chatbot will determine the appropriate response type (CONV or RAG) based on the query and generate a response accordingly.

3. The conversation history will be stored in JSON files in the `memory/druid` directory.

## Customization

- To modify the character's background and personality, update the `template` variable in the code.
- To change the language model, update the `model` variable and ensure you have the necessary dependencies installed.
- Adjust the `decide_response_type` function to fine-tune the decision-making process for selecting the response type.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Langchain](https://github.com/hwchase17/langchain) - The framework used for building the chatbot.
- [Anthropic](https://www.anthropic.com/) - The API used for decision-making and language modeling.

Feel free to customize and expand upon this README based on your specific project details and requirements.
