from llama_cpp import Llama
import copy


class Tokenizer:
    def __init__(self):
        pass

    def encode_system_prompt(self, prompt):
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{ prompt }<|eot_id|>"

    def encode_user_message(self, message):
        return f"<|start_header_id|>user<|end_header_id|>{ message }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    def encode_assistant(self, model_answer):
        return f"{ model_answer }<|eot_id|>"

    def encode(self, messages):
        encoded_parts = []
        for message in messages:
            content = message["content"]
            if message["role"] == "system":
                encoded_parts.append(self.encode_system_prompt(content))
            elif message["role"] == "user":
                encoded_parts.append(self.encode_user_message(content))
            elif message["role"] == "assistant":
                encoded_parts.append(self.encode_assistant(content))
            else:
                raise ValueError("Invalid role in message.")
        return "".join(encoded_parts)


class ChatCompletion:
    def __init__(self, messages, llm):
        self.messages = messages
        self.llm = llm
        self.tokenizer = Tokenizer()
        self.init_messages = copy.deepcopy(messages)

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def clear_chat(self):
        self.messages = copy.deepcopy(self.init_messages)

    def send_message(self, user_input):
        if user_input.lower() == "clear chat":
            self.clear_chat()
            print("Chat cleared.")
        else:
            self.add_message("user", user_input)
            prompt = self.tokenizer.encode(self.messages)
            print(
                "###############################################################################################################################"
            )
            print("Prompt: ", prompt)
            print("Promptlänge: ", len(prompt))
            print(
                "###############################################################################################################################"
            )

            response = self.llm(
                prompt,  # Prompt
                max_tokens=8096,  # Maximum number of tokens to generate
                echo=False,  # Echo the prompt back in the output
                temperature=1,  # Randomness in Boltzmann distribution
            )

            bot_response = response["choices"][0]["text"]
            self.add_message("assistant", bot_response)
            print(bot_response)
            print(
                "###############################################################################################################################"
            )


class ChatBot:
    def __init__(self, messages, llm):
        self.chat = ChatCompletion(messages, llm)

    def run(self):
        try:
            while True:
                user_input = input("You: ")
                self.chat.send_message(user_input)
        except KeyboardInterrupt:
            print("Chatbot terminated by user.")


def main():
    messages = [
        {
            "role": "system",
            "content": "Your are a helpful assistant. Your language to answer is German.",
        },
    ]
    # Instanciate the model
    llm = Llama(
        model_path="models/your_model.gguf",
        verbose=True,
        n_threads=8,
        n_gpu_layers=-1,
        n_ctx=8096,
    )

    chat_bot = ChatBot(messages, llm)
    chat_bot.run()


if __name__ == "__main__":
    main()
