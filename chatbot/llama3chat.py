from llama_cpp import Llama


class Tokenizer:
    def __init__(self):
        self.forbidden_characters = ""

    def encode_system_prompt(self, prompt):
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{ prompt }<|eot_id|>"

    def encode_user_message(self, message):
        return f"<|start_header_id|>user<|end_header_id|>{ message }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    def encode_assistant(self, model_answer):
        return f"{ model_answer }<|eot_id|>"

    def clean_text(self, text):

        return "".join(char for char in text if char not in self.forbidden_characters)

    def encode(self, messages):

        prompt = ""
        for message in messages:
            content = self.clean_text(message["content"])
            if message["role"] == "system":
                prompt += self.encode_system_prompt(content)
            elif message["role"] == "user":
                prompt += self.encode_user_message(content)
            elif message["role"] == "assistant":
                prompt += self.encode_assistant(content)
            else:
                raise ValueError("Invalid role in message.")
        return prompt


class ChatCompletion:
    def __init__(self, messages, llm):
        self.messages = messages
        self.llm = llm
        self.tokenizer = Tokenizer()

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def clear_chat(self):
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistiant. Your language to answer is German.",
            }
        ]

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

            response = llm(
                prompt,  # Prompt
                max_tokens=256,  # Maximum number of tokens to generate
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


messages = [
    {
        "role": "system",
        "content": "Your are a helpful assistant. Your language to answer is German.",
    },
]
# Instanciate the model
llm = Llama(
    model_path="models/Meta-Llama-3-8B-Instruct.Q5_k_m_with_temp_stop_token_fix.gguf",
    verbose=True,
    n_threads=8,
    n_gpu_layers=-1,
    n_ctx=8096,
)

chat_bot = ChatBot(messages, llm)
chat_bot.run()
