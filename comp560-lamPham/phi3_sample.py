from phi3_lib import Phi3Config, Phi3Local


def ask_math(question: str, system_prompt: str = "Ban la tro ly toan hoc ngan gon.") -> str:
    cfg = Phi3Config(
        use_4bit=True,
        max_new_tokens=120,
        temperature=0.0,
    )
    llm = Phi3Local(cfg)
    return llm.chat(question, system_prompt=system_prompt)


def main():
    question = "Tinh 123456789 + 987654321 va chi tra loi ket qua."
    answer = ask_math(question)

    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
