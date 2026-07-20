from rag_chain import ask

while True:

    q = input("\nQuestion: ")

    if q == "exit":
        break

    print()

    print(ask(q))
