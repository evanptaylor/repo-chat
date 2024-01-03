import repochat as rc
import openai
import termcolor
import time
import os 

def verify_api_key(api_key):
    openai.api_key = api_key
    try:
        openai.models.list()
    except openai.AuthenticationError as e:
        print(termcolor.colored(f"Invalid API key", 'light_red', attrs=["bold"]) + "\nGrab your API key from: "+termcolor.colored(f"https://platform.openai.com/account/api-keys", 'light_blue', attrs=["underline"]))
        exit()
    else:
        print(termcolor.colored(f"API key is valid", 'light_green', attrs=["bold"]))

def print_by_char(start_string, text):
    print(start_string, end=" ")
    for char in text:
        print(termcolor.colored(char, 'green'), end='', flush=True)
        time.sleep(0.01)
    print()

def main():
    print(termcolor.colored(f'=========REPO CHAT=========', attrs=["bold"]))

    api_key = input("Input OpenAI Key: ")
    verify_api_key(api_key)
    time.sleep(1)
    
    print('\n\n')

    repo_link = input('Input repository link: ')
    print('LOADING...')

    repo = rc.RepoChat(repo_link)
    repo.preprocess()
    repo.embed()
    os.system('clear')

    print(termcolor.colored(f'=========REPO CHAT=========', attrs=["bold"]))
    print('type q to quit')

    while True:
        question = input('Question: ')
        if question in ['q', 'quit', 'exit']:
            break
        ans = repo.ask(question)
        print_by_char('>>> ', ans)

main()