from colorama import Fore, Back, Style, init


def info_log(msg):
    init(autoreset=True)
    print(Fore.CYAN + "[INFO]", end=" ")
    print(msg)
    init(autoreset=True)


def output_log(msg):
    init(autoreset=True)
    print(Fore.GREEN + "[OUTPUT]", end=" ")
    print(msg)
    init(autoreset=True)


def warning_log(msg):
    init(autoreset=True)
    print(Fore.BLUE + "[WARNING]", end=" ")
    print(msg)
    init(autoreset=True)


def error_log(msg):
    init(autoreset=True)
    print(Fore.RED + "[ERROR]", end=" ")
    print(msg)
    init(autoreset=True)
