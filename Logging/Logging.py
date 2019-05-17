from colorama import Fore, Back, Style, init


def info_log(msg):
    print(Fore.CYAN + "[INFO]", end=" ")
    print(msg)


def output_log(msg):
    print(Fore.GREEN + "[OUTPUT]", end=" ")
    print(msg)


def warning_log(msg):
    print(Fore.BLUE + "[WARNING]", end=" ")
    print(msg)


def error_log(msg):
    print(Fore.RED + "[ERROR]", end=" ")
    print(msg)
