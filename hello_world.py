#!/usr/bin/env python3

import sys, time


def fprint(text):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)
    sys.stdout.write("\n")


def main():
    try:
        with open('hello_world.txt', 'r') as file:
            for line in file:
                fprint(line.strip())
    except FileNotFoundError:
        print("Error: hello_world.txt not found")


if __name__ == "__main__":
    main()
