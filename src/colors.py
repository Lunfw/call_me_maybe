from pydantic import BaseModel, Field
from shutil import get_terminal_size
from sys import stdout
from typing import IO, List, Callable, Any


class Colors(BaseModel):
    '''
        A small library for color codes and such.
    '''
    RED: str = Field(default='\033[1;31m')
    GREEN: str = Field(default='\033[1;32m')
    YELLOW: str = Field(default='\033[1;33m')
    BLUE: str = Field(default='\033[1;34m')
    PURPLE: str = Field(default='\033[1;35m')
    CYAN: str = Field(default='\033[1;36m')
    NONE: str = Field(default='\033[0m')
    RESET: str = Field(default='\033[0m')

    WHITE: str = Field(default='\033[1;37m')
    BLACK: str = Field(default='\033[1;30m')
    GREY: str = Field(default='\033[1;90m')

    MAGENTA: str = Field(default='\033[38;5;198m')
    LIME: str = Field(default='\033[38;5;154m')
    MAROON: str = Field(default='\033[38;5;88m')
    ORANGE: str = Field(default='\033[38;5;214m')
    DARKRED: str = Field(default='\033[38;5;52m')
    CRIMSON: str = Field(default='\033[38;5;160m')
    BROWN: str = Field(default='\033[38;5;130m')
    GOLD: str = Field(default='\033[38;5;220m')
    VIOLET: str = Field(default='\033[38;5;135m')
    DARKGREY: str = Field(default='\033[38;5;240m')

    @property
    def RAINBOW(self) -> Callable[[List[str] | str], str | List[str]]:
        def _rainbow(message: (List[str] | str)) -> str | List[str]:
            ex = {'RESET', 'NONE', 'BLACK', 'WHITE', 'GREY', 'DARKGREY'}
            colors = [k for k in Colors.model_fields.keys() if k not in ex]
            if (type(message) is str):
                result: str = ''
                for i, char in enumerate(message):
                    result += getattr(self, colors[i % len(colors)]) + char
                return (result + self.RESET)
            Lresult: List[str] = []
            for line in message:
                temp = ''
                for i, char in enumerate(line):
                    temp += getattr(self, colors[i % len(colors)]) + char
                Lresult.append(temp + self.RESET)
            return (Lresult)
        return (_rainbow)

    def get_colors(self) -> List[str]:
        exclude = ['RESET']
        temp: List[str] = ['RAINBOW']
        temp += [i for i in Colors.model_fields.keys() if i not in exclude]
        return (temp)


class Format(BaseModel):
    '''
        Small lib for text formatting.
    '''
    colors: Colors = Field(default=Colors())

    @staticmethod
    def centered(text: (List[str] | str), fd: IO[str] = stdout) -> None:
        width = get_terminal_size().columns
        if (type(text) is str):
            print(text.center(width), file=fd, flush=True)
            return (None)
        for line in text:
            print(line.center(width), file=fd, flush=True)

    @staticmethod
    def putstr(text: (List[str] | str), fd: IO[str] = stdout) -> None:
        if (type(text) is str):
            print(text, file=fd)
            return
        for word in text:
            print(word, file=fd)

    @staticmethod
    def listing(text: (List[str] | str),
                marker: str = ' ') -> str | List[str]:
        temp: List[str] = []
        if (type(text) is str):
            return (f'[{marker}]    -> ' + text)
        max_len: int = len(text[0])
        for line in text:
            if (max_len < len(line)):
                max_len = len(line)
        for line in text:
            while (len(line) < max_len):
                line += ' '
            temp.append(f'[{marker}]    ->  ' + line)
        return (temp)

    @staticmethod
    def ls(text: (List[str] | str)) -> str | List[str]:
        temp: List[str] = []
        if (type(text) is str):
            return (text)
        for line in range(len(text)):
            if (line < len(text)):
                temp.append(text[line])
            else:
                temp.append(text[line])
        return (temp)

    @staticmethod
    def colored(text: (List[str] | str), color: str) -> Any:
        color = color.upper()
        colors = Colors()
        if (color == 'RAINBOW'):
            return (colors.RAINBOW(text))
        if (type(text) is str):
            return (getattr(Colors(), color) + text + Colors().RESET)
        for word in text:
            word = getattr(Colors(), color) + word + Colors().RESET
        return (text)

    def draw_margin(self) -> None:
        width = get_terminal_size().columns
        margin = '\033[1;90m' + '|< '
        for i in range(width - 6):
            margin += '='
        margin += ' >|' + '\033[0m'
        self.centered(margin)
