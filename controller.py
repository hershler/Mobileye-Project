from typing import List

from tfl_manager import TFLMan


def remove_endl(line: str):
    return line[:-1] if line and '\n' == line[-1] else line


class Controller:
    def __init__(self, pls_file) -> None:
        with open(pls_file, 'r') as pls_file:
            pkl_path: str = pls_file.readline()[:-1]
            lines: List[str] = pls_file.readlines()
            print(remove_endl(lines[0]))
            self.__frame_list: List[str] = lines[1:]

        self.__tfl_manager = TFLMan(pkl_path)

    def start(self) -> None:
        for i, frame in enumerate(self.__frame_list):
            if frame[-1] == '\n':
                frame = frame[:-1]
            self.__tfl_manager.run(frame, i)
