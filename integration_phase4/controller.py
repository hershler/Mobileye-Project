from tfl_man import TFL_Man


class Controller:

    def __init__(self, pls_file):
        with open(pls_file, 'r') as pls_file:
            pkl_path = pls_file.readline()
            self.__frame_list = pls_file.readlines()

        self.__tfl_manager = TFL_Man(pkl_path)


    def start(self):
        for frame in self.__frame_list:
            self.__tfl_manager.run(frame)
