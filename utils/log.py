from utils.misc import now

class Logger:
    def __init__(self, filename):
        self.filename = filename
        
        try:
            self.file = open(self.filename, 'a')
            self.file.close()
        except:
            raise ValueError('Cannot open file {}'.format(self.filename))

    def append(self, message):
        """
        Log message to file
        :param message: message to log, string
        """

        with open(self.filename, 'a') as log_file:
            log_file.write(now() + '\t' + message + '\n')

    def abort(self):
        with open(self.filename, 'a') as log_file:
            log_file.write(now() + '\t' + 'EXPERIMENT ABORTED' + '\n')


    def terminate(self):
        with open(self.filename, 'a') as log_file:
            log_file.write(now() + '\t' + 'EXPERIMENT TERMINATED' + '\n')