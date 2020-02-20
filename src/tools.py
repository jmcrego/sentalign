import logging
import sys

def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        sys.stderr.write("Invalid log level={}\n".format(loglevel))
        sys.exit()

    if logfile is None:
        logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.info('Created Logger level={}'.format(loglevel))
    else:
        logging.basicConfig(filename=logfile, format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=numeric_level)
        logging.info('Created Logger level={} file={}'.format(loglevel, logfile))







