#! /usr/bin/env python

import argparse
import sys
import logging
import logging.config

import yaml
from cellmaps_utils import logutils
from cellmaps_utils import constants
import cellmaps_vnn
from cellmaps_vnn.annotate import VNNAnnotate
from cellmaps_vnn.exceptions import CellmapsvnnError
from cellmaps_vnn.predict import VNNPredict
from cellmaps_vnn.runner import CellmapsvnnRunner, SLURMCellmapsvnnRunner
from cellmaps_vnn.train import VNNTrain

logger = logging.getLogger(__name__)


def _parse_arguments(desc, args):
    """
    Parses command line arguments

    :param desc: description to display on command line
    :type desc: str
    :param args: command line arguments usually :py:func:`sys.argv[1:]`
    :type args: list
    :return: arguments parsed by :py:mod:`argparse`
    :rtype: :py:class:`argparse.Namespace`
    """
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=constants.ArgParseFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Command to run. Type <command> -h for more help')
    subparsers.required = True

    VNNTrain.add_subparser(subparsers)
    VNNPredict.add_subparser(subparsers)
    VNNAnnotate.add_subparser(subparsers)
    parser.add_argument('--logconf', default=None,
                        help='Path to python logging configuration file in '
                             'this format: https://docs.python.org/3/library/'
                             'logging.config.html#logging-config-fileformat '
                             'Setting this overrides -v parameter which uses '
                             ' default logger. (default None)')
    parser.add_argument('--exitcode', help='Exit code this command will return',
                        default=0, type=int)
    parser.add_argument('--skip_logging', action='store_true',
                        help='If set, output.log, error.log '
                             'files will not be created')
    parser.add_argument('--provenance',
                        help='Path to file containing provenance '
                             'information about input files in JSON format. '
                             'This is required and not including will output '
                             'and error message with example of file')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                        help='Increases verbosity of logger to standard '
                             'error for log messages in this module. Messages are '
                             'output at these python logging levels '
                             '-v = WARNING, -vv = INFO, '
                             '-vvv = DEBUG, -vvvv = NOTSET (default ERROR '
                             'logging)')
    parser.add_argument('--version', action='version',
                        version=('%(prog)s ' +
                                 cellmaps_vnn.__version__))

    return parser.parse_args(args)


def main(args):
    """
    Main entry point for program

    :param args: arguments passed to command line usually :py:func:`sys.argv[1:]`
    :type args: list

    :return: return value of :py:meth:`cellmaps_vnn.runner.CellmapsvnnRunner.run`
             or ``2`` if an exception is raised
    :rtype: int
    """
    desc = """
    Version {version}

    Invokes run() method on CellmapsvnnRunner

    """.format(version=cellmaps_vnn.__version__)
    theargs = _parse_arguments(desc, args[1:])
    theargs.program = args[0]
    theargs.version = cellmaps_vnn.__version__

    if theargs.command == VNNTrain.COMMAND or theargs.command == VNNPredict.COMMAND:
        if theargs.config_file is not None:
            with open(theargs.config_file, "r") as file:
                config = yaml.safe_load(file)

                for key, value in config.items():
                    if hasattr(theargs, key):
                        setattr(theargs, key, value)

        required_args = ['cell2id', 'mutations', 'cn_deletions', 'cn_amplifications']
        if theargs.command == VNNTrain.COMMAND:
            required_args.append('gene2id')
            required_args.append('training_data')
        else:
            required_args.append('predict_data')
        for arg in required_args:
            if getattr(theargs, arg) is None:
                raise CellmapsvnnError(f"The argument --{arg} is required either in command line or config file.")

    try:
        logutils.setup_cmd_logging(theargs)

        if theargs.command == VNNTrain.COMMAND:
            cmd = VNNTrain(theargs)
        elif theargs.command == VNNPredict.COMMAND:
            cmd = VNNPredict(theargs)
        elif theargs.command == VNNAnnotate.COMMAND:
            cmd = VNNAnnotate(theargs)
            theargs.inputdir = theargs.model_predictions
        else:
            raise Exception('Invalid command: ' + str(theargs.command))

        if theargs.slurm:
            use_gpu = True if (theargs.command != VNNAnnotate.COMMAND and theargs.use_gpu) else False
            slurm_partition = 'nrnb-gpu' if (theargs.slurm_partition is None and use_gpu) else theargs.slurm_partition
            slurm_account = 'nrnb-gpu' if (theargs.slurm_account is None and use_gpu) else theargs.slurm_account

            runner = SLURMCellmapsvnnRunner(outdir=theargs.outdir,
                                            command=cmd,
                                            args=theargs,
                                            gpu=use_gpu,
                                            slurm_partition=slurm_partition,
                                            slurm_account=slurm_account,
                                            input_data_dict=theargs.__dict__
                                            )
        else:
            runner = CellmapsvnnRunner(outdir=theargs.outdir,
                                       command=cmd,
                                       inputdir=theargs.inputdir,
                                       exitcode=theargs.exitcode,
                                       skip_logging=theargs.skip_logging,
                                       input_data_dict=theargs.__dict__)

        return runner.run()
    except Exception as e:
        logger.exception('Caught exception: ' + str(e))
        return 2
    finally:
        logging.shutdown()


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main(sys.argv))
