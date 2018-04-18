from tfrddlsim.parser import RDDLParser

import sys

if __name__ == '__main__':

    # read RDDL file
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        rddl = f.read()

    # build parser
    parser = RDDLParser()
    parser.build()

    # parse RDDL
    rddl = parser.parse(rddl)
    domain = rddl.domain
    non_fluents = rddl.non_fluents
    instance = rddl.instance

    # compile RDDL to TensorFlow
    # TODO
