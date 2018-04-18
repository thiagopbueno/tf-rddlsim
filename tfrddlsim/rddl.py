class RDDL(object):

    def __init__(self, blocks):
        self.blocks = blocks
        self.__dict__.update(blocks)


class Domain(object):

    def __init__(self, name, requirements, sections):
        self.name = name
        self.requirements = requirements
        self.__dict__.update(sections)


class Instance(object):

    def __init__(self, name, sections):
        self.name = name
        self.__dict__.update(sections)


class NonFluents(object):

    def __init__(self, name, sections):
        self.name = name
        self.__dict__.update(sections)
