class Document(object):
    '''
    Documents characterized by features used for learing to rank.
    A relevance grade is assigned for each document.
    '''
    def __init__(self, rel, qid, features):
        '''
        rel: a relevance grade (positive integer)
        qid: a query ID (positive integer)
        features: a dict of { feature: value },
            where `feature` is a positive integer.
        '''
        self.rel = rel
        self.qid = qid
        self.features = features

    @classmethod
    def readline(cls, line):
        '''
        Read a document in the line format:
            <line>    .=. <target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
            <target>  .=. <positive integer>
            <qid>     .=. <positive integer>
            <feature> .=. <positive integer>
            <value>   .=. <float>
            <info>    .=. <string>
        '''
        if "#" in line:
            index = line.index("#")
            line = line[:index]
        ls = [l.strip() for l in line.split(" ")]
        rel = int(ls[0])
        qid = ls[1].split(":")[1]
        features = [l for l in ls[2:] if len(l) > 0]
        features = [tuple(f.split(":")) for f in features]
        features = {int(i): float(v) for i, v in features}
        result = Document(rel, qid, features)
        return result
