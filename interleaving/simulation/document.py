class Document(object):

    def __init__(self, rel, qid, features):
        self.rel = rel
        self.qid = qid
        self.features = features

    @classmethod
    def readline(cls, line):
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
