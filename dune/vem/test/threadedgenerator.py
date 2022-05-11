from concurrent.futures import ThreadPoolExecutor
class ThreadedGenerator:
    def __init__(self, maxWorkers):
        self.maxWorkers = maxWorkers
        self.builders = []
    def add(self, builder, *args, **kwargs):
        self.builders += [(builder,args,kwargs)]
    def clear(self):
        self.builders = []
    def execute(self):
        with ThreadPoolExecutor(max_workers = 4) as executor:
            thread = executor.map(ThreadedGenerator._gen, self.builders)
        self.clear()
        return [m for m in thread]
    @staticmethod
    def _gen(arg):
        builder = arg[0]
        args = arg[1]
        kwargs = arg[2]
        if builder:
            ret = builder(*args,**kwargs)
        else:
            ret = None
        # print("Building", builder, ret, flush=True)
        return ret

def main():
    def s(x,y):
        return x+y
    def m(x,y):
        return x*y
    def axpy(a,x,*,y):
        return y+a*x
    g = ThreadedGenerator(4)
    g.add(s,x=1,y=2)
    g.add(m,2,2)
    g.add(None) # this is needed to allow for a conditional module generation
    g.add(axpy,3,2,y=2)
    m = g.execute()
    print(m)

if __name__ == "__main__":
    main()
